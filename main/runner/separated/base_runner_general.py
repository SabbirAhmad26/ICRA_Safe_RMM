    
import time
# import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from main.algorithms.utils.separated_buffer import SeparatedReplayBuffer
from main.algorithms.utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.random_mapping_agents = self.all_args.random_mapping_agents
        self.discretize = self.all_args.discretize

        if self.env_name == 'CARLA':
            self.cav_force_straight_step = self.all_args.cav_fs_step

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            # self.save_dir = str(wandb.run.dir)
            pass
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if self.algorithm_name == "wmappo":
            from main.algorithms.r_mappo.r_wocar_mappo import R_WOCAR_MAPPO as TrainAlgo
            from main.algorithms.r_mappo.algorithm.r_Wocar_MAPPOPolicy import R_WOCAR_MAPPOPolicy as Policy
        elif self.algorithm_name == "rmappo" or self.algorithm_name == "mappo":
            from main.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from main.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        elif self.algorithm_name == "maa2c":
            from main.algorithms.r_mappo.r_maa2c import R_MAA2C as TrainAlgo
            from main.algorithms.r_mappo.algorithm.rMAA2CPolicy import R_MAA2CPolicy as Policy
        else:
            raise NotImplementedError

        self.policy = []
        for agent_id in range(self.num_agents):
            # for shared_policy
            if self.use_centralized_V: 
                share_observation_space = self.envs.share_observation_space[agent_id] 
                # share_observation_space = self.envs.observation_space[0]
            else: 
                share_observation_space = self.envs.observation_space[agent_id]

            # share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

        if self.env_name == 'CARLA':
            self.envs.set_policies(self.policy)

        if not self.all_args.test_only:

            self.trainer = []
            self.buffer = []
            
            for agent_id in range(self.num_agents):
                # algorithm
                tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
                # buffer
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
                bu = SeparatedReplayBuffer(self.all_args,
                                           self.envs.observation_space[agent_id],
                                           share_observation_space,
                                           self.envs.action_space[agent_id])
                self.buffer.append(bu)
                self.trainer.append(tr)

        if self.model_dir is not None:
            #!!! here should load not only policies
            if self.all_args.test_only:
                self.restore(actor_only=True)
            else:
                self.restore()
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            if self.algorithm_name == "wmappo":
                policy_worstq = self.trainer[agent_id].policy.worst_Q
                torch.save(policy_worstq.state_dict(), str(self.save_dir) + "/worstq_agent" + str(agent_id) + ".pt")
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.save_dir) + "/vnrom_agent" + str(agent_id) + ".pt")

    def restore(self, actor_only=False):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            if not actor_only:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
                if self.algorithm_name == "wmappo":
                    policy_worstq_state_dict = torch.load(str(self.model_dir) + '/worstq_agent' + str(agent_id) + '.pt')
                    self.policy[agent_id].worst_Q.load_state_dict(policy_worstq_state_dict)
                    self.policy[agent_id].target_worst_Q.load_state_dict(policy_worstq_state_dict)
                if self.trainer[agent_id]._use_valuenorm:
                    policy_vnrom_state_dict = torch.load(str(self.model_dir) + '/vnrom_agent' + str(agent_id) + '.pt')
                    self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)
        print("Loading successfully from: {}".format(self.model_dir))

    def load_policy(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    # wandb.log({agent_k: v}, step=total_num_steps)
                    pass
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    # wandb.log({k: np.mean(v)}, step=total_num_steps)
                    pass
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
