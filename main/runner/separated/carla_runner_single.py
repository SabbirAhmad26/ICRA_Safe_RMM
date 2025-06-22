    
import time
# import wandb
import os
import numpy as np
from itertools import chain
import torch

from main.algorithms.utils.util import update_linear_schedule
from main.runner.separated.base_runner_single import Runner
from main.envs.carla.enums import Behavior
from main.envs.carla.control_0 import mpc_exec1, throttle_brake_mapping1, path_plannar

def _t2n(x):
    return x.detach().cpu().numpy()

class CARLARunner(Runner):
    def __init__(self, config):
        super(CARLARunner, self).__init__(config)
        self.vid_2_idx = None
        self.idx_2_vid = None
        self.safe_actions = None
        self.action_2_command = {0: "lane_keeping", 1: "left_lc", 2: "right_lc"}
        self.command_2_action = {"la": 0, "le": 1, "ri": 2} # "la"ne_keeping, "le"ft_lc, "ri"ght_lc
       
    def run(self):
        
        #episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episodes = self.all_args.episodes

        for episode in range(episodes):
            start = time.time()
            error_flag = False
            
            obs, combined_obs, all_car_info_dict = self.warmup()

            for agent_id in range(self.num_agents):
                # vid = self.idx_2_vid[agent_id]
                if self.use_centralized_V:
                    self.buffer[agent_id].share_obs[0] = combined_obs.copy()
                else:
                    self.buffer[agent_id].share_obs[0] = obs[:, agent_id].copy()
                self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            all_rewards = []
            flow_rwds = []
            cols_rwds = []
            dest_rwds = []
            sact_rwds = []
            
            # mpc-cbf default is lane_keeping
            command = "lane_keeping"
            previous_command = command
            reference_lane_index = -5
            lc_timeout_dict = {}

            for step in range(self.episode_length):
                conflicting_cars_state_degree1 = []
                conflicting_cars_state_degree2 = []
                preceding_cars_state = []

                # Sample actions
                # needs to fix: what if no safe action,
                # actions should always allow only normal actions
                # actions_env should contain Emergency_Stop if needed.
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, chosen_actions \
                    = self.collect(step, available_actions=self.safe_actions)
                
                assert len(actions_env) == self.num_agents
                
                for cavid in actions_env:
                    cav = self.envs.CAV_dict[cavid]

                    command = self.action_2_command[actions_env[cavid]]

                    if cavid in lc_timeout_dict and lc_timeout_dict[cavid] > 0:
                        command = previous_command
                        lc_timeout_dict[cavid] -= 1
                    elif command != "lane_keeping":
                        lc_timeout_dict[cavid] = self.all_args.timeout_lc

                    # switch back to lane keeping 
                    # flag1: lane change finished
                    # flag2: lane change aborted
                    desired_phi, reference_lane_index, flag1, flag2 = path_plannar(self.envs, cav, command, previous_command,
                                                                   reference_lane_index)

                for vid, info_dict in all_car_info_dict.items():
                    if vid in actions_env:
                        agent_state_dict = all_car_info_dict[vid]
                    else:
                        if all_car_info_dict[vid]["lane_id"] == all_car_info_dict[cavid]["lane_id"]:
                            if all_car_info_dict[vid]["x"] > all_car_info_dict[cavid]["x"]:
                                x_ip = all_car_info_dict[vid]["x"]
                                y_ip = all_car_info_dict[vid]["y"]
                                phi_ip = all_car_info_dict[vid]["phi"]
                                vel_ip = all_car_info_dict[vid]["vel"]/3.6
                                preceding_cars_state.append([x_ip, y_ip, phi_ip, vel_ip])
                            else:
                                x_ip = 1000
                                y_ip = all_car_info_dict[cavid]["y"]
                                phi_ip = 0
                                vel_ip = 100
                                preceding_cars_state.append([x_ip, y_ip, phi_ip, vel_ip])
                        else:
                            if all_car_info_dict[vid]["lane_id"] == reference_lane_index:
                                x_ic = all_car_info_dict[vid]["x"]
                                y_ic = all_car_info_dict[vid]["y"]
                                phi_ic = all_car_info_dict[vid]["phi"]
                                vel_ic = all_car_info_dict[vid]["vel"]/3.6
                                conflicting_cars_state_degree1.append([x_ic, y_ic, phi_ic, vel_ic])
                            else:
                                x_ic = all_car_info_dict[vid]["x"]
                                y_ic = all_car_info_dict[vid]["y"]
                                phi_ic = all_car_info_dict[vid]["phi"]
                                vel_ic = all_car_info_dict[vid]["vel"]/3.6
                                conflicting_cars_state_degree2.append([x_ic, y_ic, phi_ic, vel_ic])


                states = [agent_state_dict['x'], agent_state_dict['y'], \
                          agent_state_dict['phi'], agent_state_dict['vel']/3.6]
                status, action, next_states, model_acc = mpc_exec1(desired_phi, states, \
                                preceding_cars_state, conflicting_cars_state_degree1, \
                                conflicting_cars_state_degree2, 1000)

                a = float(action[0])
                throttle, brake = throttle_brake_mapping1(a)

                # Obser reward and next obs; infos is not necessary
                # !!! The shape of each item here might need adjustment
                # the true action could contains E_S, meantime the corresponding rewards should get penalized.
                obs_dict, rewards, done, all_car_info_dict, \
                    rwd_items, safe_actions, force_continue = \
                    self.envs.step({self.idx_2_vid[0]: [throttle, brake, action[1]]}, step_n=step)
                
                previous_command = command

                if force_continue:
                    error_flag = True
                    break

                all_rewards.append(rewards.copy())
                flow_rwds.append(rwd_items[0])
                cols_rwds.append(rwd_items[1])
                dest_rwds.append(rwd_items[2])
                sact_rwds.append(rwd_items[3])

                executed_actions = []
                true_actions = {}
                for agent_id in range(self.num_agents):
                    # executed_action = chosen_actions[agent_id]
                    executed_action = self.command_2_action[command[:2]]
                    true_actions[self.idx_2_vid[agent_id]] = executed_action
                    executed_actions.append([executed_action])
                executed_actions = np.expand_dims(np.array(executed_actions), axis=0)
                
                self.safe_actions = {self.vid_2_idx[vid]: safe_act for vid, safe_act in safe_actions.items()} 

                # --------- printing step-wise infomation --------- #
                self.print_stepwise_msg(actions_env, true_actions, all_car_info_dict, episode, step, rewards, rwd_items)
                # --------- printing step-wise infomation --------- #

                obs, combined_obs = self.prepare_obs(obs_dict)
                def prep_carla_data(rewards, done):
                    rewards = np.expand_dims(rewards, axis=0)
                    dones = np.expand_dims(np.array([done]*self.num_agents), axis=0)
                    return rewards, dones

                rewards, dones = prep_carla_data(rewards, done)
                #data = obs, rewards, dones, infos, values, actions, action_log_probs, \
                #       rnn_states, rnn_states_critic, combined_obs
                data = obs, rewards, dones, None, values, executed_actions, action_log_probs, \
                       rnn_states, rnn_states_critic, combined_obs
                
                # insert data into buffer
                self.insert(data)

            if error_flag:
                self.vid_2_idx = None
                self.idx_2_vid = None
                self.envs.close()
                print("Error in simulation, continue to next episode...")
                continue

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes))

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            for count, info in enumerate(infos):
                                if 'individual_reward' in infos[count][agent_id].keys():
                                    idv_rews.append(infos[count][agent_id].get('individual_reward', 0))
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                        print("Agent {}: average episode rewards is {}".format(agent_id, train_infos[agent_id]["average_episode_rewards"]))
                elif self.env_name == 'CARLA':
                    mean_episode_rewards = np.array(all_rewards).sum(0).mean()
                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update({"avg_eps_rewards": mean_episode_rewards})
                        train_infos[agent_id].update({"avg_eps_flow_rewards": sum(flow_rwds)})
                        train_infos[agent_id].update({"avg_eps_dest_rewards": sum(dest_rwds)})
                        train_infos[agent_id].update({"avg_eps_cols_rewards": sum(cols_rwds)})
                        train_infos[agent_id].update({"avg_eps_sact_rewards": sum(sact_rwds)})
                    print("Episode {:d} any collision {}; has reward {:.3f}, flow {:.3f}, cols {:.3f}, dest {:.3f}, sact {:.3f}, takes time {:.2f} minutes."\
                        .format(episode, str(self.envs.done_collision), mean_episode_rewards, sum(flow_rwds), \
                        sum(cols_rwds), sum(dest_rwds), sum(sact_rwds), (end-start)/60))
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                self.log_train(train_infos, episode)

            self.vid_2_idx = None
            self.idx_2_vid = None

            self.envs.close()
            # eval
            #if episode % self.eval_interval == 0 and self.use_eval:
            #    self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs_dict, all_car_info_dict, safe_action_dict = self.envs.reset(scenario=self.all_args.scenario_name)

        self.vid_2_idx = {vid: idx for idx, vid in enumerate(obs_dict.keys())}
        self.idx_2_vid = {idx: vid for vid, idx in self.vid_2_idx.items()}

        print("Start Episode: CAV list -- ", list(self.vid_2_idx.keys()))

        self.safe_actions = {self.vid_2_idx[vid]: safe_act for vid, safe_act in safe_action_dict.items()}

        obs, combined_obs = self.prepare_obs(obs_dict)
        return obs, combined_obs, all_car_info_dict
        


    '''
    This is the inference step happens before environment.step()
    within this handle the safe action stuff
    '''
    @torch.no_grad()
    def collect(self, step, available_actions):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []
        actions_env = {}
        chosen_actions = {}

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout() # set modules to untrainable
            # what if no available action?
            value, action, action_log_prob, rnn_state, rnn_state_critic, all_action_probs \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            available_actions=torch.tensor(available_actions[agent_id]).unsqueeze(0))
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action) # shape = (1,1,1)
            chosen_actions[agent_id] = action.item() # keys are agents
            if sum(available_actions[agent_id]) == 0: # if no safe action, then execute E_S
                actions_env[self.idx_2_vid[agent_id]] = -1
            else: # else try execute selected action (although not necessarily been executed eventually)
                actions_env[self.idx_2_vid[agent_id]] = action.item()
            #print(action)

            actions.append(action)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # for MPE: [envs, agents, dim]
        # for carla we have a list of dictionaries: [{agent: dim}] * N_envs
        '''
        actions_env = [] # each item is a thread
        for i in range(self.n_rollout_threads):
            one_hot_action_dict = {}
            for i, temp_action_env in enumerate(temp_actions_env):
                one_hot_action_dict[self.idx_2_vid[i]] = temp_action_env
            actions_env.append(one_hot_action_dict) 
        '''
        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2) # shape = (1,3,1)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        # actions are integer values
        # actions_env is one-hot encoding
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, chosen_actions

    '''
    In MPE
    values (10, 3, 1)
    actions (10, 3, 1)
    action_log_probs (10, 3, 1)
    rnn_states (10, 3, 1, 64)
    rnn_states_critic (10, 3, 1, 64)
    obs (10, 3, 18)
    rewards (10, 3, 1)
    dones (10, 3)
    '''
    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, \
            rnn_states, rnn_states_critic, combined_obs = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = combined_obs

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = obs[:, agent_id].copy()

            self.buffer[agent_id].insert(share_obs,
                                        obs[:, agent_id],
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])

    @torch.no_grad()
    def eval(self):
        #episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episodes = self.all_args.episodes

        for episode in range(episodes):
            start = time.time()

            train_infos = []
            error_flag = False
            
            obs, combined_obs, all_car_info_dict = self.warmup()
            eval_rnn_states = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

            all_rewards = []
            flow_rwds = []
            cols_rwds = []
            dest_rwds = []
            sact_rwds = []

            # mpc-cbf default is lane_keeping
            command = "lane_keeping"
            previous_command = command
            reference_lane_index = -5
            lc_timeout_dict = {}
            
            for step in range(self.episode_length):
                # mpc-cbf
                conflicting_cars_state_degree1 = []
                conflicting_cars_state_degree2 = []
                preceding_cars_state = []

                actions_env = {}
                chosen_actions = {}
                available_actions = self.safe_actions
                for agent_id in range(self.num_agents):
                    #self.trainer[agent_id].prep_rollout() # set modules to untrainable
                    self.policy[agent_id].actor.eval()

                    eval_action, eval_rnn_state = self.policy[agent_id].act(obs[:, agent_id], eval_rnn_states[:, agent_id], eval_masks[:, agent_id], 
                                                                            available_actions=torch.tensor(available_actions[agent_id]).unsqueeze(0), 
                                                                            deterministic=False)
                    eval_action = _t2n(eval_action) # shape = (1,1,1)
                    chosen_actions[agent_id] = eval_action.item() # keys are agents
                    if sum(available_actions[agent_id]) == 0: # if no safe action, then execute E_S
                        actions_env[self.idx_2_vid[agent_id]] = -1
                    else: # else try execute selected action (although not necessarily been executed eventually)
                        actions_env[self.idx_2_vid[agent_id]] = eval_action.item()
                    #print(action)

                    eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

                # mpc-cbf
                assert len(actions_env) == 1
                for cavid in actions_env:
                    cav = self.envs.CAV_dict[cavid]

                    command = self.action_2_command[actions_env[cavid]]

                    if cavid in lc_timeout_dict and lc_timeout_dict[cavid] > 0:
                        command = previous_command
                        lc_timeout_dict[cavid] -= 1
                    elif command != "lane_keeping":
                        lc_timeout_dict[cavid] = self.all_args.timeout_lc
                    
                    desired_phi, reference_lane_index=path_plannar(self.envs, cav, command, previous_command,
                                                                   reference_lane_index)
                for vid, info_dict in all_car_info_dict.items():
                    if vid == cavid:
                        agent_state_dict = all_car_info_dict[vid]
                    else:
                        if all_car_info_dict[vid]["lane_id"] == all_car_info_dict[cavid]["lane_id"]:
                            if all_car_info_dict[vid]["x"] > all_car_info_dict[cavid]["x"]:
                                x_ip = all_car_info_dict[vid]["x"]
                                y_ip = all_car_info_dict[vid]["y"]
                                phi_ip = all_car_info_dict[vid]["phi"]
                                vel_ip = all_car_info_dict[vid]["vel"]/3.6
                                preceding_cars_state.append([x_ip, y_ip, phi_ip, vel_ip])
                            else:
                                x_ip = 1000
                                y_ip = all_car_info_dict[cavid]["y"]
                                phi_ip = 0
                                vel_ip = 100
                                preceding_cars_state.append([x_ip, y_ip, phi_ip, vel_ip])
                        else:
                            if all_car_info_dict[vid]["lane_id"] == reference_lane_index:
                                x_ic = all_car_info_dict[vid]["x"]
                                y_ic = all_car_info_dict[vid]["y"]
                                phi_ic = all_car_info_dict[vid]["phi"]
                                vel_ic = all_car_info_dict[vid]["vel"]/3.6
                                conflicting_cars_state_degree1.append([x_ic, y_ic, phi_ic, vel_ic])
                            else:
                                x_ic = all_car_info_dict[vid]["x"]
                                y_ic = all_car_info_dict[vid]["y"]
                                phi_ic = all_car_info_dict[vid]["phi"]
                                vel_ic = all_car_info_dict[vid]["vel"]/3.6
                                conflicting_cars_state_degree2.append([x_ic, y_ic, phi_ic, vel_ic])


                states = [agent_state_dict['x'], agent_state_dict['y'], \
                          agent_state_dict['phi'], agent_state_dict['vel']/3.6]
                status, action, next_states, model_acc = mpc_exec1(desired_phi, states, \
                                preceding_cars_state, conflicting_cars_state_degree1, \
                                conflicting_cars_state_degree2, 1000)

                a = float(action[0])
                throttle, brake = throttle_brake_mapping1(a)
                
                # mpc-cbf 
                obs_dict, rewards, done, all_car_info_dict, \
                    rwd_items, safe_actions, force_continue = \
                    self.envs.step({self.idx_2_vid[0]: [throttle, brake, action[1]]}, step_n=step)

                previous_command = command
                
                if force_continue:
                    error_flag = True
                    break

                all_rewards.append(rewards.copy())
                flow_rwds.append(rwd_items[0])
                cols_rwds.append(rwd_items[1])
                dest_rwds.append(rwd_items[2])
                sact_rwds.append(rwd_items[3])

                self.safe_actions = {self.vid_2_idx[vid]: safe_act for vid, safe_act in safe_actions.items()} 

                true_actions = {}
                for agent_id in range(self.num_agents):
                    true_actions[self.idx_2_vid[agent_id]] = self.command_2_action[command[:2]]

                # --------- printing step-wise infomation --------- #
                self.print_stepwise_msg(actions_env, true_actions, all_car_info_dict, episode, step, rewards, rwd_items)
                # --------- printing step-wise infomation --------- #

                obs, combined_obs = self.prepare_obs(obs_dict)
                def prep_carla_data(rewards, done):
                    rewards = np.expand_dims(rewards, axis=0)
                    dones = np.expand_dims(np.array([done]*self.num_agents), axis=0)
                    return rewards, dones

                _, dones = prep_carla_data(rewards, done)

                # in buffer, rnn_state_shape = (episode+1, n_thread, recurrent_N, hidden_size)
                # in buffer, rnn_state_shape = (episode+1, n_thread, 1)
                # here, eval_rnn_states: (1, 3, recurN, hidden)
                # eval_mask: (1,3,1)
                eval_rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
                eval_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            if error_flag:
                self.vid_2_idx = None
                self.idx_2_vid = None
                self.envs.close()
                print("Error in simulation, continue to next episode...")
                continue

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes))

                if self.env_name == 'CARLA':
                    mean_episode_rewards = np.array(all_rewards).sum(0).mean()
                    for agent_id in range(self.num_agents):
                        train_info = {}
                        train_info.update({"avg_eps_rewards": mean_episode_rewards})
                        train_info.update({"avg_eps_flow_rewards": sum(flow_rwds)})
                        train_info.update({"avg_eps_dest_rewards": sum(dest_rwds)})
                        train_info.update({"avg_eps_cols_rewards": sum(cols_rwds)})
                        train_info.update({"avg_eps_sact_rewards": sum(sact_rwds)})
                        train_infos.append(train_info)
                    print("Episode {:d} any collision {}; has reward {:.3f}, flow {:.3f}, cols {:.3f}, dest {:.3f}, sact {:.3f}, takes time {:.2f} minutes."\
                        .format(episode, str(self.envs.done_collision), mean_episode_rewards, sum(flow_rwds), \
                        sum(cols_rwds), sum(dest_rwds), sum(sact_rwds), (end-start)/60))
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                self.log_train(train_infos, episode)

            self.vid_2_idx = None
            self.idx_2_vid = None

            self.envs.close()

    @torch.no_grad()
    def render(self):        
        raise NotImplementedError

    def prepare_obs(self, obs_dict):
        combined_obs = np.expand_dims(np.array(list(chain(*obs_dict.values()))), axis=0)
        obs = []
        for vid, v_obs in obs_dict.items():
            obs.append(v_obs)
        return np.expand_dims(np.array(obs), axis=0), combined_obs

    # conduct adversarial attack
    def apply_attack(self, last_states, eps=2., steps=5, attack_ratio = 0.2, attack_method="critic", aux_params=None):
        import random
        if attack_ratio < random.random():
            return last_states

        if attack_method == 'critic':
            agent_id, rnn_states_critic, masks = aux_params
            if steps > 0:
                step_eps = eps / steps 
            else:
                step_eps = eps
            clamp_min = last_states - eps 
            clamp_max = last_states + eps 

            noise = torch.empty_like(last_states).uniform_(-step_eps, step_eps)
            states = last_states + noise
            with torch.enable_grad():
                for i in range(steps):
                    states = states.clone().detach().requires_grad_()
                    value, _ = self.policy[agent_id].critic(states, rnn_states_critic, masks).mean() #dim=1
                    print(value)
                    value.backward()
                    update = states.grad.sign() * step_eps
                    # Clamp to +/- eps.
                    states.data = torch.min(torch.max(states.data - update, clamp_min), clamp_max)
                self.policy[agent_id].critic.zero_grad()
            return states.detach()

        elif attack_method == 'random':
            return last_states
        elif attack_method == 'action':
            return last_states
        elif attack_method == 'sarsa':
            return last_states
        elif attack_method == 'advpolicy':
            return last_states
        elif attack_method == 'none':
            return last_states
        else:
            raise ValueError("Unknown attack method: {}".format(attack_method))




    def print_stepwise_msg(self, actions, true_actions, all_car_info_dict, episode, step, rewards, rwd_items):
        print('=========================================================')
        V_INFO_FMT_STR = "Car ID: {}; Location: ({:3.1f}, {:3.1f}); Road ID: {}; Lane ID: {}; Is_Junction: {}; Lane_type: {}; Lane_Change: {}; Velocity: {:.2f}km/h"
        STEP_LOG_STR = "Epi: {:4d}; Step: {:4d}; Rewards: #####; flow_r: {:.2f}; cols_r: {:.2f}; dest_r: {:.2f}; sact_r: {:.2f}"

        for vid, car in all_car_info_dict.items():
            print(V_INFO_FMT_STR.format(vid, car['x'], car['y'], car['road_id'], \
                car['lane_id'], car['jct_id'], car['lane_type'], car['lane_chg'], car['vel']))
        chosen_action_str = "{} || ".format(self.algorithm_name)
        true_action_str = "True || "
        for i, v in actions.items():
            chosen_action_str += "Agent {}: {}; ".format(i, Behavior(v))
            true_action_str += "Agent {}: {}; ".format(i, true_actions[i])
        print(chosen_action_str)
        print(true_action_str)

        def get_log_string(n_agents):
            r_log_str = ",".join(["{:.2f}"]*n_agents)
            return STEP_LOG_STR.replace("#####", r_log_str)

        step_log_str = get_log_string(self.num_agents)
        print(step_log_str.format(episode, step, *rewards.tolist(), rwd_items[0], rwd_items[1], rwd_items[2], rwd_items[3]))