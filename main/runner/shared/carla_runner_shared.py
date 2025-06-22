    
import time
import pickle
# import wandb
import os
import traceback
import numpy as np
from itertools import chain
import torch
from copy import deepcopy
from main.algorithms.utils.util import update_linear_schedule, compute_discounted_return, param_schedule
from main.runner.shared.base_runner_shared import Runner
from main.runner.runner_util import print_stepwise_msg, print_episode_msg, group_episode_results
from main.envs.carla.enums import Behavior
from main.envs.carla.control import throttle_brake_mapping1, path_plannar, find_conflicting_cars
from main.envs.carla.control import mpc_exec_highway, mpc_exec_crossing, high_level_lane_changing
import re

'''
    u_min = -1; umax = 1
    speed limit: 
    Keep_lane: maximum u_ref
    KL_0: u_ref = 0; keep speed
    KL_i:
    discretize = 3
    KL_0, 1, 2Kep_lane = 1
    KL_0 = -1, KL1 = -0.33, KL2 = 0.33
'''
def map_kl_x(kl_x_str, umax, c):   
    # Extract the integer part x from the string kl_x
    x = int(kl_x_str.split('_')[1])
   
    # Calculate the mapped value
    u = x / c * umax
    return u

# Example usage
umax = 10  # Example value for umax
c = 5      # Example value for c


def _t2n(x):
    return x.detach().cpu().numpy()

class CARLARunner(Runner):
    def __init__(self, config):
        super(CARLARunner, self).__init__(config)
        # self.vid_2_idx = None #Use this index; it is consistent
        # self.idx_2_vid = None
        self.vid_list = None
        self.apply_safe_reward = self.all_args.apply_safe_reward
        self.encourage_r = self.all_args.encourage_r
        self.avg_coef = self.all_args.avg_coef
        
        # num_actions = discretize + 3
        self.action_2_command = {0: "lane_keeping", 1: "left_lc", 2: "right_lc", 
                                 3: "KL_0", 4: "KL_1", 5: "KL_2", \
                                 6: "KL_3", 7: "KL_4"} 
        self.lane_change_commands = ['left_lc', 'right_lc']

        self.rule_based_r = self.all_args.rule_based_r
        # we won't have rule-based reward and manual encouraging reward at the same time
        assert (not self.rule_based_r) or (not self.all_args.enco or self.all_args.enco == 'none')
        
        if not self.all_args.enco or self.all_args.enco == 'none':
            self.encourage_r = 0
            self.encouraging_commands = []
        elif self.all_args.enco == 'speed':
            self.encouraging_commands = ["lane_keeping"] + ["KL_0","KL_1","KL_2", "KL_3","KL_4"][self.discretize-1:self.discretize]
        elif self.all_args.enco == 'lane_change':
            self.encouraging_commands = ["left_lc", "right_lc"]
        self.command_2_action = {v: k for k, v in self.action_2_command.items()}
        
        # acc = f(velocity, throttle, brake)
        self.timeout_lanechange = 80

        '''
            To be saved in the pickle (episode):
            For entire experiment:
                all_arguments

            For every episode:
                - stepwise rewards: n_steps * n_agents
                - stepwise flow rewards: n_steps * n_agents
                - stepwise dest rewards: n_steps * n_agents
                
                - stepwise average collision rewards: n_steps * 1
                - stepwise average safety rewards: n_steps * 1
                - stepwise average encourage rewards: n_steps * 1

                - returns: n_agents * 1
                - discounted returns: n_agents * 1
                - unweighted flow return: n_agents * 1
                - destination_return: n_agents * 1
        '''
        self.store_dict = {'args': self.all_args}
       
    def run(self):
        
        #episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episodes = self.all_args.episodes
        if self.algorithm_name == 'wmappo':
            self.trainer.robust_eps_scheduler.set_epoch_length(episodes)
            self.trainer.q_weight_scheduler.set_epoch_length(episodes)

        for episode in range(episodes):
            start = time.time()
            
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            if self.algorithm_name == 'wmappo':
                self.trainer.robust_eps_scheduler.step_epoch()
                self.trainer.q_weight_scheduler.step_epoch()
                print("Eps: {}; q_weight: {}".format(self.trainer.robust_eps_scheduler.get_eps(),\
                        self.trainer.q_weight_scheduler.get_eps()))

            error_flag = False
            
            # obs shape [1, n_agents, obs_shape]
            # combined_obs shape [1, n_agents, n_agents*obs_shape]
            obs, combined_obs, all_car_info_dict = self.warmup()
            '''
                episode = 300, n_thread = 1, num_agents = 3, obs_shape=128,\
                share_obs_shape = 128*3 = 384

            In the buffer: (not using centralized V)
                share_obs: [300, 1, 3, 128]
                obs: [300, 1, 3, 128]
                rnn_states: [300, 1, 3, recurrent_N, hidden_size]

            '''
            # here store the initial obs 
            if self.use_centralized_V:
                self.buffer.share_obs[0] = combined_obs.copy()
            else:
                self.buffer.share_obs[0] = obs.copy()
            self.buffer.obs[0] = obs.copy()

            all_rewards = []
            all_flow_rewards = []
            all_dest_rewards = []

            flow_rwds = []
            cols_rwds = []
            dest_rwds = []
            safe_rwds = []
            enco_rwds = []

            # newly-added
            agents_dict = self.envs.get_agents_dict()
            cavs_list = list(self.envs.CAV_dict.keys())
            # why do we need these two lists
            all_car_list = list(all_car_info_dict.keys())
            non_cavs_list = [elem for elem in all_car_list if elem not in cavs_list]
            non_cavs_lane_ids = {key: all_car_info_dict[key]["lane_id"] for key in non_cavs_list}
            non_cavs_position_list = {key: dict(pre_x = 0, pre_y = 0) for key in non_cavs_list}

            for vid in non_cavs_list:
                non_cavs_position_list[vid]['pre_x'] = all_car_info_dict[vid]["x"]
                non_cavs_position_list[vid]['pre_y'] = all_car_info_dict[vid]['y']

            for vid in self.envs.CAV_dict.keys():
                agents_dict[vid]["initial_position"]['x'] = all_car_info_dict[vid]['x']
                agents_dict[vid]["initial_position"]['y'] = all_car_info_dict[vid]['y']
            # # mpc-cbf default is lane_keeping
            # command = "lane_keeping"
            # previous_command = command
            # reference_lane_index = -5
            # lc_timeout_dict = {}
            tt = 1
            vid_select = None
            acc_ref = 3.92

            for step in range(self.episode_length):
                
                # [0,0,0,0] = [err_x, err_y, err_vx, err_vy]
                # perturbation_dict = dict.fromkeys(all_car_info_dict, [3,1,3,3])
                perturbation_dict = dict.fromkeys(all_car_info_dict, [0,0,0,0])
                
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env \
                    = self.collect(step, available_actions=None)
                
                assert len(actions_env) == self.num_agents

                safety_score_per_step = 0.
                encourage_rwds = dict.fromkeys(cavs_list, 0.)

                for vid, cav in self.envs.CAV_dict.items():
                    # provide a velocity reference
                    ego_id = vid
                    # command = self.action_2_command[actions_env[vid]]
                    command = self.action_2_command[actions_env[vid]]
                    ncav_loc = cav.get_location()
                    ncav_waypoint = self.envs.map.get_waypoint(ncav_loc)
                    current_lane_index = ncav_waypoint.lane_id

                    #if current_lane_index == -5 and ncav_loc.x > -60 and tt == 1:
                    #    tt -= 1
                    #    vid_select = vid
                    highway_lanebound = None
                    if self.all_args.middle3:
                        highway_lanebound = [-4,-6]

                    # command = "lane_keeping"
                    agents_dict[vid]["recomended_command"] = high_level_lane_changing(self.envs, cav, agents_dict, all_car_info_dict, non_cavs_list, ego_id, 0, 0.3, 2, perturbation_dict, self.all_args.scenario_name, highway_lanebound=highway_lanebound)

                    if agents_dict[vid]["completion"] > 0 and step == agents_dict[vid]["lc_starting_time"] + self.timeout_lanechange:
                        agents_dict[vid]["timeout"] = 1
                            
                        
                    if re.match(r'^KL_\d+$', command):
                        acc_ref = map_kl_x(command, 3.92, self.discretize)
                    else:
                        command = command
                        acc_ref = 3.92

                    agents_dict[vid], agents_dict[vid]["reference_lane_index"], \
                        agents_dict[vid]["current_lane_index"] = \
                            path_plannar(step, self.envs, cav, vid, agents_dict, agents_dict[vid], \
                                         command, agents_dict[vid]["previous_command"], cavs_list, \
                                         agents_dict[vid]["reference_lane_index"], \
                                         scenario=self.all_args.scenario_name, highway_lanebound=highway_lanebound)

                    agents_dict[vid], states, preceding_cars_state, \
                        conflicting_cars_state_degree1, conflicting_cars_state_degree2, \
                        MPs, conflicting_cars_state_degree3 = \
                            find_conflicting_cars(self.envs, agents_dict, cav, agents_dict[vid], \
                                                  ego_id, all_car_info_dict, agents_dict[vid]["reference_lane_index"], \
                                                  non_cavs_lane_ids, non_cavs_position_list, perturbation_dict, \
                                                  scenario=self.all_args.scenario_name)


                    agents_dict[vid]["position"]['x'],agents_dict[vid]["position"]['y'] =  states[0], states[1]

                    try:
                        force_continue = False
                        if self.all_args.scenario_name == "crossing":
                            status, action, slack_vars = \
                                mpc_exec_crossing(agents_dict[vid], acc_ref, states, preceding_cars_state,\
                                    conflicting_cars_state_degree1, conflicting_cars_state_degree2,vid, \
                                    vid_select, current_lane_index, non_cavs_lane_ids, MPs, \
                                    conflicting_cars_state_degree3, scenario=self.all_args.scenario_name)
                        else:
                            status, action, slack_vars = \
                                mpc_exec_highway(agents_dict[vid], acc_ref, states, preceding_cars_state, \
                                    conflicting_cars_state_degree1, conflicting_cars_state_degree2, vid, \
                                    vid_select, current_lane_index, non_cavs_lane_ids, MPs, \
                                    conflicting_cars_state_degree3, scenario=self.all_args.scenario_name)
                        
                        if slack_vars > 5:
                            # has range [0,1]
                            safety_score_per_step = safety_score_per_step - 0.2

                        if self.rule_based_r:
                            if agents_dict[vid]["recommended_command"] != agents_dict[vid]["command"]:
                                if agents_dict[vid]["recommended_command"] in self.lane_change_commands:
                                    encourage_rwds[vid] = param_schedule(episode, 120, 180, -5., 0., method='quad')
                                else:
                                    encourage_rwds[vid] = param_schedule(episode, 120, 180, -0.5, 0., method='quad')
                            
                        elif self.encourage_r != 0 and agents_dict[vid]["command"] in self.encouraging_commands:
                            '''
                                To use param_schedule:
                                param_schedule(curr_time, start_time, end_time, start_value, end_value, method='linear')
                            '''
                            encourage_rwds[vid] = param_schedule(episode, 120, 180, self.encourage_r, 0, method='quad')
                            # encourage_rwds[vid] = self.encourage_r

                        if agents_dict[vid]["command"] in self.lane_change_commands and \
                            agents_dict[vid]["current_lane_index"] != self.envs.previous_lane_ids[vid]:
                            encourage_rwds[vid] = param_schedule(episode, 120, 180, 5., 0, method='quad')
                            if agents_dict[vid]['previous_completion'] >= 0.8:
                                self.envs.previous_lane_ids[vid] = agents_dict[vid]["current_lane_index"]

                        agents_dict[vid]["previous_command"] = agents_dict[vid]["command"]
                        agents_dict[vid]["previous_completion"] = agents_dict[vid]["completion"]
                        a = float(action[0])
                        throttle, brake = throttle_brake_mapping1(a)
                        agents_dict[vid]["throttle"] = throttle
                        agents_dict[vid]["brake"] = brake
                        agents_dict[vid]["steer"] = action[1]
                        agents_dict[vid]["acceleration"] = action[0]

                    except RuntimeError:
                        traceback.print_exc()
                        print("Control: Early termination because the controller failed.")
                        print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        force_continue = True
                    except:
                        traceback.print_exc()
                        print("Control: Unexpected error.")
                        print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        force_continue = True
                    finally:
                        if force_continue:
                            break

                if force_continue:
                    error_flag = True
                    break

                safety_rwds = dict.fromkeys(cavs_list, safety_score_per_step)
                # Obser reward and next obs; infos is not necessary
                # !!! The shape of each item here might need adjustment
                # the true action could contains E_S, meantime the corresponding rewards should get penalized.
                action_dict = {vid: [agents_dict[vid]["throttle"], agents_dict[vid]["brake"], agents_dict[vid]["steer"]]
                               for vid in cavs_list}

                # for vid in action_dict:
                #     print(vid, agents_dict[vid]["throttle"], agents_dict[vid]["brake"], agents_dict[vid]["steer"])
                
                for noncavid in non_cavs_list:
                    non_cavs_position_list[noncavid]['pre_x'] = all_car_info_dict[noncavid]["x"]
                    non_cavs_position_list[noncavid]['pre_y'] = all_car_info_dict[noncavid]['y']

                # rwd_items: (vels_rwds, dest_rwds, col_penalties, safe_rwds)
                obs_dict, done, all_car_info_dict, rwd_items, _, force_continue, error_dict = \
                    self.envs.step(action_dict=action_dict, step_n=step)
                perturbation_dict = error_dict

                if force_continue:
                    error_flag = True
                    break
                
                # mean_rwd_items = (mean_vels_rwd, mean_dest_rwd, mean_cols_rwd, mean_safe_rwd, mean_encourage_rwd)
                # rewards, dones shape: [1, n_agents, 1]
                rewards, dones, mean_rwd_items, flow_rewards, dest_rewards = \
                    self.prep_reward_data(rwd_items, encourage_rwds, safety_rwds, done, avg_coef=self.avg_coef)
                # self.prep_reward_data(rwd_items, encourage_rwds, safety_rwds, done, agents_dict, avg_coef=self.avg_coef)

                # rewards was originally an np array of length num_cav
                rewards_log = np.squeeze(deepcopy(rewards))
                all_rewards.append(rewards_log)
                all_flow_rewards.append(flow_rewards)
                all_dest_rewards.append(dest_rewards)

                flow_rwds.append(mean_rwd_items[0])
                dest_rwds.append(mean_rwd_items[1])
                cols_rwds.append(mean_rwd_items[2])
                safe_rwds.append(mean_rwd_items[3])
                enco_rwds.append(mean_rwd_items[4])

                executed_actions = []
                true_actions = {}
                for vid in self.vid_list:
                    executed_command = agents_dict[vid]["previous_command"]
                    executed_action = self.command_2_action[executed_command]
                    
                    # the reason we did the below is that, in the path_planner and controller, all u_refs are treated equally as Keep_lane 
                    if executed_action != 1 and executed_action != 2:
                        executed_action = actions_env[vid]
                    true_actions[vid] = executed_action
                    executed_actions.append([executed_action])
                executed_actions = np.expand_dims(np.array(executed_actions), axis=0)
                
                obs, combined_obs = self.prepare_obs_combined(obs_dict)
  
                data = obs, rewards, dones, None, values, executed_actions, action_log_probs, \
                       rnn_states, rnn_states_critic, combined_obs
                
                # insert data into bufferentropy_coef
                self.insert(data)

                # --------- printing step-wise infomation --------- #
                print_stepwise_msg(self.algorithm_name, cavs_list, actions_env, true_actions, all_car_info_dict, \
                    episode, step, rewards_log, mean_rwd_items, list(rwd_items[:3])+[safety_rwds, encourage_rwds], Behavior)
                # --------- printing step-wise infomation ---------


            if error_flag:
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

                if self.env_name == 'CARLA':
                    all_rewards = np.asarray(all_rewards)
                    episode_returns = all_rewards.sum(axis=0)
                    mean_episode_rewards = episode_returns.mean()
                    discounted_returns = compute_discounted_return(all_rewards, self.all_args.gamma)
                    avg_rwd_items = [flow_rwds, dest_rwds, cols_rwds, safe_rwds, enco_rwds]

                    train_infos.update({"avg_eps_rewards": mean_episode_rewards})
                    train_infos.update({"avg_eps_flow_rewards": sum(flow_rwds)})
                    train_infos.update({"avg_eps_dest_rewards": sum(dest_rwds)})
                    train_infos.update({"avg_eps_cols_rewards": sum(cols_rwds)})
                    train_infos.update({"avg_eps_safe_rewards": sum(safe_rwds)})
                    train_infos.update({"avg_eps_enco_rewards": sum(enco_rwds)})

                    print_episode_msg(episode, self.envs.done_collision, (end-start)/60, episode_returns, avg_rwd_items, \
                        discounted_returns, all_flow_rewards, all_dest_rewards, self.all_args.flow_reward_coef)

                    episode_dict = group_episode_results(all_rewards, all_flow_rewards, all_dest_rewards, \
                        cols_rwds, safe_rwds, enco_rwds, self.all_args.flow_reward_coef, self.all_args.gamma)
                    episode_dict['switch'] = self.envs.switch

                    self.store_dict[episode] = episode_dict

                self.log_train(train_infos, episode)

            self.envs.close()

    def warmup(self):
        # reset env
        obs_dict, all_car_info_dict, _ = self.envs.reset(scenario=self.all_args.scenario_name)

        # have a fixed order of vehicle_id list
        self.vid_list = list(obs_dict.keys())

        print("Start Episode: CAV list -- ", self.vid_list)

        if self.cav_force_straight_step > 0:
            dummy_actions = dict.fromkeys(obs_dict.keys(), [1.0, 0., 0.])

            for step in range(self.cav_force_straight_step):
                if step == self.cav_force_straight_step - 1:
                    obs_dict, _, _ = self.envs.step_cav_only(dummy_actions) 
                else:
                    _, all_car_info_dict, _ = self.envs.step_cav_only(dummy_actions)
            print("Finished freezing and Acceleration: step {}".format(self.cav_force_straight_step))

        obs, combined_obs = self.prepare_obs_combined(obs_dict)
        return obs, combined_obs, all_car_info_dict

    '''
    This is the inference step happens before environment.step()
    within this handle the safe action stuff
    '''
    @torch.no_grad()
    def collect(self, step, available_actions=None):
        self.trainer.prep_rollout()
        # values = []
        # actions = []
        # action_log_probs = []
        # rnn_states = []
        # rnn_states_critic = []
        actions_env = {}

        if available_actions:
            available_actions = torch.tensor(available_actions).unsqueeze(0) 
        else:
            available_actions = None

        # value should have shape [n_thread, n_agent, 1]
        # all_action_probs is not used here
        value, action, action_log_prob, rnn_states, rnn_states_critic, all_action_probs \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            available_actions=available_actions)
        
        # torch.Size([3, 1]) torch.Size([3, 1]) torch.Size([3, 1]) 
        # torch.Size([3, 1, 64]) torch.Size([3, 1, 64]) torch.Size([3, 8])
        # print(value.shape, action.shape, action_log_prob.shape, rnn_states.shape, \
        #     rnn_states_critic.shape, all_action_probs.shape if all_action_probs is not None else str(None))

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        # (1, 3, 1) (1, 3, 1) (1, 3, 1) (1, 3, 1, 64) (1, 3, 1, 64)
        # print(values.shape, actions.shape, action_log_probs.shape, rnn_states.shape, \
        #     rnn_states_critic.shape)
        for i, vid in enumerate(self.vid_list):
            actions_env[vid] = np.squeeze(actions)[i]

        # actions are integer values
        # actions_env is one-hot encoding
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

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

        if not self.use_centralized_V:
            share_obs = obs
        else:
            share_obs = combined_obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self):
        #episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        episodes = self.all_args.episodes

        for episode in range(episodes):
            start = time.time()

            error_flag = False
            
            # the self.vid_list is set in warmup, and the order of vids matters
            obs, _, all_car_info_dict = self.warmup()
            eval_rnn_states = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

            all_rewards = []
            all_flow_rewards = []
            all_dest_rewards = []

            flow_rwds = []
            cols_rwds = []
            dest_rwds = []
            safe_rwds = []
            enco_rwds = []

            agents_dict = self.envs.get_agents_dict()
            cavs_list = list(self.envs.CAV_dict.keys())
            all_car_list = list(all_car_info_dict.keys())
            non_cavs_list = [elem for elem in all_car_list if elem not in cavs_list]
            non_cavs_lane_ids = {key: all_car_info_dict[key]["lane_id"] for key in non_cavs_list}
            non_cavs_position_list = {key: dict(pre_x = 0, pre_y = 0) for key in non_cavs_list}

            for vid in non_cavs_list:
                non_cavs_position_list[vid]['pre_x'] = all_car_info_dict[vid]["x"]
                non_cavs_position_list[vid]['pre_y'] = all_car_info_dict[vid]['y']

            for vid in self.envs.CAV_dict.keys():
                agents_dict[vid]["initial_position"]['x'] = all_car_info_dict[vid]['x']
                agents_dict[vid]["initial_position"]['y'] = all_car_info_dict[vid]['y']

            # # mpc-cbf default is lane_keeping
            # command = "lane_keeping"
            # previous_command = command
            # reference_lane_index = -5
            # lc_timeout_dict = {}
            tt = 1
            vid_select = None
            # acc_ref = 3.92

            for step in range(self.episode_length):
                self.policy.actor.eval()

                perturbation_dict = dict.fromkeys(all_car_info_dict, [0,0,0,0])
                
                # rnn_states shape: [n_thread=1, n_agents, n_rnn, hidden_size]
                eval_action, eval_rnn_states = self.policy.act(np.concatenate(obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=self.all_args.deterministic)

                eval_action = _t2n(eval_action)
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                
                actions_env = {}
                for i, vid in enumerate(self.vid_list):
                    actions_env[vid] = np.squeeze(eval_action)[i]

                #------------------------- mpc-cbf --------------------------------
                assert len(actions_env) == self.num_agents

                safety_score_per_step = 0.
                encourage_rwds = dict.fromkeys(cavs_list, 0.)

                for vid, cav in self.envs.CAV_dict.items():
                    # provide a velocity reference
                    ego_id = vid
                    # command = self.action_2_command[actions_env[vid]]
                    command = self.action_2_command[actions_env[vid]]
                    ncav_loc = cav.get_location()
                    ncav_waypoint = self.envs.map.get_waypoint(ncav_loc)
                    current_lane_index = ncav_waypoint.lane_id
                    # if current_lane_index == -4 and tt == 1:
                    #     tt -= 1
                    #     vid_select = vid

                    if agents_dict[vid]["completion"] > 0 and step == agents_dict[vid]["lc_starting_time"] + self.timeout_lanechange:
                        agents_dict[vid]["timeout"] = 1

                    if re.match(r'^KL_\d+$', command):
                        acc_ref = map_kl_x(command, 3.92, self.discretize)
                    else:
                        command = command
                        acc_ref = 3.92

                    highway_lanebound = None
                    if self.all_args.middle3:
                        highway_lanebound = [-4,-6]
                    
                    agents_dict[vid], agents_dict[vid]["reference_lane_index"], \
                        agents_dict[vid]["current_lane_index"] = \
                            path_plannar(step, self.envs, cav, vid, agents_dict, agents_dict[vid], \
                                         command, agents_dict[vid]["previous_command"], cavs_list, \
                                         agents_dict[vid]["reference_lane_index"], \
                                         scenario=self.all_args.scenario_name, highway_lanebound=highway_lanebound)

                    agents_dict[vid], states, preceding_cars_state, \
                        conflicting_cars_state_degree1, conflicting_cars_state_degree2, \
                        MPs, conflicting_cars_state_degree3 = \
                            find_conflicting_cars(self.envs, agents_dict, cav, agents_dict[vid], \
                                                  ego_id, all_car_info_dict, agents_dict[vid]["reference_lane_index"], \
                                                  non_cavs_lane_ids, non_cavs_position_list, perturbation_dict, \
                                                  scenario=self.all_args.scenario_name)
                    
                    agents_dict[vid]["position"]['x'],agents_dict[vid]["position"]['x'] = states[0], states[1]
                    
                    try:
                        force_continue = False
                        if self.all_args.scenario_name == "crossing":
                            status, action, slack_vars = \
                                mpc_exec_crossing(agents_dict[vid], acc_ref, states, preceding_cars_state,\
                                    conflicting_cars_state_degree1, conflicting_cars_state_degree2, vid, \
                                    vid_select, current_lane_index, non_cavs_lane_ids, MPs, \
                                    conflicting_cars_state_degree3, scenario=self.all_args.scenario_name)
                        else:
                            status, action, slack_vars = \
                                mpc_exec_highway(agents_dict[vid], acc_ref, states, preceding_cars_state, \
                                    conflicting_cars_state_degree1, conflicting_cars_state_degree2, vid, \
                                    vid_select, current_lane_index, non_cavs_lane_ids, MPs, \
                                    conflicting_cars_state_degree3, scenario=self.all_args.scenario_name)

                        if slack_vars > 5:
                            # has range [0,1]
                            safety_score_per_step = safety_score_per_step - 0.3

                        if self.encourage_r != 0 and agents_dict[vid]["command"] in self.encouraging_commands:
                            encourage_rwds[vid] = self.encourage_r

                        agents_dict[vid]["previous_command"] = agents_dict[vid]["command"]
                        agents_dict[vid]["previous_completion"] = agents_dict[vid]["completion"]
                        a = float(action[0])
                        throttle, brake = throttle_brake_mapping1(a)
                        agents_dict[vid]["throttle"] = throttle
                        agents_dict[vid]["brake"] = brake
                        agents_dict[vid]["steer"] = action[1]
                        agents_dict[vid]["acceleration"] = action[0]
                    
                    except RuntimeError:
                        traceback.print_exc()
                        print("Control: Early termination because the controller failed.")
                        print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        force_continue = True
                    except:
                        traceback.print_exc()
                        print("Control: Unexpected error.")
                        print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        force_continue = True
                    finally:
                        if force_continue:
                            break

                if force_continue:
                    error_flag = True
                    break

                safety_rwds = dict.fromkeys(cavs_list, safety_score_per_step)

                action_dict = {vid: [agents_dict[vid]["throttle"], agents_dict[vid]["brake"], agents_dict[vid]["steer"]]
                               for vid in cavs_list}

                for noncavid in non_cavs_list:
                    non_cavs_position_list[noncavid]['pre_x'] = all_car_info_dict[noncavid]["x"]
                    non_cavs_position_list[noncavid]['pre_y'] = all_car_info_dict[noncavid]['y']

                #------------------------- mpc-cbf completed --------------------------------

                # rwd_items: (vels_rwds, dest_rwds, col_penalties, sact_rwds)
                obs_dict, done, all_car_info_dict, rwd_items, _, force_continue, error_dict = \
                    self.envs.step(action_dict=action_dict, step_n=step)
                perturbation_dict = error_dict



                # mean_rwd_items = (mean_vels_rwd, mean_dest_rwd, mean_cols_rwd, mean_safe_rwd, mean_encourage_rwd)
                rewards, dones, mean_rwd_items, flow_rewards, dest_rewards = \
                    self.prep_reward_data(rwd_items, encourage_rwds, safety_rwds, done, avg_coef=self.avg_coef)

                if force_continue:
                    error_flag = True
                    break

                # rewards was originally an np array of length num_cav
                rewards_log = np.squeeze(deepcopy(rewards))
                all_rewards.append(rewards_log)
                all_flow_rewards.append(flow_rewards)
                all_dest_rewards.append(dest_rewards)

                flow_rwds.append(mean_rwd_items[0])
                dest_rwds.append(mean_rwd_items[1])
                cols_rwds.append(mean_rwd_items[2])
                safe_rwds.append(mean_rwd_items[3])
                enco_rwds.append(mean_rwd_items[4])

                true_actions = {}
                # always in the order of policy 0,1,2,...
                for vid in self.vid_list:
                    executed_command = agents_dict[vid]["previous_command"]
                    executed_action = self.command_2_action[executed_command]
                    # the reason we did the below is that, in the path_planner and controller, all u_refs are treated equally as Keep_lane 
                    if executed_action != 1 and executed_action != 2:
                        executed_action = actions_env[vid]
                    true_actions[vid] = executed_action

                obs, combined_obs = self.prepare_obs_combined(obs_dict)

                # --------- printing step-wise infomation --------- #
                print_stepwise_msg(self.algorithm_name, cavs_list, actions_env, true_actions, all_car_info_dict, \
                    episode, step, rewards_log, mean_rwd_items, list(rwd_items[:3])+[safety_rwds, encourage_rwds], Behavior)
                # --------- printing step-wise infomation --------- #


                # in buffer, rnn_state_shape = (episode+1, n_thread, n_agent, recurrent_N, hidden_size)
                # in buffer, rnn_state_shape = (episode+1, n_thread, 1)
                
                # !!!here, eval_rnn_states: (n_thread=1, 3, recurN, hidden)
                # eval_mask: (1,3,1)
                eval_rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
                eval_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            if error_flag:
                # self.vid_2_idx = None
                # self.idx_2_vid = None
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
                    all_rewards = np.asarray(all_rewards)
                    episode_returns = all_rewards.sum(axis=0)
                    mean_episode_rewards = episode_returns.mean()
                    discounted_returns = compute_discounted_return(all_rewards, self.all_args.gamma)
                    avg_rwd_items = [flow_rwds, dest_rwds, cols_rwds, safe_rwds, enco_rwds]

                    eval_info = {}
                    eval_info.update({"avg_eps_rewards": mean_episode_rewards})
                    eval_info.update({"avg_eps_flow_rewards": sum(flow_rwds)})
                    eval_info.update({"avg_eps_dest_rewards": sum(dest_rwds)})
                    eval_info.update({"avg_eps_cols_rewards": sum(cols_rwds)})
                    eval_info.update({"avg_eps_safe_rewards": sum(safe_rwds)})
                    eval_info.update({"avg_eps_enco_rewards": sum(enco_rwds)})

                    print_episode_msg(episode, self.envs.done_collision, (end-start)/60, episode_returns, avg_rwd_items, \
                        discounted_returns, all_flow_rewards, all_dest_rewards, self.all_args.flow_reward_coef)                    

                    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    episode_dict = group_episode_results(all_rewards, all_flow_rewards, all_dest_rewards, \
                        cols_rwds, safe_rwds, enco_rwds, self.all_args.flow_reward_coef, self.all_args.gamma)
                    episode_dict['switch'] = self.envs.switch

                    self.store_dict[episode] = episode_dict

                self.log_train(eval_info, episode)

            self.envs.close()

    @torch.no_grad()
    def render(self):        
        raise NotImplementedError

    def prepare_obs_combined(self, obs_dict):
        obs = []
        for vid in self.vid_list:
            obs.append(obs_dict[vid])
        
        combined_obs = deepcopy(obs)
        # shape [1, n_agent*obs_shape]
        combined_obs = np.expand_dims(np.array(list(chain(*combined_obs))), axis=0)
        # shape [1, n_agents, n_agent*obs_shape]
        combined_obs = np.expand_dims(combined_obs, 1).repeat(self.num_agents, axis=1)
        # shape [1, n_agents, obs_shape]
        obs = np.expand_dims(np.array(obs), axis=0)

        return obs, combined_obs

    # rwd_item = (vels_rwds, dest_rwds, col_penalties, sact_rwds)
    # each item is a dictionary {vid: rwd}
    # by default, safe
    def prep_reward_data(self, rwd_items, encourage_rwds, safety_rwds, done, avg_coef=0.5):
        vels_rwds, dest_rwds, col_penalties, _ = rwd_items
        # compute average reward of velocity, destination and collision
        
        mean_vels_rwd = np.mean(list(vels_rwds.values()))
        mean_dest_rwd = np.mean(list(dest_rwds.values()))
        mean_cols_rwd = np.mean(list(col_penalties.values()))
        mean_safe_rwd = np.mean(list(safety_rwds.values()))
        mean_encourage_rwd = np.mean(list(encourage_rwds.values()))

        rewards = []
        flow_rewards, dest_rewards = [], []
        for vid in self.vid_list:
            agent_reward = (vels_rwds[vid] + dest_rwds[vid] + col_penalties[vid] + safety_rwds[vid] + encourage_rwds[vid]) * (1 - avg_coef) + \
                        (mean_vels_rwd + mean_dest_rwd + mean_cols_rwd + mean_safe_rwd + mean_encourage_rwd) * avg_coef
            rewards.append(agent_reward)
            flow_rewards.append(vels_rwds[vid])
            dest_rewards.append(dest_rwds[vid])

        rewards = np.expand_dims(np.array(rewards), axis=(0,-1))
        dones = np.expand_dims(np.array([done]*self.num_agents), axis=0)
        return rewards, dones, (mean_vels_rwd, mean_dest_rwd, \
            mean_cols_rwd, mean_safe_rwd, mean_encourage_rwd), flow_rewards, dest_rewards

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

    def save_as_pickle(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.store_dict, file)
