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
from main.runner.separated.base_runner_general import Runner
from main.runner.runner_util import print_stepwise_msg, print_episode_msg, group_episode_results
from main.envs.carla.enums import Behavior
from main.envs.carla.control import throttle_brake_mapping1, path_plannar, find_conflicting_cars
from main.envs.carla.control import mpc_exec_highway, mpc_exec_crossing,high_level_controller, high_level_lane_changing
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
        self.vid_2_idx = None #Use this index; it is consistent
        self.idx_2_vid = None
        self.safe_actions = None
        self.apply_safe_reward = self.all_args.apply_safe_reward
        self.encourage_r = self.all_args.encourage_r
        self.avg_coef = self.all_args.avg_coef
        # To Sabbir & Ehsan: This is the modified version of action-command mapping
        # The last few actions refer to different ranges that you want to define in your controller
        # If I am understanding it correctly, Let's say the "u_ref" can be chosen from [-1,1], 
        # where -1 is the maximum braking, achieved when (brake=1, throttle=0);
        # and that 1 is the maximum accelerating, achieved when (brake=0, throttle=1).
        #
        # Then, I will have this --discretize argument for each time run the code, 
        # you can access it from this runner with self.all_args.discretize.
        #
        # And discretize will take one of the values from [1,3,4,5,6] only. If it's 1, the action set
        # will have one keep lane only as what we have right now.
        # For a specfic value of discretize, for example 5, you can freely choose how to split [-1,1] into
        # 5 ranges, as long as they are well-defined. You don't have to equally split the [-1, 1].
        #
        # And again for example discretize = 5, then in total there will be 7 actions. And the action integer that 
        # RL generates will always be one of [0,1,...,6], and the corresponding command will also be as below defined,
        # [left_lc, right_lc, KL_0, ..., KL_4]. 

        # discretize = 0; 3, 4, 5
        # num_actions = discretize + 3
        self.action_2_command = {0: "lane_keeping", 1: "left_lc", 2: "right_lc", 
                                 3: "KL_0", 4: "KL_1", 5: "KL_2", \
                                 6: "KL_3", 7: "KL_4"} 

        if not self.all_args.enco or self.all_args.enco == 'none':
            self.encourage_r = 0
            self.encouraging_commands = []
        elif self.all_args.enco == 'speed':
            self.encouraging_commands = ["lane_keeping"] + ["KL_0","KL_1","KL_2", "KL_3","KL_4"][self.discretize-1:self.discretize]
        elif self.all_args.enco == 'lane_change':
            self.encouraging_commands = ["left_lc", "right_lc"]
        '''
            0, 0.4, 0.8 ,.9 1.0
            discretize == 3: KL0 -> 0., KL1 -> 0.33, KL2 -> 0.66
            discretize == 4: KL0 -> 0., KL1 -> 0.25, KL2 -> 0.5, KL3 -> 0.75
            discretize == 5: KL0 -> 0., KL1 -> 0.2, KL2 -> 0.4, KL3 -> 0.6, KL4 -> 0.8
            KL_0 = 0., KL1 = 0.4, KL2 = 0.8, KL
        '''

        '''
            show results of crossing, highway, discretize = 0,3,4,5
            3cav - crossing (0, 3, 4, 5)
            3cav - highway (0, 3, 4, 5)
            3cav - crossing - pid
            3cav - highway - pid
        '''
        
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
        for episode in range(episodes):
            start = time.time()
            train_infos = []
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
            rewards = [0,0,0]
            sum_rewards = 0
            step = 0
            done = False
            while not done:
                step += 1
                # Sample actions
                # needs to fix: what if no safe action,
                # actions should always allow only normal actions
                # actions_env should contain Emergency_Stop if needed.
                '''
                    apart from action_env and chosen_actions, 
                    all the others are ordered by agent_id 0,1,2,...etc.

                '''
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, chosen_actions \
                    = self.collect(step, available_actions=None)
                
                assert len(actions_env) == self.num_agents

                safety_score_per_step = 0.
                encourage_rwds = dict.fromkeys(cavs_list, 0.)

                for vid, cav in self.envs.CAV_dict.items():
                    # provide a velocity reference
                    ego_id = vid
                    
                    ncav_loc = cav.get_location()
                    
                    ncav_waypoint = self.envs.map.get_waypoint(ncav_loc)
                    current_lane_index = ncav_waypoint.lane_id

                    if agents_dict[ego_id]["position"]["x"] > self.envs.desti_dict[ego_id][0]:
                        agents_dict[ego_id]["done"] = True
                    
                    if current_lane_index == -4 and tt == 1:
                        tt -= 1
                        vid_select = vid

                    if agents_dict[vid]["completion"] > 0 and step == agents_dict[vid]["lc_starting_time"] + self.timeout_lanechange:
                        agents_dict[vid]["timeout"] = 1
                    
                    command = high_level_lane_changing(self.envs, cav, agents_dict, all_car_info_dict, non_cavs_list, ego_id, 0, 0.3, 2, self.all_args.scenario_name)
                    if re.match(r'^KL_\d+$', command):
                        acc_ref = map_kl_x(command, 3.92, self.discretize)
                    else:
                        command = command
                        acc_ref = 3.92
                    
                    agents_dict[vid], agent_state_dict, preceding_cars_state, \
                        conflicting_cars_state_degree1, conflicting_cars_state_degree2, \
                        MPs, conflicting_cars_state_degree3 = \
                            find_conflicting_cars(self.envs, agents_dict, cav, agents_dict[vid], \
                                                  ego_id, all_car_info_dict, agents_dict[vid]["reference_lane_index"], \
                                                  non_cavs_lane_ids, non_cavs_position_list, \
                                                  scenario=self.all_args.scenario_name)
                
                    agents_dict[vid], agents_dict[vid]["reference_lane_index"], \
                        agents_dict[vid]["current_lane_index"] = \
                            path_plannar(step, self.envs, cav, vid, agents_dict, agents_dict[vid], \
                                         command, agents_dict[vid]["previous_command"], cavs_list,\
                                         agents_dict[vid]["reference_lane_index"],\
                                         scenario=self.all_args.scenario_name)
                    
                    states = [agent_state_dict['x'], \
                              agent_state_dict['y'], \
                              agent_state_dict['phi'], \
                              agent_state_dict['vel'] / 3.6]
                    agents_dict[vid]["position"]['x'],agents_dict[vid]["position"]['y'] =  states[0], states[1]

                    try:
                        force_continue = False
                        if self.all_args.scenario_name == "crossing":
                            status, action, slack_vars = \
                                high_level_controller(agents_dict[vid], acc_ref, states, preceding_cars_state,\
                                    conflicting_cars_state_degree1, conflicting_cars_state_degree2,vid, \
                                    vid_select, current_lane_index, non_cavs_lane_ids, MPs, \
                                    conflicting_cars_state_degree3, scenario=self.all_args.scenario_name)
                        else:
                            status, action, slack_vars = \
                                mpc_exec_highway(agents_dict[vid], acc_ref, states, preceding_cars_state, \
                                    conflicting_cars_state_degree1, conflicting_cars_state_degree2, vid, \
                                    vid_select, current_lane_index, non_cavs_lane_ids, MPs, \
                                    conflicting_cars_state_degree3, scenario=self.all_args.scenario_name)
                        
                        # color = self.envs._carla.Color(r=255, g=0, b=0)  # Green color
                        # life_time = 0.1  # Time each point of the ellipse will be visible (0.1 seconds)
                        # segments = 36  # Number of points to approximate the ellipse

                        # # Draw the ellipse using points
                        # for i in range(segments):
                        #     angle = 2 * np.pi * i / segments  # Angle around the ellipse
                        #     for vehicle_states in preceding_cars_state:
                        #         radius_x = 1*agents_dict[vid]['ellipsoid']['a']  # Radius along the x-axis (length)
                        #         radius_y = 1*agents_dict[vid]['ellipsoid']['b']  # Radius along the y-axis (width)
                        #         point = self.envs._carla.Location(
                        #             x=vehicle_states[0] + radius_x * np.cos(angle),
                        #             y=vehicle_states[1] + radius_y * np.sin(angle),
                        #             z=0  # Keeping the same z-level as the car
                        #         )
                        #         self.envs.world.debug.draw_point(point, size=0.1, color=color, life_time=life_time)
                        
                        if slack_vars > 5:
                            # has range [0,1]
                            safety_score_per_step = safety_score_per_step - 0.2

                        if self.encourage_r != 0 and agents_dict[vid]["command"] in self.encouraging_commands:
                            '''
                                To use param_schedule:
                                param_schedule(curr_time, start_time, end_time, start_value, end_value, method='linear')
                            '''
                            encourage_rwds[vid] = param_schedule(episode, 60, 120, self.encourage_r, 0, method='quad')
                            # encourage_rwds[vid] = self.encourage_r

                        agents_dict[vid]["previous_command"] = agents_dict[vid]["command"]
                        agents_dict[vid]["previous_completion"] = agents_dict[vid]["completion"]
                        a = float(action[0])
                        agents_dict[vid]["acceleration"] = a
                        throttle, brake = throttle_brake_mapping1(a)
                        agents_dict[vid]["throttle"] = throttle
                        agents_dict[vid]["brake"] = brake
                        agents_dict[vid]["steer"] = action[1]
                    
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
                obs_dict, done, all_car_info_dict, rwd_items, safe_action_dict, force_continue = \
                    self.envs.step(agents_dict, action_dict=action_dict, step_n=step)

                if force_continue:
                    error_flag = True
                    break

                # mean_rwd_items = (mean_vels_rwd, mean_dest_rwd, mean_cols_rwd, mean_safe_rwd, mean_encourage_rwd)
                rewards, dones, mean_rwd_items, flow_rewards, dest_rewards = \
                    self.prep_reward_data(rwd_items, encourage_rwds, safety_rwds, done, avg_coef=self.avg_coef)

                # rewards was originally an np array of length num_cav
                all_rewards.append(deepcopy(rewards[0]))
                all_flow_rewards.append(flow_rewards)
                all_dest_rewards.append(dest_rewards)
                
                # add rwd_items into the storing array,
                # particularly, the rewards should be arranged in the order of policy_id
                # vels_rwds, dest_rwds, col_penalties, _ = rwd_items
                # vels_rwds_ar = [vels_rwds[self.idx_2_vid[i]] for i in range(self.num_agents)]
                # dest_rwds_ar = [dest_rwds[self.idx_2_vid[i]] for i in range(self.num_agents)]
                # cols_rwds_ar = [col_penalties[self.idx_2_vid[i]] for i in range(self.num_agents)]
                # encourage_rwds_ar = [encourage_rwds[self.idx_2_vid[i]] for i in range(self.num_agents)]
                # safety_rwds_ar = [safety_rwds[self.idx_2_vid[i]] for i in range(self.num_agents)]

                flow_rwds.append(mean_rwd_items[0])
                dest_rwds.append(mean_rwd_items[1])
                cols_rwds.append(mean_rwd_items[2])
                safe_rwds.append(mean_rwd_items[3])
                enco_rwds.append(mean_rwd_items[4])

                executed_actions = []
                true_actions = {}
                for agent_id in range(self.num_agents):
                    executed_command = agents_dict[self.idx_2_vid[agent_id]]["previous_command"]
                    # executed_action = chosen_actions[agent_id]
                    executed_action = self.command_2_action[executed_command]
                    if executed_action != 1 and executed_action != 2:
                        executed_action = actions_env[self.idx_2_vid[agent_id]]
                    true_actions[self.idx_2_vid[agent_id]] = executed_action
                    executed_actions.append([executed_action])
                executed_actions = np.expand_dims(np.array(executed_actions), axis=0)
                
                self.safe_actions = {self.vid_2_idx[vid]: safe_act for vid, safe_act in safe_action_dict.items()} 

                obs, combined_obs = self.prepare_obs(obs_dict)
                sum_reward = sum(rewards[0])
                sum_rewards = 0.99*sum_rewards + sum_reward
                # print(sum_rewards)
                data = obs, rewards, dones, None, values, executed_actions, action_log_probs, \
                       rnn_states, rnn_states_critic, combined_obs
                
                # insert data into buffer
                self.insert(data)

                # --------- printing step-wise infomation --------- #
                print_stepwise_msg(self.algorithm_name, cavs_list, actions_env, true_actions, all_car_info_dict, \
                    episode, step, rewards[0], mean_rwd_items, list(rwd_items[:3])+[safety_rwds, encourage_rwds], Behavior)
                # --------- printing step-wise infomation ---------


            if error_flag:
                self.vid_2_idx = None
                self.idx_2_vid = None
                self.envs.close()
                print("Error in simulation, continue to next episode...")
                continue

            # compute return and update network
            # self.compute()
            # train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            # if (episode % self.save_interval == 0 or episode == episodes - 1):
            #     self.save()

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

                    print_episode_msg(episode, self.envs.done_collision, (end-start)/60, episode_returns, avg_rwd_items, \
                        discounted_returns, all_flow_rewards, all_dest_rewards, self.all_args.flow_reward_coef)

                    episode_dict = group_episode_results(all_rewards, all_flow_rewards, all_dest_rewards, \
                        cols_rwds, safe_rwds, enco_rwds, self.all_args.flow_reward_coef, self.all_args.gamma)
                    episode_dict['switch'] = self.envs.switch

                    self.store_dict[episode] = episode_dict

            self.vid_2_idx = None
            self.idx_2_vid = None

            self.envs.close()

    def warmup(self):
        # reset env
        obs_dict, all_car_info_dict, _ = self.envs.reset(scenario=self.all_args.scenario_name)

        if self.random_mapping_agents:
            policy_ids = list(range(len(obs_dict)))
            np.random.shuffle(policy_ids)
            self.vid_2_idx = {vid: pid for vid, pid in zip(obs_dict, policy_ids)}
        else:
            self.vid_2_idx = {vid: idx for idx, vid in enumerate(obs_dict.keys())}

        self.idx_2_vid = {idx: vid for vid, idx in self.vid_2_idx.items()}

        print("Start Episode: CAV list -- ", list(self.vid_2_idx.keys()))
        for vid, pid in self.vid_2_idx.items():
            print("Policy Mapping: CAV {} - Policy {}".format(vid, pid))

        if self.cav_force_straight_step > 0:
            dummy_actions = dict.fromkeys(self.vid_2_idx.keys(), [0.96, 0., 0.])

            for step in range(self.cav_force_straight_step):
                if step == self.cav_force_straight_step - 1:

                    obs_dict, _, _ = self.envs.step_cav_only(dummy_actions) 
                else:
                    _, all_car_info_dict, _ = self.envs.step_cav_only(dummy_actions)
            print("Finished freezing and Acceleration: step {}".format(self.cav_force_straight_step))

        obs, combined_obs = self.prepare_obs(obs_dict)
        return obs, combined_obs, all_car_info_dict

    '''
    This is the inference step happens before environment.step()
    within this handle the safe action stuff
    '''
    @torch.no_grad()
    def collect(self, step, available_actions=None):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []
        actions_env = {}
        chosen_actions = {}

        if available_actions:
            available_actions = torch.tensor(available_actions).unsqueeze(0) 
        else:
            available_actions = None

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout() # set modules to untrainable
            # what if no available action?
            value, action, action_log_prob, rnn_state, rnn_state_critic, all_action_probs \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            available_actions=available_actions)
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action) # shape = (1,1,1)
            chosen_actions[agent_id] = action.item() # keys are agents
            actions_env[self.idx_2_vid[agent_id]] = action.item()

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
            
            # the idx_2_vid & vid_2_idx mapping dictionaries are updated in warmup()
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

                actions_env = {}
                chosen_actions = {}

                for agent_id in range(self.num_agents):
                    #self.trainer[agent_id].prep_rollout() # set modules to untrainable
                    self.policy[agent_id].actor.eval()

                    eval_action, eval_rnn_state = self.policy[agent_id].act(obs[:, agent_id], 
                                                                            eval_rnn_states[:, agent_id], 
                                                                            eval_masks[:, agent_id], 
                                                                            deterministic=False)
                    eval_action = _t2n(eval_action) # shape = (1,1,1)
                    chosen_actions[agent_id] = eval_action.item() # keys are agents
                    actions_env[self.idx_2_vid[agent_id]] = eval_action.item()

                    eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

                # mpc-cbf
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
                    if current_lane_index == -4 and tt == 1:
                        tt -= 1
                        vid_select = vid

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
                                         scenario=self.all_args.scenario_name)

                    agents_dict[vid], agent_state_dict, preceding_cars_state, \
                        conflicting_cars_state_degree1, conflicting_cars_state_degree2, \
                        MPs, conflicting_cars_state_degree3 = \
                            find_conflicting_cars(self.envs, agents_dict, cav, agents_dict[vid], \
                                                  ego_id, all_car_info_dict, agents_dict[vid]["reference_lane_index"], \
                                                  non_cavs_lane_ids, non_cavs_position_list, \
                                                  scenario=self.all_args.scenario_name)

                    states = [agent_state_dict['x'], \
                              agent_state_dict['y'], \
                              agent_state_dict['phi'], \
                              agent_state_dict['vel'] / 3.6]
                    
                    agents_dict[vid]["position"]['x'],agents_dict[vid]["position"]['x'] = states[0], states[1]
                    
                    try:
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

                # rwd_items: (vels_rwds, dest_rwds, col_penalties, sact_rwds)
                obs_dict, done, all_car_info_dict, rwd_items, _, force_continue = \
                    self.envs.step(action_dict=action_dict, step_n=step)

                # mean_rwd_items = (mean_vels_rwd, mean_dest_rwd, mean_cols_rwd, mean_safe_rwd, mean_encourage_rwd)
                rewards, dones, mean_rwd_items, flow_rewards, dest_rewards = \
                    self.prep_reward_data(rwd_items, encourage_rwds, safety_rwds, done, avg_coef=self.avg_coef)

                if force_continue:
                    error_flag = True
                    break

                all_rewards.append(deepcopy(rewards[0]))
                all_flow_rewards.append(flow_rewards)
                all_dest_rewards.append(dest_rewards)

                flow_rwds.append(mean_rwd_items[0])
                dest_rwds.append(mean_rwd_items[1])
                cols_rwds.append(mean_rwd_items[2])
                safe_rwds.append(mean_rwd_items[3])
                enco_rwds.append(mean_rwd_items[4])

                true_actions = {}
                # always in the order of policy 0,1,2,...
                for agent_id in range(self.num_agents):
                    executed_command = agents_dict[self.idx_2_vid[agent_id]]["previous_command"]
                    # executed_action = chosen_actions[agent_id]
                    executed_action = self.command_2_action[executed_command]
                    true_actions[self.idx_2_vid[agent_id]] = executed_action

                obs, combined_obs = self.prepare_obs(obs_dict)

                # --------- printing step-wise infomation --------- #
                print_stepwise_msg(self.algorithm_name, cavs_list, actions_env, true_actions, all_car_info_dict, \
                    episode, step, rewards[0], mean_rwd_items, list(rwd_items[:3])+[safety_rwds, encourage_rwds], Behavior)
                # --------- printing step-wise infomation --------- #


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
                    all_rewards = np.asarray(all_rewards)
                    episode_returns = all_rewards.sum(axis=0)
                    mean_episode_rewards = episode_returns.mean()
                    discounted_returns = compute_discounted_return(all_rewards, self.all_args.gamma)
                    avg_rwd_items = [flow_rwds, dest_rwds, cols_rwds, safe_rwds, enco_rwds]

                    train_info = {}
                    train_info.update({"avg_eps_rewards": mean_episode_rewards})
                    train_info.update({"avg_eps_flow_rewards": sum(flow_rwds)})
                    train_info.update({"avg_eps_dest_rewards": sum(dest_rwds)})
                    train_info.update({"avg_eps_cols_rewards": sum(cols_rwds)})
                    train_info.update({"avg_eps_safe_rewards": sum(safe_rwds)})
                    train_info.update({"avg_eps_enco_rewards": sum(enco_rwds)})
                    train_infos.append(train_info)

                    print_episode_msg(episode, self.envs.done_collision, (end-start)/60, episode_returns, avg_rwd_items, \
                        discounted_returns, all_flow_rewards, all_dest_rewards, self.all_args.flow_reward_coef)

                    episode_dict = group_episode_results(all_rewards, all_flow_rewards, all_dest_rewards, \
                        cols_rwds, safe_rwds, enco_rwds, self.all_args.flow_reward_coef, self.all_args.gamma)
                    episode_dict['switch'] = self.envs.switch

                    self.store_dict[episode] = episode_dict

                self.log_train(train_infos, episode)

            self.vid_2_idx = None
            self.idx_2_vid = None

            self.envs.close()

    @torch.no_grad()
    def render(self):        
        raise NotImplementedError

    def prepare_obs(self, obs_dict):
        obs = []
        for agent_id in range(self.num_agents):
            obs.append(obs_dict[self.idx_2_vid[agent_id]])
        
        combined_obs = deepcopy(obs)
        combined_obs = np.expand_dims(np.array(list(chain(*combined_obs))), axis=0)

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
        for agent_id in range(self.num_agents):
            vid = self.idx_2_vid[agent_id]
            agent_reward = (vels_rwds[vid] + dest_rwds[vid] + col_penalties[vid] + safety_rwds[vid] + encourage_rwds[vid]) * (1 - avg_coef) + \
                        (mean_vels_rwd + mean_dest_rwd + mean_cols_rwd + mean_safe_rwd + mean_encourage_rwd) * avg_coef

            rewards.append(agent_reward)
            flow_rewards.append(vels_rwds[vid])
            dest_rewards.append(dest_rwds[vid])

        rewards = np.expand_dims(np.array(rewards), axis=0)
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
