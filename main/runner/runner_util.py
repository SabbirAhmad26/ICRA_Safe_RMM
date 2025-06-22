from main.algorithms.utils.util import compute_discounted_return
from main.envs.carla.enums import Behavior
import numpy as np


def get_log_string(n_agents):
    STEP_LOG_STR = "Epi: {:4d}; Step: {:4d}; Rewards: #####; flow_r: {:.2f}; dest_r: {:.2f}; cols_r: {:.2f}; safe_r: {:.2f}; enco_r: {:.2f}"
    r_log_str = ",".join(["{:.2f}"]*n_agents)
    return STEP_LOG_STR.replace("#####", r_log_str)

def join_numbers(ns, fmt='{:.2f}'):
	return ', '.join([fmt.format(n) for n in ns])

def print_stepwise_msg(algorithm_name, cav_list, actions, true_actions, all_car_info_dict, episode, step, rewards, mean_rwd_items, rwd_items, Behavior):
    print('===============================================================================')
    # V_INFO_FMT_STR = "Car ID: {}; Location: ({:3.1f}, {:3.1f}); Road ID: {}; Lane ID: {}; Is_Junction: {}; Lane_type: {}; Lane_Change: {}; Velocity: {:.2f}km/h"
    CAV_INFO_FMT_STR = "Car ID: {}; Location: ({:3.1f}, {:3.1f}); Road ID: {}; Lane ID: {}; Is_Junction: {}; Rs: {:s}; Velocity: {:.2f}km/h"
    UCV_INFO_FMT_STR = "Car ID: {}; Location: ({:3.1f}, {:3.1f}); Road ID: {}; Lane ID: {}; Is_Junction: {}; Velocity: {:.2f}km/h"

    for vid, car in all_car_info_dict.items():
        if vid in cav_list:
            rwd_str = ', '.join(['{:.1f}'.format(rwd[vid]) for rwd in rwd_items])
            print(CAV_INFO_FMT_STR.format(vid, car['x'], car['y'], car['road_id'], \
                car['lane_id'], car['jct_id'], rwd_str, car['vel']))
        else:
            print(UCV_INFO_FMT_STR.format(vid, car['x'], car['y'], car['road_id'], car['lane_id'], car['jct_id'], car['vel']))
    chosen_action_str = "{} || ".format(algorithm_name)
    true_action_str = "True || "
    for i, v in actions.items():
        chosen_action_str += "Agent {}: {}; ".format(i, Behavior(v))
        true_action_str += "Agent {}: {}; ".format(i, Behavior(true_actions[i]))
    print(chosen_action_str)
    print(true_action_str)

    step_log_str = get_log_string(len(cav_list))
    print(step_log_str.format(episode, step, *rewards.tolist(), mean_rwd_items[0], \
        mean_rwd_items[1], mean_rwd_items[2], mean_rwd_items[3], mean_rwd_items[4]))


def print_episode_msg(episode, any_collision, episode_time, episode_returns, avg_rwd_items, discounted_returns, agent_flow_rs, agent_dest_rs, flow_reward_coef=1.):
	EPISODE_FMT_STR = "Episode {:d} any collision {}; Mean Return {:.2f}, flow {:.2f}, dest {:.2f}, cols {:.2f}, safe {:.2f}, enco {:.2f}, takes time {:.2f} minutes."
	RETURN_FMT_STR = "$Episode {:d}; Discounted Returns: {:s}; Returns: {:s}"
	FLOW_DEST_FMT_STR = "#Episode {:d}; Unweighted Flow+Dest: {:s}; Unweighted Flow: {:s}"
	FINAL_FD_FMT_STR = "&Episode {:d}; (Unweighted) Final Flow: {:s}; Final Destination: {:s}"
	
	rwd_items = [sum(avg_r_item) for avg_r_item in avg_rwd_items] + [0.] * max(5-len(avg_rwd_items), 0)
	mean_episode_return = episode_returns.mean()
	print(EPISODE_FMT_STR.format(episode, any_collision, mean_episode_return, rwd_items[0], \
		rwd_items[1], rwd_items[2], rwd_items[3], rwd_items[4], episode_time))

	agent_return_str = join_numbers(episode_returns)
	agent_disc_return_str = join_numbers(discounted_returns)
	print(RETURN_FMT_STR.format(episode, agent_disc_return_str, agent_return_str))

	unweighted_agent_flow_rs = np.array(agent_flow_rs) / flow_reward_coef
	unweighted_flow_returns = unweighted_agent_flow_rs.sum(axis=0)
	dest_returns = np.array(agent_dest_rs).sum(axis=0)
	final_flow, final_dest = unweighted_agent_flow_rs[-1], agent_dest_rs[-1]

	UF_D_str = join_numbers(unweighted_flow_returns+dest_returns)
	UF_str = join_numbers(unweighted_flow_returns)
	final_flow_str = join_numbers(final_flow)
	final_dest_str = join_numbers(final_dest)

	print(FLOW_DEST_FMT_STR.format(episode, UF_D_str, UF_str))
	print(FINAL_FD_FMT_STR.format(episode, final_flow_str, final_dest_str))
	print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


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
def group_episode_results(all_rewards, all_flow_rewards, all_dest_rewards, \
		avg_cols_rewards, avg_safe_rewards, avg_enco_rewards=None, flow_reward_coef=1.0, gamma=0.99):
	episode_dict = {}

	all_rewards = np.asarray(all_rewards)
	all_flow_rewards = np.asarray(all_flow_rewards)
	all_dest_rewards = np.asarray(all_dest_rewards)
	avg_cols_rewards = np.asarray(avg_cols_rewards)
	avg_safe_rewards = np.asarray(avg_safe_rewards)

	returns = all_rewards.sum(axis=0)
	unweighted_flow_returns = all_flow_rewards.sum(axis=0) / flow_reward_coef
	dest_returns = all_dest_rewards.sum(axis=0)
	discounted_returns = compute_discounted_return(all_rewards, gamma)

	episode_dict['rewards'] = all_rewards
	episode_dict['flow_rewards'] = all_flow_rewards
	episode_dict['dest_rewards'] = all_dest_rewards
	episode_dict['avg_cols_rewards'] = avg_cols_rewards
	episode_dict['avg_safe_rewards'] = avg_safe_rewards
	if avg_enco_rewards:
		avg_enco_rewards = np.asarray(avg_enco_rewards)
		episode_dict['avg_enco_rewards'] = avg_enco_rewards
	
	episode_dict['returns'] = returns
	episode_dict['unweighted_flow_returns'] = unweighted_flow_returns
	episode_dict['dest_returns'] = dest_returns
	episode_dict['discounted_returns'] = discounted_returns

	return episode_dict
