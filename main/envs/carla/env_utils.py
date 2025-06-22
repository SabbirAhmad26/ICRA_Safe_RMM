#!/usr/bin/env python
from statistics import mean, stdev
import time
import os
import random
import math
import numpy as np
# import pandas as pd
from copy import deepcopy
from math import sin, cos, tan, atan
from .tools.misc import angle_between_vec
#from .tools.gen_scenegraph import SceneGraph

reference_speed = 27.
# reference_speed = 30.

'''
    config: {
        "Base_locations": ,
        "Random_loc":,
        "Target_speeds": ,
        "Destination"
    }
'''

def compute_collision_penalty(collisions, cav_ids, vids, col_r=True, include_env_collision=True):
    cav_collision_dict = dict.fromkeys(cav_ids, False)
    col_penalties = dict.fromkeys(cav_ids, 0.)
    accumulated_collision_penalty = 0.
    has_collision = False
    
    if not collisions:
        return col_penalties, cav_collision_dict, accumulated_collision_penalty, has_collision
    
    for col_intensity, id1, id2 in collisions:
        # all agents suffer the same from any collision with any CAV
        if (not include_env_collision) and ((id1 not in vids) or (id2 not in vids)):
            continue

        accumulated_collision_penalty -= col_intensity
        
        if id1 in cav_ids:
            col_penalties[id1] -= col_intensity
            cav_collision_dict[id1] = True
        
        if id2 in cav_ids:
            col_penalties[id2] -= col_intensity
            cav_collision_dict[id2] = True

        has_collision = True

    if not col_r:
        col_penalties = dict.fromkeys(cav_ids, 0.)

    return col_penalties, cav_collision_dict, accumulated_collision_penalty, has_collision




def spawn_from_config(config_dict):
    base_locs = config_dict["Base_locations"]
    rnd_locs = config_dict["Random_loc"]
    spawn_locs = []
    for base, rnd in zip(base_locs, rnd_locs):
        for i in range(len(base)):
            if rnd[i] != 0:
                base[i] += np.random.rand()*rnd[i] - rnd[i]/2.0
        spawn_locs.append(base)
    return spawn_locs

def spawn_from_config_new(config_dict, n_cav, n_ucv, scenario='highway', episode_length=240, cav_force_straight_steps=0):
    cav_candidate_locs = config_dict["CAV_Base_loc"]
    cav_rand_locs = config_dict["CAV_Rand_loc"]
    cav_candidate_dest = config_dict["CAV_Destination"]
    ucv_candidate_locs = config_dict["UCV_Base_loc"]
    ucv_rand_locs = config_dict["UCV_Rand_loc"]

    switch = False
    # in some cases, we switch the initial points of cav with ucvs
    if "allow_switch" in config_dict and config_dict["allow_switch"] and (random.random() < 0.05):
        cav_candidate_locs = config_dict["UCV_Base_loc"]
        ucv_candidate_locs = config_dict["CAV_Base_loc"]
        switch = True
        print("Switch the initialization between CAV and UCV")
    
    if scenario == 'highway':
        ucv_2nc_candidate_locs, ucv_2nd_rand_locs = config_dict["UCV_2nd_Base_loc"], config_dict["UCV_2nd_Rand_loc"]
    else:
        ucv_2nc_candidate_locs, ucv_2nd_rand_locs = [], []

    assert n_cav <= len(cav_candidate_locs)
    assert n_ucv <= len(ucv_candidate_locs) + len(ucv_2nc_candidate_locs)

    spawn_locs = []
    cav_destinations = []
    for idx in random.sample(range(len(cav_candidate_locs)), n_cav):
        base, rnd, dest = cav_candidate_locs[idx], cav_rand_locs[idx], cav_candidate_dest[idx]
        for i in range(len(base)):
            if rnd[i] != 0:
                base[i] += np.random.rand() * rnd[i] - rnd[i] / 2.0
        spawn_locs.append(base)
        # compute the destination
        # with force straight steps of CAV, need to modify this
        base_copy = deepcopy(base)
        # if we force cav to drive at the begining, then we should set the destination to be further
        base_copy[0] += dest * (episode_length + cav_force_straight_steps) / config_dict["Default_Steps"]
        cav_destinations.append(base_copy)


    if n_ucv >= 4:
        assert scenario == 'highway'
        n_ucv_2nd = n_ucv // 2
        n_ucv -= n_ucv_2nd

        for idx in random.sample(range(len(ucv_2nc_candidate_locs)), n_ucv_2nd):
            base, rnd = ucv_2nc_candidate_locs[idx], ucv_2nd_rand_locs[idx]
            for i in range(len(base)):
                if rnd[i] != 0:
                    base[i] += np.random.rand() * rnd[i] - rnd[i] / 2.0
            spawn_locs.append(base)

    for idx in random.sample(range(len(ucv_candidate_locs)), n_ucv):
        base, rnd = ucv_candidate_locs[idx], ucv_rand_locs[idx]
        for i in range(len(base)):
            if rnd[i] != 0:
                base[i] += np.random.rand() * rnd[i] - rnd[i] / 2.0
        spawn_locs.append(base)

    return spawn_locs, cav_destinations, switch

def generate_safe_action(vids, n_discrete=3, mute_lane_change=False, simple_n=-1):
    safe_action_dict = {}
    for vid in vids:
        if simple_n > 2:
            if mute_lane_change:
                safe_action_dict[vid] = [0,0] + [1] * (simple_n-2) # left_lc, right_lc, KL_0, KL_1, ...
            else:
                safe_action_dict[vid] = [1] * simple_n

        elif mute_lane_change:
            safe_action_dict[vid] = [1,0,0,1] + [1] * (2*n_discrete)
        else:
            safe_action_dict[vid] = [1] * (4 + 2 * n_discrete)
    return safe_action_dict

'''
    inputs:
        poly_dict: defining all the v-a curves for throttle value in [-1, 1]
        v: current velocity of the vehicle
        ranges: a list of ranges in the form of tuples: [(-50, 0)] 
'''
def find_max_min_acc(poly_dict, v, ranges):
    minmax = []
    for start, end in ranges:
        param = poly_dict[start]
        acc = np.poly1d(param)(v)
        acc_min, acc_max = acc, acc
        for i in range(start+1, end):
            param = poly_dict[i]
            acc = np.poly1d(param)(v)
            acc_min = min(acc_min, acc)
            acc_max = max(acc_max, acc)
        minmax.append((acc_min, acc_max))
    return minmax

def find_acc(poly_dict, v, throttle=0):
    tb = max(min(throttle, 1), -1)
    tb = int(throttle * 100)
    param = poly_dict[tb]
    acc = np.poly1d(param)(v)
    return acc

def search_throttle_value(poly_dict, target_acc, v, acc_range=None):
    if acc_range:
        start, end = acc_range
        param = poly_dict[start]
        acc = np.poly1d(param)(v)
        acc_error = abs(target_acc-acc)
        target_throttle = start
        for i in range(start+1, end):
            param = poly_dict[i]
            acc = np.poly1d(param)(v)
            if abs(target_acc-acc) < acc_error:
                acc_error = abs(target_acc-acc)
                target_throttle = i
        return target_throttle / 100.
    else:
        acc_error = 10000.
        target_throttle = 0.
        for i, param in poly_dict.items():
            acc = np.poly1d(param)(v)
            if abs(target_acc-acc) < acc_error:
                acc_error = abs(target_acc-acc)
                target_throttle=i
        return target_throttle / 100.

def compute_safe_dist(c1_info, c2_info, base_SD=1., max_SD=5.):
    # loc 1-> 2; vel 1-> 2
    x1, y1, vx1, vy1 = c1_info
    x2, y2, vx2, vy2 = c2_info
    dloc=(x2 - x1, y2 - y1)
    dv = (vx1 - vx2, vy1 - vy2)
    phi = angle_between_vec(dloc, dv)
    return base_SD + max_SD * cos(phi)


# return a (steps, 2) vector 
def compute_trajectory(loc, vel, acc, steps=20, dt=0.05):
    px = np.poly1d([0.5*acc[0], vel[0], loc[0]])
    py = np.poly1d([0.5*acc[1], vel[1], loc[1]])
    #px = np.poly1d([vel[0], loc[0]])
    #py = np.poly1d([vel[1], loc[1]])
    ts = np.linspace(dt, steps*dt, steps)
    trajectory = np.vstack((px(ts), py(ts))).transpose() # (step, 2) np array
    return trajectory

def generate_error_new(cav_ids, all_ids, e_type, error_bound, scenario, current_step=None, target_start_end=None, error_pos=None):
    assert e_type in ['noise', 'tgt_v', 'tgt_t']
    # error_bound = [err_x, err_y, err_vx, err_vy]
    assert isinstance(error_bound, list) and len(error_bound)==4

    # error_pos is a dict {vid: [+-] * 4} defining whether the error is positive or negative
    # the purpose is to output more consistent error in simulation
    if e_type == 'tgt_t':
        assert (current_step >= 0) and (target_start_end) and error_pos
    elif e_type == 'tgt_v':
        assert error_pos and all([v in all_ids for v in error_pos])
    error = {}
    error['type'] = e_type

    for vid in all_ids:
        bound = error_bound
        if scenario == 'crossing' and (vid not in cav_ids):
            bound = [error_bound[1], error_bound[0], error_bound[3], error_bound[2]]

        if e_type == 'noise':
            error[vid] = [random.uniform(-b, b) for b in bound]

        elif e_type == 'tgt_v':
            # when choosing tgt_v - targeting a few vehicle to perturb entire episode
            # the error_pos doesn't contain all vids, but only the targeted ones
            if vid in error_pos:
                error[vid] = [pos * random.uniform(0.8*b, b) for pos, b in zip(error_pos[vid], bound)]
            else:
                error[vid] = [0.,0.,0.,0.]

        elif e_type == 'tgt_t':
            if target_start_end[0] <= current_step < target_start_end[1]:
                error[vid] = [pos * random.uniform(0.8*b, b) for pos, b in zip(error_pos[vid], bound)]
            else:
                error[vid] = [0.,0.,0.,0.]

        else:
            raise NotImplementedError

    old_error = {'type': e_type}
    
    for cav_id in cav_ids:
        ego_cav_err = {}
        for vid in all_ids:
            if vid == cav_id:
                ego_cav_err[vid] = [0., 0.]
            elif (vid not in cav_ids) and scenario == 'crossing':
                ego_cav_err[vid] = [error[vid][1], error[vid][3]]
            else:
                ego_cav_err[vid] = [error[vid][0], error[vid][2]]
        old_error[cav_id] = ego_cav_err

    return error, old_error



'''
    Method used to generate error given the connected autonomous vehicle and all cars' ids
        Input: cav_ids, all_ids
                n_type: ['noise', 'miss']
'''
def generate_error(cav_ids, all_ids, e_type='noise', noise_std=3., miss_rate=0.1, \
                    current_step=None, target_start_end=None, target_vehicles=None):
    assert e_type in ['noise', 'miss', 'l_attack', 'f_attack', 'critic', 'tgt_v', 'tgt_t']

    if e_type == 'tgt_t':
        assert (current_step >= 0) and (target_start_end) and target_vehicles
    elif e_type == 'tgt_v':
        assert target_vehicles and all([v in all_ids for v in target_vehicles])
    error = {}
    error['type'] = e_type
    for cav_id in cav_ids:
        if e_type == 'miss':
            detected_cars_id = [cav_id]
            for temp_id in all_ids:
                if (temp_id != cav_id) and (np.random.rand()>= miss_rate):
                    detected_cars_id.append(temp_id)
            error[cav_id] = detected_cars_id
        elif e_type in ['noise', 'l_attack', 'f_attack']:
            all_car_noises = {}
            for vid in all_ids:
                if vid == cav_id:
                    all_car_noises[vid] = (0., 0.)
                else: # perturb the x location and velocity
                    if e_type == 'noise':
                        xv_error = (random.uniform(-noise_std, noise_std), random.uniform(-noise_std, noise_std))
                    elif e_type == 'l_attack':
                        xv_error = (random.gauss(10, noise_std), 0)
                    else:
                        xv_error = (random.gauss(12, noise_std), 0)
                    all_car_noises[vid] = xv_error
            error[cav_id] = all_car_noises
        
        elif e_type == 'tgt_t':
            all_car_noises = {}
            for vid in all_ids:
                if vid == cav_id:
                    #all_car_noises[vid] = (0,0)
                    pass
                elif current_step >= target_start_end[0] and current_step < target_start_end[1]:
                    #xv_error = (random.gauss(target_vehicles[vid][0], 1.), \
                    #            random.gauss(target_vehicles[vid][1], 1.))
                    all_car_noises[vid]  = (random.uniform(target_vehicles[vid][0]-0.5, target_vehicles[vid][0]+0.5),
                                            random.uniform(target_vehicles[vid][1]-0.5, target_vehicles[vid][1]+0.5))
                else:
                    #all_car_noises[vid] = (0,0)
                    pass
            error[cav_id] = all_car_noises

        #Attact targeting time frame. assume 
        elif e_type == 'tgt_v':
            # target_vehicles = {vid: (base_perturbation_x, bp_v)}
            all_car_noises = {} # this is observation received by agent cav_id
            for vid in all_ids:
                if vid == cav_id: # self observation is correct
                    #all_car_noises[vid] = (0, 0)
                    pass
                elif vid in target_vehicles: # we only attack target vehicle
                    #all_car_noises[vid] = (random.gauss(target_vehicles[vid][0], 1.), \
                    #                       random.gauss(target_vehicles[vid][1], 1.))
                    all_car_noises[vid]  = (random.uniform(target_vehicles[vid][0]-1, target_vehicles[vid][0]+1),
                                            random.uniform(target_vehicles[vid][1]-1, target_vehicles[vid][1]+1))
                else:
                    #all_car_noises[vid] = (0,0)
                    pass
            error[cav_id] = all_car_noises
        elif e_type == 'critic':
            raise NotImplementedError
        else:
            raise NotImplementedError
    return error

def observation_from_state(state_dict, CAV_ids, max_num_car=10, brief_state=False, error=None, debug=False):
    # if brief_state: length 7
    #   state = [dx, dy, dvx, dvy, ax, ay, yaw] + [flag]
    # else: # length 16
    #   state = [x,y,z, vx, vy, vz, ax, ay, az, flag_v, flag_cav, yaw] + [dx, dy, dvx, dvy]

    obs_dict = {}
    if brief_state:
        max_obs_dim = max_num_car * 8
    else:
        max_obs_dim = max_num_car * 16

    state_dict = {vid: state_dict[vid] for vid in sorted(state_dict)}
    
    for vid in CAV_ids:
        ego_obs = deepcopy(state_dict[vid])
        # temp_cav_error = None
        # if error and (error['type'] in ['noise', 'tgt_v', 'tgt_t']):
        #     temp_cav_error = error[vid]

        ego_x, ego_y, ego_vx, ego_vy = ego_obs[0:2] + ego_obs[3:5]
        
        if brief_state:
            ego_obs = ego_obs[0:2] + ego_obs[3:5] + ego_obs[6:8] + ego_obs[11:12] + [1.]
        else:
            ego_obs = ego_obs[:9] + [1.0, 1.0] + ego_obs[11:12] + [0.] * 4
        for other_vid, other_state in state_dict.items():
            if other_vid != vid:
                cav_flag = 1. if other_vid in CAV_ids else 0.
                other_obs = deepcopy(other_state)
                if error:
                    other_obs[0] += error[other_vid][0]
                    other_obs[1] += error[other_vid][1]
                    if brief_state:
                        other_obs[2] += error[other_vid][2]
                        other_obs[3] += error[other_vid][3]
                    else:
                        other_obs[3] += error[other_vid][2]
                        other_obs[4] += error[other_vid][3]

                if brief_state:
                    other_obs = other_obs[0:2]+other_obs[3:5]+other_obs[6:8]+other_obs[11:12] + [1.0] + [cav_flag]
                else:
                    other_obs = other_obs[:9] + [1.0, cav_flag] + other_obs[11:12] + \
                                 [other_obs[0] - ego_x, other_obs[1] - ego_y, \
                                  other_obs[3] - ego_vx, other_obs[4] - ego_vy]
                ego_obs += other_obs
        
        if len(ego_obs) >= max_obs_dim:
            ego_obs = ego_obs[:max_obs_dim]
        else:
            ego_obs = ego_obs + [0.] * (max_obs_dim-len(ego_obs))

        obs_dict[vid] = ego_obs

    return obs_dict

def combine_obs_dicts(obs_dict_list):
    combined_dict = dict.fromkeys(obs_dict_list[0], [])
    for obs in obs_dict_list:
        for vid in obs:
            combined_dict[vid] = combined_dict[vid] + obs[vid]
    return combined_dict

def combine_safe_actions(dict1, dict2):
    safe_actions = {}
    for vid, s1 in dict1.items():
        s2 = dict2[vid]
        safe_actions[vid] = [a*b for a,b in zip(s1, s2)]
    return safe_actions

