from statistics import mean, stdev
import glob
import time
import os
import random
import math
import sys
import pickle
import traceback
import queue
import json
import numpy as np
from gymnasium import spaces
from copy import deepcopy
from .ipc_utils import (
    camera_handler, lidar_handler,
    process_raw_RGBA, process_raw_PC,
    ImageServer, PointCloudServer,
)
from math import sin, cos, tan, atan
from cvxopt import matrix, solvers, spdiag
from .cav_behavior_planner import CAVBehaviorPlanner
from .controller import PIDLateralController
from .tools.misc import draw_points, angle_between_vec
from .env_utils import (
    find_max_min_acc, 
    find_acc, 
    search_throttle_value,
    compute_safe_dist,
    compute_trajectory,
    generate_error_new,
    observation_from_state,
    combine_safe_actions,
    spawn_from_config,
    spawn_from_config_new,
    generate_safe_action,
    compute_collision_penalty,
    reference_speed
)
#from .tools.gen_scenegraph import SceneGraph
import carla
from .enums_old import V_Type
from .enums_old import Behavior

#np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(precision=3)

def collision_handler(event, col_queue):
    actor_id = event.actor.id
    actor_against_id = event.other_actor.id
    impulse = event.normal_impulse
    intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    print("Vehicle {:d} collide with vehicle: {:d}; intensity: {:.2f}".\
        format(actor_id, actor_against_id, intensity))
    col_queue.put((intensity, actor_id, actor_against_id))

class CarEnv:

    def __init__(self, args, num_CAVs = 3):
        
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(10.0)
        self.map_name = args.map

        self.world = self.client.load_world(self.map_name)
        #self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.map = self.world.get_map()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.timestep = args.timestep
        settings.fixed_delta_seconds = self.timestep
        settings.no_rendering_mode = args.no_render
        self.world.apply_settings(settings)
        self.spec_view = args.spec_view
        self.scenario = None
        self.normal_behave = args.normal_behave
        self._carla = carla

        self.episode_length = args.episode_length
        self.flow_reward_coef = self.args.flow_reward_coef
        self.cav_force_straight_step = self.args.cav_fs_step
        self.ucv_force_straight_step = self.args.ucv_fs_step
        self.brief_state = args.brief_state
        self.discretize = args.discretize
        self.e_type = args.e_type
        
        self.num_Vs = args.cars
        self.num_CAVs = num_CAVs
        self.safety_step = args.safety_step
        self.mute_lane_change = args.disable_lane_chg
        self.disable_CBF = args.disable_CBF
        self.remove_CBFA = args.remove_CBFA
        self.e_type = args.e_type
        self.test_only = args.test_only
        self.debug = args.debug
        self.cbf_robustness = args.cbf_robustness
        self._policies = None

        # env params
        if self.brief_state:
            base_state_length = 7
            self.max_num_car = 6
        else:
            base_state_length = 16
            self.max_num_car = 8
        self.observation_space = dict.fromkeys(range(num_CAVs), [base_state_length*self.max_num_car])
        self.share_observation_space = dict.fromkeys(range(num_CAVs), [base_state_length*self.max_num_car*num_CAVs])
        self.action_space = dict.fromkeys(range(num_CAVs), spaces.Discrete(4 + 2 * self.discretize))

        # Set up a Carla blueprint for a Tesla Model 3
        self.blueprint_library = self.world.get_blueprint_library()
        self.model3_cav = self.blueprint_library.filter("model3")[0]
        self.model3_cav.set_attribute('color', '0,255,0')
        self.model3_ncav = self.blueprint_library.filter("model3")[0]
        self.model3_ncav.set_attribute('color', '255,0,0')
        self.audi_tt = self.blueprint_library.find('vehicle.audi.tt')
        self.audi_tt.set_attribute('color', '255,0,0')

        self.tm_port = 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)

        # Instantiate state variables
        
        self.vehicle_list    = []
        self.colsensor_list  = []  # todo: unused
        self.collision_queue = queue.Queue()
        self.accumulated_collision_penalty = 0.
        self.CAV_agents      = []
        self.CAV_agents_dict = {}

        # explicitly denote the hazard vehicles inside the env
        #self.hazd_vehicles = None
        #self.hazd_states = None
        #self.hazd_steer_controllers = None
        #self.hazd_throttle = 0.

        self.ncav_vehicles = None # store as a dict: {id: vehicle_obj}
        self.ncav_states = None # store as a dict: {id: state_info}
        self.ncav_steer_controllers = None # store as a dict: {id: controller}
        self.ncav_brake = 0.
        self.hazard_ncav_ids = None
        
        # create dictionaries to store observations and actions
        self.state = None
        self.action = {}
        self.neighbor = {}
        self.speed_dict = {}
        self.allowed_behaviors_dict = {}
        self.cbf_safe_action_dict = {}
        

        # destination info
        self.destination = []

        # collision flag
        self.done_collision = False

        # statistics
        #self.CAV_v_fwds = 0
        #self.CAV_comfort = 0

        # parameters for driving algorithms
        # todo- consider a dataclass to hold these, simpler access
        self.param_dict = {
            'Tds':          10,     # check for lane changes once every Tds timesteps
            'F':            50,     # past timesteps to remember for Qf quality factor
            'w':            0.4,    # weight of Qv in lane change reward function
            'theta_CL':     2.0,    # reward function threshold to switch left
            'theta_CR':     2.0,    # reward function threshold to switch right
            'eps':          150.0,  # [m] communication radius for connected vehicles
            'theta_a':      2.0,    # [m/s^2] "uncomfortable" acceleration threshold
            'p_l':          0.01,   # random change left probability
            'p_r':          0.01,   # random change right probability
            'theta_l':      4.0,    # [m] safety radius to prevent left changes
            'theta_c':      10.0,   # [m] safety cushion in front of vehicle
            #'theta_c':      12.0,   # [m] safety cushion in front of vehicle
            'theta_r':      4.0,    # [m] safety radius to prevent right changes
            'chg_distance': 20.0}   # [m] longitudinal distance to travel while changing

        with open('./new_polys.pkl', 'rb') as fp:
            poly_dict = pickle.load(fp)
            fp.close()
        self.poly_dict = poly_dict

        ranges = [(-50, 0)]
        for i in range(self.discretize):
            acc_start = 60 + (i*40)//self.discretize
            acc_end = 60 + ((i+1)*40)//self.discretize
            brk_start = (i*60)//self.discretize
            brk_end = ((i+1)*60)//self.discretize
            ranges.append((acc_start, acc_end))
            ranges.append((brk_start, brk_end))
        self.ranges = ranges
        solvers.options['show_progress'] = False

    def set_policies(self, policies):
        self._policies = policies

    # update true state for each vehicle (cav, ncav, hazv)
    # always store the correct information
    def _update_each_vehicle_true_state(self, as_dict=False):
        all_car_info_dict = {}
        for vid, v_type, vehicle in self.vehicle_list:
            car_state = []
            car_transform = vehicle.get_transform()
            car_loc, car_rot = car_transform.location, car_transform.rotation
            car_vel = vehicle.get_velocity()
            car_acc = vehicle.get_acceleration()
            car_state += [car_loc.x, car_loc.y, car_loc.z]
            car_state += [car_vel.x, car_vel.y, car_vel.z]
            car_state += [car_acc.x, car_acc.y, car_acc.z]
            car_state += [car_rot.roll / 360., car_rot.pitch / 360., car_rot.yaw / 360.]

            kmh_velocity = math.sqrt(car_vel.x**2 + car_vel.y**2) * 3.6
            wp = self.map.get_waypoint(car_loc)
            jct_id = -1 if not wp.is_junction else wp.get_junction().id
            car_info_dict = {'id': vid, 'x': car_loc.x, 'y': car_loc.y,
                             'vel': kmh_velocity, 'road_id': wp.road_id,
                             'lane_id': wp.lane_id, 'jct_id': jct_id, 
                             'lane_type': wp.lane_type, 'lane_chg': str(wp.lane_change)}
            all_car_info_dict[vid] = car_info_dict

            if v_type == V_Type.CAV:
                self.state[vid] = car_state
                self.speed_dict[vid] = kmh_velocity
            elif v_type == V_Type.HAZV: # start to abandon this HAZC setup
                self.hazard_v_state = car_state
            elif v_type == V_Type.NCAV:
                self.ncav_states[vid] = car_state
            else:
                raise RuntimeError("Unrecognzied V_TYPE!")

        return all_car_info_dict

    
    def _null_cbf_safe_action_dict(self):
        temp_dict = {}
        for agent in self.CAV_agents:
            temp_dict[agent.vehicle.id] = [1]*(4 + 2 * self.discretize)
        return temp_dict
    

    def _get_reward(self, chosen_actions, true_actions, avg_p=1.0, col_r=True, include_env_collision=True, interpolate=1.0): # total_col_intensity=0, locs=None)
        """
        individual_rwd = w_1 * ind_speed + w_2 * col_penalty + w_3 * loc_penalty
        mean_rwd = mean([individual_rwds])
        
        :param avg_p: the percentage of average reward in individual return
        :type avg_p:  Float, [0,1]
        :param col_r: whether consider the collision reward
        :type col_r:  Bool
        :returns: 
            1. [weighted_agent_reward] = aa * mean_rwd + (1-aa) * [individual_reward]
            2. (mean(flow), mean(col_penalty), mean(loc_penalty))
        """
        
        collisions = []
        while not self.collision_queue.empty():
            col_intensity, id_1, id_2 = self.collision_queue.get()
            collisions.append((col_intensity/40. + 1, id_1, id_2))
        #print("!!!!!!!!!!!!COL LENGTH", len(collisions))
        cav_ids = self.CAV_agents_dict.keys()
        vids = [v[0] for v in self.vehicle_list]
        
        col_penalties, cav_collision_dict, accumulated_collision_penalty, has_collision = \
            compute_collision_penalty(collisions, cav_ids, vids, col_r, include_env_collision)

        self.done_collision = self.done_collision or has_collision

        vels_rwds = []
        dest_rwds = []
        safe_act = []
        flow_dict, dest_dict, sact_dict = {}, {}, {}
        
        for vid in self.CAV_agents_dict:
            vels_rwds.append((0.2 * self.flow_reward_coef) * (self.speed_dict[vid] - reference_speed) / 3.6)
            loc = self.state[vid][0:3]
            spawn, desti = self.spawn_dict[vid], self.desti_dict[vid]
            dest_rwds.append((loc[0] - spawn[0]) / ((desti[0] - spawn[0]) * interpolate))

            true_action = true_actions[vid].value
            if true_action == -1:
                safe_act.append(-0.3)
            elif true_action == chosen_actions[vid]:
                safe_act.append(0.1)
            else:
                safe_act.append(0.)
            flow_dict[vid] = vels_rwds[-1]
            dest_dict[vid] = dest_rwds[-1]
            sact_dict[vid] = safe_act[-1]

        flow_reward = np.array(vels_rwds)
        dest_reward = np.array(dest_rwds)
        cols_reward = np.array(list(col_penalties.values()))
        sact_reward = np.array(safe_act)
        
        mean_flow_rwd = np.mean(flow_reward)
        mean_dest_rwd = np.mean(dest_reward)
        mean_cols_rwd = np.mean(cols_reward)
        mean_sact_rwd = np.mean(sact_reward)

        rewards = avg_p * (mean_flow_rwd + mean_dest_rwd) + \
                  (1 - avg_p) * (flow_reward + dest_reward) + \
                  (cols_reward + sact_reward)

        reward_items = flow_dict, dest_dict, col_penalties, sact_dict
        mean_rwd_items = [mean_flow_rwd, mean_dest_rwd, mean_cols_rwd, mean_sact_rwd]

        # rewards = avg_p * (mean_flow_rwd + mean_dest_rwd + mean_cols_rwd + mean_sact_rwd) + \
        #          (1 - avg_p) * (flow_reward + loc_penalties + col_penalties + sact_reward)
        return rewards, reward_items, mean_rwd_items, flow_reward, dest_reward, cav_collision_dict

    # behavior candidates: acc * x, brk * x
    # change left, change right
    # keep lane
    def predict_traj(self, loc, vel, behavior_value, brk_range=0.6, discretize=2, steps=20, dt=0.05, single=True):
        if behavior_value == -1:
            param = self.poly_dict[-100]
            vel_scalar = math.sqrt(vel[0]**2 + vel[1]**2)
            acc = np.poly1d(param)(vel_scalar)
            if vel_scalar < 0.1:
                return [compute_trajectory(loc, vel, [acc, 0], steps, dt)]
            return [compute_trajectory(loc, vel, [acc * vel[0] / vel_scalar, \
                acc * vel[1] / vel_scalar], steps, dt)]

        elif behavior_value < 3:
            return [compute_trajectory(loc, vel, [0,0], steps, dt)]
        
        vel_scalar = math.sqrt(vel[0]**2 + vel[1]**2)
        level = (behavior_value - 4) // 2
        if behavior_value == 3:
            if single:
                candidate_throttles = [-0.25]
            else:
                candidate_throttles = [-0.5, -0.25, -0.01]
                candidate_throttles.append(np.random.random_sample()*0.5 - 0.5)
                #expect_throttle = 
        else:            
            if behavior_value % 2 == 0:
                if single:
                    candidate_throttles = [brk_range + (level + 0.5) * (1 - brk_range) / discretize]
                else:
                    temp_range = (brk_range + level*(1 - brk_range)/discretize, brk_range + (level+1)*(1 - brk_range)/discretize)
                    candidate_throttles = [temp_range[0], temp_range[1], (temp_range[0]+temp_range[1])/2.]
                    candidate_throttles.append(np.random.random_sample()*(temp_range[1]-temp_range[0]) + temp_range[0])
                    #expect_throttle = brk_range + (level + 0.5) * (1 - brk_range) / discretize
            else:
                if single:
                    candidate_throttles = [(level + 0.5) * brk_range / discretize]
                else:
                    temp_range = (level * brk_range/discretize, (level+1)*brk_range/discretize)
                    candidate_throttles = [temp_range[0], temp_range[1], (temp_range[0]+temp_range[1])/2.]
                    candidate_throttles.append(np.random.random_sample()*(temp_range[1]-temp_range[0]) + temp_range[0])
                    #expect_throttle = (level + 0.5) * brk_range / discretize
        
        candidate_trajs = []
        for throttle_v in candidate_throttles:
            param = self.poly_dict[int(100*throttle_v)]
            acc = np.poly1d(param)(vel_scalar)
            if vel_scalar < 0.1:
                candidate_trajs.append(compute_trajectory(loc, vel, [acc, 0], steps, dt))
            else:
                candidate_trajs.append(compute_trajectory(loc, vel, [acc*vel[0]/vel_scalar, acc*vel[1]/vel_scalar], steps, dt))

        return candidate_trajs
    
    def robust_cbf(self, ego_id, cbf_state, neighbors, other_vehicles=None, \
                    use_zeta=True, max_zeta=1., base_SD=15., safe_eps_1=0.4, \
                    safe_eps_2=0.4, Ts=0.05, eta=0.95, error_y=2., max_steering_angle=70, \
                    stable_steering_angle=2):
        x, y, psi, v = cbf_state
        #eps = safe_factor
        max_neg_acc = find_acc(self.poly_dict, v, throttle=-1)
        #print('CBF state', cbf_state)
        '''
        front = 'cf' in neighbors
        back = 'cb' in neighbors
        left_front = 'lf' in neighbors
        left_back = 'lb' in neighbors
        right_front = 'rf' in neighbors
        right_back = 'rb' in neighbors
        '''
        '''
        -eta*h(x) <= dh/dt + Lfh(x) + Lgh(x)u - a(y)
        NEW version: 
        dist(x, x_ft)
        dist(x_bt, x)
        '''
        keys = ['cf', 'cb', 'lf', 'lb', 'rf', 'rb']
        # define constraints
        # G(u) <= h
        Gs = {}
        hs = {}
        #print("{} neighbors: {}".format(str(ego_id), str(neighbors)))
        for key in keys: #keys for neighbors 
            if key in neighbors:
                if key[-1]=='f': # target v is a front vehicle
                    target_vid, d_x, v_ft, a_ft = neighbors[key] # the first term is already relative distance
                    if self.remove_CBFA:
                        a_ft = 0.

                    max_neg_acc_ft = find_acc(self.poly_dict, v_ft, throttle=-1)
                    max_pos_acc_ft = find_acc(self.poly_dict, v_ft, throttle=0.99)
                    '''
                        dh_dt + Lfh + Lgh u - a(y) >= -eta * h_x
                        -Lgh * u <= dh_dt + Lfh + eta * h_x - a_y
                        h_x = x_ft - x - Safe_Distance
                    '''
                    safe_distance = safe_eps_1 * v + safe_eps_2 * (pow(v,2)/(2*abs(max_neg_acc)) - pow(v_ft,2)/(2*abs(max_neg_acc_ft))) + base_SD
                    if self.debug:
                    #if True:
                        print("CBF {} Front-{} {} safe_distance: ".format(str(ego_id), key, str(target_vid)), safe_distance)

                    dh_dt = v_ft + safe_eps_2 * v_ft * a_ft / abs(max_neg_acc_ft)
                    Lfh = -v * cos(psi)
                    Lgh = [- Ts * cos(psi) - safe_eps_1 - safe_eps_2*v/abs(max_neg_acc), v * sin(psi)]
                    h_x = d_x - safe_distance
                    '''
                        lipschitz_dhdt = max(a_ft + safe_eps_2/abs(max_neg_acc_ft) * a_ft^2)
                        lipschitz_eta_h = eta * (max(1 + safe_eps_2* v_ft/abs(max_neg_acc_ft)))
                    '''
                    lipschitz_dh_dt = 1 + safe_eps_2 * abs(max_pos_acc_ft) / abs(max_neg_acc_ft)
                    lipschitz_eta_h = eta * (1 + v_ft * (safe_eps_2 / abs(max_neg_acc_ft) ))
                    a_y = (lipschitz_dh_dt + lipschitz_eta_h) * error_y
                    #if True:
                    #    print("CBF {} Front-{} {} lips term: ".format(str(ego_id), key, str(target_vid)), a_y, lipschitz_dh_dt, lipschitz_eta_h)

                    Gs[key] = np.array([-Lgh[0], -Lgh[1], 0., 0., 0., 0.])
                    hs[key] = np.array([dh_dt + Lfh + eta*h_x - a_y])
                    #print(-Lgh[0], -Lgh[1], dh_dt, Lfh, eta*h_x)
                else: # target v is a back vehicle
                    target_vid, d_x, v_bt, a_bt = neighbors[key]
                    if self.remove_CBFA:
                        a_bt = 0.
                    
                    max_pos_acc_ft = find_acc(self.poly_dict, v_bt, throttle=0.99)
                    max_neg_acc_bt = find_acc(self.poly_dict, v_bt, throttle=-1)
                    '''
                        h_x = x - x_bt - Safe_Distance
                    '''
                    safe_distance = safe_eps_1 * v_bt + safe_eps_2 * (pow(v_bt,2)/(2*abs(max_neg_acc_bt)) - pow(v,2)/(2*abs(max_neg_acc))) + base_SD
                    if self.debug:
                    #if True:
                        print("CBF {} Front-{} {} safe_distance: ".format(str(ego_id), key, str(target_vid)), safe_distance)

                    dh_dt = - v_bt - safe_eps_1 * a_bt - safe_eps_2 * v_bt * a_bt / abs(max_neg_acc_bt)
                    Lfh = v * cos(psi)
                    Lgh = [Ts * cos(psi) + safe_eps_2*v/abs(max_neg_acc), -v * sin(psi)]
                    h_x = - d_x - safe_distance
                    '''
                        lipschitz_dhdt = max(||-a_bt - safe_eps_2/abs(max_neg_acc_bt) * a_bt^2 ||)
                        lipschitz_eta_h = eta * (max(||-1 - safe_eps_1 - safe_eps_2 * v_bt / abs(max_neg_acc_bt) ||))
                    '''
                    #lipschitz_dh_dt = abs(max_neg_acc_bt)
                    # or alternatively: lipschitz_eta_h = eta * (1 + safe_eps_1 + safe_eps_2* 10. / abs(max_neg_acc_bt))
                    #lipschitz_eta_h = eta * (1 + safe_eps_1)

                    lipschitz_dh_dt = (1 + safe_eps_2 * abs(max_pos_acc_ft) / abs(max_neg_acc_bt))
                    lipschitz_eta_h = eta * (1 + safe_eps_1 + safe_eps_2 * v_bt / abs(max_neg_acc_bt))
                    a_y = (lipschitz_dh_dt + lipschitz_eta_h) * error_y

                    # -Lgh u <= dh/dt + Lfh + eta*h(x) - - a_y
                    Gs[key] = np.array([-Lgh[0], -Lgh[1], 0., 0., 0., 0.])
                    hs[key] = np.array([dh_dt + Lfh + eta*h_x - a_y])
                    #print("Target Back vehicle G(x)<=h", -Lgh[0], -Lgh[1], dh_dt, Lfh, eta*h_x)
            #else:
            #    Gs[key] = np.zeros(5)
            #    hs[key] = np.zeros(1)

        # multiple zeta for each individual barrier function

        # G_amin(x) <= H_amin // -acc <= -amin
        G_amin = np.array([[-1., 0., 0., 0., 0., 0.]])
        # G_amax(x) <= H_amax // acc <= amax
        G_amax = np.array([[1., 0., 0., 0., 0., 0.]])
        # G_betamin(x) <= H_betamin
        G_betamin = np.array([[0., -1., 0., 0., 0., 0.]])

        max_beta = math.atan2(0.5 * tan(math.radians(max_steering_angle)), 1)
        H_betamin = np.array([max_beta])
        H_beta_stable_min = np.array([math.radians(stable_steering_angle)])
        # G_betamax(x) <= H_betamax
        G_betamax = np.array([[0., 1., 0., 0., 0., 0.]])
        H_betamax = np.array([max_beta])
        H_beta_stable_max = np.array([math.radians(stable_steering_angle)])
        

        minmax = find_max_min_acc(self.poly_dict, v, self.ranges)

        '''
        objective:
        min 1/2(xT*P*x) + q(x)
        '''
        # keep_lane, CL, CR, 
        # BRK_0
        # ACC_1, BRK_1, ACC_2, BRK_2, ACC_3, BRK_3, ....
        allowed_behaviors = []
        cbf_solution = []
        for behavior in range(4+2*self.discretize): 
            # G_zeta(x) <= H_zeta
            G_zeta = np.zeros((8, 6))
            h_zeta = np.zeros(8)
            zeta_i = 0

            if behavior==0: # keep_lane_speed
                q = np.array([0., 0., 0., 0., 0., 0.])
                H_amin = np.array([1.])
                H_amax = np.array([1.])
                constraints_A = [G_amin, G_amax, G_betamin, G_betamax]
                constraints_b = [H_amin, H_amax, H_beta_stable_min, H_beta_stable_max]
                for key in ['cf', 'cb']:
                    if key in neighbors:
                        # fill in the terms for zeta(relaxation term)
                        Gs[key][zeta_i+2] = -1.
                        # set constraints for zeta_i
                        G_zeta[2*zeta_i, zeta_i+2] = -1.
                        G_zeta[2*zeta_i+1, zeta_i+2] = 1.
                        h_zeta[2*zeta_i+1] = max_zeta
                        q[zeta_i+2] = 10000.
                        zeta_i += 1
                        constraints_A.append(Gs[key])
                        constraints_b.append(hs[key])

            elif behavior==1: # CL
                beta_ref = math.atan2(0.3, 1)
                q = np.array([0., -beta_ref, 0., 0., 0., 0.]) # change_dist = 10m, lane_width~=3m
                H_amin = np.array([1.])
                H_amax = np.array([1.])
                constraints_A = [G_amin, G_amax, G_betamin, G_betamax]
                constraints_b = [H_amin, H_amax, H_betamin, H_betamax]
                for key in ['cf', 'cb', 'lf', 'lb']:
                    if key in neighbors:
                        # fill in the terms for zeta(relaxation term)
                        Gs[key][zeta_i+2] = -1.
                        # set constraints for zeta_i
                        G_zeta[2*zeta_i, zeta_i+2] = -1.
                        G_zeta[2*zeta_i+1, zeta_i+2] = 1.
                        h_zeta[2*zeta_i+1] = max_zeta
                        q[zeta_i+2] = 10000.
                        zeta_i += 1
                        constraints_A.append(Gs[key])
                        constraints_b.append(hs[key])

            elif behavior==2: # CR
                beta_ref = math.atan2(-0.3, 1)
                q = np.array([0., -beta_ref, 0., 0., 0., 0.]) # change_dist = 10m, lane_width~=3m
                H_amin = np.array([1.])
                H_amax = np.array([1.])
                constraints_A = [G_amin, G_amax, G_betamin, G_betamax]
                constraints_b = [H_amin, H_amax, H_betamin, H_betamax]
                for key in ['cf', 'cb', 'rf', 'rb']:
                    if key in neighbors:
                        # fill in the terms for zeta(relaxation term)
                        Gs[key][zeta_i+2] = -1.
                        # set constraints for zeta_i
                        G_zeta[2*zeta_i, zeta_i+2] = -1.
                        G_zeta[2*zeta_i+1, zeta_i+2] = 1.
                        h_zeta[2*zeta_i+1] = max_zeta
                        q[zeta_i+2] = 10000.
                        zeta_i += 1
                        constraints_A.append(Gs[key])
                        constraints_b.append(hs[key])

            else: # BRK_0, ACC_1, BRK_1, ACC_2, BRK_2, ...
                amin, amax = minmax[behavior-3]
                a_ref = 0.5*(amin + amax)
                H_amin = np.array([-amin])
                H_amax = np.array([amax])
                q = np.array([-a_ref, 0., 0., 0., 0., 0.])
                constraints_A = [G_amin, G_amax, G_betamin, G_betamax]
                constraints_b = [H_amin, H_amax, H_beta_stable_min, H_beta_stable_max]
                
                for key in ['cf', 'cb']:
                    if key in neighbors:
                        # fill in the terms for zeta(relaxation term)
                        Gs[key][zeta_i+2] = -1.
                        # set constraints for zeta_i
                        G_zeta[2*zeta_i, zeta_i+2] = -1.
                        G_zeta[2*zeta_i+1, zeta_i+2] = 1.
                        h_zeta[2*zeta_i+1] = max_zeta
                        q[zeta_i+2] = 10000.
                        zeta_i += 1
                        constraints_A.append(Gs[key])
                        constraints_b.append(hs[key])
            
            if use_zeta:
                if zeta_i>0:
                    G_zeta = G_zeta[:2*zeta_i, :]
                    h_zeta = h_zeta[:2*zeta_i]
                    constraints_A.append(G_zeta)
                    constraints_b.append(h_zeta)
            else:
                zeta_i = 0
            
            G = np.vstack(constraints_A)
            h = np.hstack(constraints_b)

            P = np.diagflat([1., 1., 0., 0., 0., 0.][:zeta_i+2])
            q = q[:zeta_i+2]
            G = G[:, :zeta_i+2]
            #print('behavior: ', behavior)
            #if behavior>=3:
            #    print("MINMAX range: ", minmax[behavior-3])

            P = matrix(P, tc='d')
            q = matrix(q, tc='d')
            G = matrix(G.astype(np.double), tc='d')
            h = matrix(np.squeeze(h).astype(np.double), tc='d')

            try:
                sol = solvers.qp(P, q, G, h)
                if sol['status'] == 'optimal':
                    x1, x2, *zetas = np.array(sol['x']).squeeze().tolist()
                    #print(zetas)
                    if behavior < 3:
                        target_throttle = search_throttle_value(self.poly_dict, x1, v)
                    else:
                        target_throttle = search_throttle_value(self.poly_dict, x1, v, self.ranges[behavior-3])
                    
                    steering = math.degrees(math.atan2(tan(x2)*2, 1))
                    #steering = math.degrees(x2)
                    if behavior in [1,2]:
                        steering = max(min(steering, max_steering_angle), -max_steering_angle)
                    else:
                        steering = max(min(steering, stable_steering_angle), -stable_steering_angle)
                    steering = steering / 70.
                    #print("Behavior: {}, Output: {:.2f}, {:.2f}; throttle: {:.2f}, steering: {:.2f}".format(behavior, x1, x2, target_throttle, steering))
                    #print(' '.join(["Zetas:"]+['{:.2f}']*len(zetas)).format(*zetas))
                    allowed_behaviors.append(1)
                    cbf_solution.append((target_throttle, steering))
                else:
                    #print("Behavior: {} unknown SOLUTION".format(behavior))
                    allowed_behaviors.append(0)
                    cbf_solution.append(None) 

            except:
                #print("Behavior: {} NO SOLUTION".format(behavior))
                allowed_behaviors.append(0)
                cbf_solution.append(None)

        return allowed_behaviors, cbf_solution

    def reset(self, scenario):
        """
        todo
        :returns:
        :rtype:
        """
        # first clean the vehicles and sensors in the environment
        for actor in [v[2] for v in self.vehicle_list] + self.colsensor_list:
            if actor.is_alive:
                actor.destroy()

        assert scenario in ['highway', 'crossing', 'both']
        if scenario == 'both':
            self.scenario = random.choice(['highway', 'crossing'])
        else:
            self.scenario = scenario

        # Reset various state information
        self.vehicle_list = []
        self.colsensor_list = []
        self.CAV_agents = []
        self.CAV_agents_dict = {}
        # CAV info
        self.state = {}
        self.speed_dict = {}
        # re_initiate ncav parameters
        self.ncav_vehicles = {}
        self.ncav_states = {}
        self.ncav_steer_controllers = {}
        self.hazard_ncav_ids = []
        self.spawn_dict = {}
        self.desti_dict = {}
        
        self.cbf_solutions_dict = None
        self.done_collision = False
        self.switch = False

        # !!! here it should change to scenario-based values
        self.lane_bounds = (0, 0) 
        if self.map_name=='Town05':
            self.lane_bounds = (1, 2) 
        elif self.map_name=='Town06':
            if self.scenario == 'highway':
                self.lane_bounds = (-3, -7)
                if self.args.middle3:
                    self.lane_bounds = (-4, -6)
            elif self.scenario == 'crossing':
                self.lane_bounds = (-4, -6)

        with self.collision_queue.mutex:
            self.collision_queue.queue.clear()
        self.accumulated_collision_penalty = 0.

        '''
            configuration:
            location
                base_location: [(x,y,z)] * n_cars
                random_term: [(True, False, False)] * n_cars
            velocity:
                base_throttle: [value] * n_cars
                random_term: [True/False] * n_cars
                target_speeds: [speed] * n_cars

        '''
        if self.map_name == 'Town06':
            #initially {lane_id: y}; -3: 136.4, -4: 139.8, -5: 143.3, -6: 146.6, -7: 150 
            #eventually {lane_id: y}; -3: 137.5, -4: 141.0, -5: 144.8, -6: 148.5, -7: 152.6 
            
            # we initialize storage for ncavs as they participate in highway scenario
            self.ncav_vehicles = {}
            self.ncav_steer_controllers = {}

            with open("./configs/map06_new.json", ) as f:
                config = json.load(f)
            if self.scenario == 'highway':
                self.num_UCVs = self.num_Vs - self.num_CAVs
                assert 1 <= self.num_CAVs <= 3 and 1 <= self.num_UCVs <= 6
                if self.test_only:
                    if self.args.scene_n == 0:
                        config_dict = config["Highway-0903-Test"]
                    elif self.args.scene_n == 1:
                        config_dict = config["Highway-0909-Test"]
                    else:
                        raise NotImplementedError
                else:
                    if self.args.scene_n == 0:
                        config_dict = config["Highway-0903"]
                    elif self.args.scene_n == 1:
                        config_dict = config["Highway-0909"]
                    else:
                        raise NotImplementedError

                self.spawn_locs, self.destinations, self.switch = spawn_from_config_new(config_dict, self.num_CAVs, self.num_UCVs, \
                                                    self.scenario, self.episode_length, self.cav_force_straight_step)
            
            elif self.scenario == 'crossing':
                self.num_UCVs = self.num_Vs - self.num_CAVs
                assert 1 <= self.num_CAVs <= 3 and 1 <= self.num_UCVs <= 2
                if self.test_only:
                    if self.args.scene_n == 0:
                        config_dict = config["Crossing"]
                    elif self.args.scene_n == 1:
                        config_dict = config["Crossing-1"]
                    elif self.args.scene_n == 2:
                        config_dict = config["Crossing-2"]
                    elif self.args.scene_n == 3:
                        config_dict = config["Crossing-3-Test"]
                    else:
                        raise NotImplementedError
                else:
                    if self.args.scene_n == 0:
                        config_dict = config["Crossing"]
                    elif self.args.scene_n == 1:
                        config_dict = config["Crossing-1"]
                    elif self.args.scene_n == 2:
                        config_dict = config["Crossing-2"]
                    elif self.args.scene_n == 3:
                        config_dict = config["Crossing-3"]
                    else:
                        raise NotImplementedError
                
                self.spawn_locs, self.destinations, self.switch = spawn_from_config_new(config_dict, self.num_CAVs, self.num_UCVs, \
                            self.scenario, self.episode_length, self.cav_force_straight_step)

            else:
                raise NotImplementedError
                
            # for loc in spawn_locs:
            #     transform = self.map.get_waypoint(carla.Location(x=loc[0], y=loc[1]), project_to_road=False).transform
    
            #     transform.location.z = transform.location.z + 0.5
            #     #transform = carla.Transform(carla.Location(*loc))
            #     spawn_points.append(transform)

        else:
            raise RuntimeError("Unrecognzied Map!")

        target_speed = config_dict["Target_speed"]
        self.num_HAZ = config_dict["NUM_HAZ"]
        self.default_steps_per_episode = config_dict["Default_Steps"]

        if self.scenario == 'crossing':
            self.hzv_target_velocity = np.random.rand() * 5 + 7.5
            self.ucv_target_velocity = np.random.rand() * 2 + 9
        elif self.scenario == 'highway':
            #self.hazard_throttle = np.random.rand()*0.2 + 0.8
            self.ucv_target_velocity = np.random.rand() * 2 + (config_dict["ucv_target_velocity"] - 1)
            self.ncav_brake = np.random.rand()*0.1 + 0.80
            if not self.normal_behave:
                self.hzv_hardbrake_start = random.randrange(self.ucv_force_straight_step+20, self.ucv_force_straight_step+40)
                print("NCAV Hazard Hard Brake start at step {:d}".format(self.hzv_hardbrake_start))

        if self.scenario == 'highway':
            cav_initial_velocity = carla.Vector3D(x=7.5, y=0, z=0)
            ucv_initial_velocity = carla.Vector3D(x=7.5, y=0, z=0)# 10 m/s in the x-direction
        else:
            ucv_initial_velocity = carla.Vector3D(x=0, y=0, z=0)
            cav_initial_velocity = carla.Vector3D(x=config_dict['CAV_inital_vel'], y=0, z=0)

        spawn_points = []
        # Spawn CAVs at each given spawn point
        for i in range(self.num_Vs):
            loc = self.spawn_locs[i]
            transform = self.map.get_waypoint(carla.Location(x=loc[0], y=loc[1]), project_to_road=False).transform
            transform.location.z = transform.location.z + 0.5
            spawn_points.append(transform)

            # Connected autonomous vehicles
            if i < self.num_CAVs:
                vehicle = self.world.spawn_actor(self.model3_cav, transform)
                vehicle.set_target_velocity(cav_initial_velocity)
                self.vehicle_list.append((vehicle.id, V_Type.CAV, vehicle))
                self.spawn_dict[vehicle.id] = loc
                self.desti_dict[vehicle.id] = self.destinations[i]

                # Attach collision sensor
                colsensor = self.blueprint_library.find("sensor.other.collision")
                colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=vehicle)
                self.colsensor_list.append(colsensor)
                colsensor.listen(lambda event: collision_handler(event, self.collision_queue))
                print("CAV {:d} ; Initial Spawn ({:.2f}, {:.2f}); Destination ({:.2f}, {:.2f})".\
                        format(vehicle.id, transform.location.x, transform.location.y, \
                        self.destinations[i][0], self.destinations[i][1]))
            
            else: # environment vehicle
                vehicle = self.world.spawn_actor(self.model3_ncav, transform)
                vehicle.set_target_velocity(ucv_initial_velocity)
                # NCAV
                self.ncav_vehicles[vehicle.id] = vehicle
                self.ncav_steer_controllers[vehicle.id] = PIDLateralController(vehicle=vehicle, \
                                K_P=0.5, K_D=0.0, K_I=0.1, dt=0.05)
                self.vehicle_list.append((vehicle.id, V_Type.NCAV, vehicle))

                if i >= self.num_Vs - self.num_HAZ:
                    self.hazard_ncav_ids.append(vehicle.id)
                    
                if self.scenario == 'crossing':
                    if i >= self.num_Vs - self.num_HAZ:
                        print("NCAV {:d} target speed: {:.2f}; Initial Spawn ({:.2f}, {:.2f})".\
                            format(vehicle.id, self.hzv_target_velocity, transform.location.x, transform.location.y))
                    else:
                        print("NCAV {:d} target speed: {:.2f}; Initial Spawn ({:.2f}, {:.2f})".\
                            format(vehicle.id, self.ucv_target_velocity, transform.location.x, transform.location.y))
                else:
                    print("NCAV {:d} target speed: {:.2f} m/s; Initial Spawn ({:.2f}, {:.2f})".\
                        format(vehicle.id, self.ucv_target_velocity, transform.location.x, transform.location.y))

        markers = [str(i) for i in range(1,5)]

        for i, (vid, _, vehicle) in enumerate(self.vehicle_list[:self.num_CAVs]):
            new_agent = CAVBehaviorPlanner(
                dt           = self.timestep,
                target_speed = target_speed,
                vehicle      = vehicle,
                map_name    = self.map_name,
                param_dict = self.param_dict,
                CAV_agents_dict = self.CAV_agents_dict,
                num_CAVs = self.num_CAVs,
                marker = markers[i%len(markers)],
                discretize=self.discretize, 
                lane_bounds=self.lane_bounds,
                remove_CBFA=self.remove_CBFA,
                spawn_transform=spawn_points[i]
            )
            self.CAV_agents.append(new_agent)
            self.CAV_agents_dict[vid] = new_agent

        self.spectator = self.world.get_spectator()
        self.world.tick()

        # crossing
        spec_transform = carla.Location(x=-40, y=-100, z=25)
        spec_rotation = carla.Rotation(pitch=-50, yaw=135)
        
        # original method: self.vehicle_list[0].get_transform()
        # merging
        if self.spec_view == 'birdeye':
            if self.map_name == 'Town06' and self.scenario == 'crossing':
                # fixed above
                xs, ys = [sp[0] for sp in self.spawn_locs], [sp[1] for sp in self.spawn_locs]
                height_addon = (stdev(xs) + stdev(ys))/ 2.
                spec_transform = carla.Location(x=0, y=45, z=50 + height_addon)
                spec_rotation = carla.Rotation(pitch=-90, yaw=90.0)
            else:
                spec_transform = carla.Location(x=self.spawn_locs[0][0]-5, y=self.spawn_locs[0][1]+5, z=25)
                spec_rotation = carla.Rotation(pitch=-90, yaw=90.0)
        elif self.spec_view == 'front_up':
            spec_transform = carla.Location(x=self.spawn_locs[0][0]+26., y=self.spawn_locs[0][1], z=15)
            spec_rotation = carla.Rotation(pitch=-50, yaw=180, roll=0)
        elif self.spec_view == 'side_up':
            if self.map_name == 'Town06' and self.scenario == 'crossing':
                #spec_transform = carla.Location(x=mean(loc_xs[:self.num_CAVs]), y=min(35, stdev(loc_xs)), z=min(50, stdev(loc_xs)))
                spec_transform = carla.Location(x=self.spawn_locs[0][0]+5, y=self.spawn_locs[0][1]+35, z=50)
                spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
            elif self.scenario == 'highway':
                spec_transform = carla.Location(x=self.spawn_locs[0][0]-10, y=self.spawn_locs[0][1]+30, z=40)
                spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
        else:
            if self.map_name == 'Town06' and self.scenario == 'crossing':
                #spec_transform = carla.Location(x=30, y=60, z=30)
                #spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
                spec_transform = carla.Location(x=0, y=45, z=50)
                spec_rotation = carla.Rotation(pitch=-90, yaw=90.0)

        self.spectator.set_transform(
            carla.Transform(spec_transform, spec_rotation)
        )

        # when attack method is tgt_t (time_duration_attack) or tgt_v (target vehicle attack)
        # we need to define several parameters 
        if self.e_type:
            all_vids = [v[0] for v in self.vehicle_list]

            # current_step, target_start_end, error_pos
            if self.e_type == 'tgt_t':
                start = random.choice(range(50, 80))
                self.target_start_end = (start, start + 100)
                error_pos = {}
                for vid in all_vids:
                    error_pos[vid] = [random.choice([-1,1]) for _ in range(4)]
                self.error_pos = error_pos
                print("ATTACK: Time_duration: ({}, {})".format(start, start+100), \
                    *["{:d}: ({:d},{:d},{:d},{:d}) |".format(k,*v) for k, v in self.error_pos.items()])

            # error_pos
            elif self.e_type == 'tgt_v':
                num_perturbed_v = random.choice(range(self.num_Vs//2, self.num_Vs//2+2))
                targeted_vid = random.sample(all_vids, num_perturbed_v)
                error_pos = {}
                for vid in targeted_vid:
                    error_pos[vid] = [random.choice([-1,1]) for _ in range(4)]
                self.error_pos = error_pos
                print("ATTACK: Target Vehicles: ", \
                    *["{:d}: ({:d},{:d},{:d},{:d}) |".format(k,*v) for k, v in self.error_pos.items()])


        all_car_info_dict = self._update_each_vehicle_true_state()

        # from self.state to observation
        obs_dict = observation_from_state(self.state, self.CAV_agents_dict.keys(), self.max_num_car, self.brief_state)

        # here should add a safety shield
        self.cbf_safe_action_dict = generate_safe_action(self.CAV_agents_dict.keys(), \
                self.discretize, mute_lane_change=self.mute_lane_change)

        #assert set(obs_dict.keys())==set(self.allowed_behaviors_dict.keys())
        #assert set(obs_dict.keys())==set(self.cbf_safe_action_dict.keys())

        #safe_actions = combine_safe_actions(self.allowed_behaviors_dict, self.cbf_safe_action_dict)
        safe_actions = self.cbf_safe_action_dict

        return obs_dict, safe_actions, all_car_info_dict

    def step_cav_only(self, action_dict=None):
        try:
            batch = []
            for vid, agent in self.CAV_agents_dict.items():    
                if not action_dict or vid not in action_dict:
                    action = [0., 0., 0.]
                    print("Using hardcode action")
                else:
                    action = action_dict[vid]

                cav_control = carla.VehicleControl(throttle=action[0], \
                                                   brake=action[1], \
                                                   steer=action[2])
                batch.append(carla.command.ApplyVehicleControl(agent.vehicle, cav_control))

        # ----- This part could be simplified ------ #            
        except (AttributeError, RuntimeError):
            traceback.print_exc()
            print("ENV: Early termination of episode during the run_step function.")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
            force_continue = True
        except IndexError:
            traceback.print_exc()
            print("ENV: Early termination of episode during the run_step function: IndexError.")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
            force_continue = True

        #self.client.apply_batch(batch)
        responses = self.client.apply_batch_sync(batch)
        self.world.tick()

        for response in responses:
            if response.has_error():
                print("Env error: ", response.actor_id, response.error)
        
        # 4. Also record state information and neighbor information
        all_car_info_dict = self._update_each_vehicle_true_state()

        if self.spec_view != 'none':
            loc_xs = [state[0] for state in self.state.values()] + [state[0] for state in self.ncav_states.values()]
            loc_ys = [state[1] for state in self.state.values()] + [state[1] for state in self.ncav_states.values()]
            #print("Mean and STD of Xs:", mean(loc_xs), stdev(loc_xs))
            #haz_y, haz_z = self.ncav_states[self.hazard_ncav_ids[0]][1:3]
            if self.spec_view=='side_up':
                if self.map_name == 'Town06' and self.scenario == 'crossing':
                    spec_transform = carla.Location(x=mean(loc_xs[:self.num_CAVs]), y=min(35, stdev(loc_xs)), z=min(50, stdev(loc_xs)))
                    spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
                elif self.scenario == 'highway':
                    #spec_transform = spec_rotation = None
                    #spec_transform = carla.Location(x=50., y=25, z=30)
                    #spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
                    spec_transform = carla.Location(x=mean(loc_xs[:self.num_CAVs])-10, y=mean(loc_ys[:self.num_CAVs])+30., z=40.)
                    spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
            elif self.spec_view == 'front_up':
                spec_transform = carla.Location(x=mean(loc_xs[:self.num_CAVs])+26, y=min(30, stdev(loc_xs)), z=min(30, stdev(loc_xs)))
                spec_rotation = carla.Rotation(pitch=-50, yaw=180, roll=0)
            elif self.spec_view == 'birdeye':
                if self.map_name == 'Town06' and self.scenario == 'crossing':
                    # move camera up and down
                    height_addon = (stdev(loc_xs) + stdev(loc_ys))/ 2.
                    spec_transform = carla.Location(x=0, y=45, z=50 + height_addon)
                    spec_rotation = carla.Rotation(pitch=-90, yaw=90.0)
                else:
                    spec_transform = carla.Location(x=mean(loc_xs[:self.num_CAVs]), y=min(30, stdev(loc_xs))+25, z=min(30, stdev(loc_xs))+20)
                    spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
            
            if spec_transform and spec_rotation:
                self.spectator.set_transform(
                    carla.Transform(spec_transform, spec_rotation)
                )

        # from self.state to observation
        obs_dict = observation_from_state(self.state, self.CAV_agents_dict.keys(), self.max_num_car, self.brief_state)

        # currently set all action as safe action
        # here should add a safety shield
        self.cbf_safe_action_dict = generate_safe_action(self.CAV_agents_dict.keys(), \
                self.discretize, mute_lane_change=self.mute_lane_change)

        return obs_dict, all_car_info_dict, self.cbf_safe_action_dict

    def step(self, action, step_n=-1):
        """
        todo
        
        :param action: 
        :type action:  
        :returns:
        :rtype:
        """
        #print(action)
        # 1. Record the action with the BPs
        force_continue = False
        for agent in self.CAV_agents:
            selected_action = action[agent.vehicle.id]
            agent.set_behavior(Behavior(selected_action), set_step=step_n, debug=self.debug)
            if self.cbf_solutions_dict and selected_action != -1:
                cbf_solution = self.cbf_solutions_dict[agent.vehicle.id][selected_action]
                #print("CAV: ", agent.vehicle.id, cbf_solution)
                agent.set_cbf_solution(cbf_solution)

        # 2. Create and apply a batch of Carla VehicleControls.
        #    (Must be applied as batch-- see Carla documentation.)
        # Runs one step of behavior planning, path planning, and control
        try:

            batch = []
            true_actions = {}
            true_control_dict = {}
            #print("!!!!!!!!!!!!!!!!!", self.cbf_safe_action_dict.keys())
            for agent in self.CAV_agents:
                true_control, true_action = agent.run_step(self.scenario, debug=self.debug, step_n=step_n, \
                        force_straight_step=self.cav_force_straight_step, mute_lane_change=self.mute_lane_change, \
                        previous_speed=self.speed_dict[agent.vehicle.id], cbf_safe_action=self.cbf_safe_action_dict[agent.vehicle.id])
                true_actions[agent.vehicle.id] = true_action
                true_control_dict[agent.vehicle.id] = true_control
                batch.append(carla.command.ApplyVehicleControl(
                    agent.vehicle, true_control))

# ----- This part could be simplified ------ #
# ----- (the logics for HAZV and NCAV could be deduced from all_car_info_dict) ----- #
            if self.map_name=='Town06':
                for vid, ncav in self.ncav_vehicles.items():
                # set ncav with constant throttle value
                    ncav_steer = 0
                    if self.args.sabbir_control:
                        pass
                    elif step_n < self.ucv_force_straight_step:
                        force_throttle = 0.85
                        if self.scenario == 'crossing':
                            force_throttle = 0.85
                        ncav_control = carla.VehicleControl(throttle=force_throttle, steer=ncav_steer)
                        batch.append(carla.command.ApplyVehicleControl(ncav, ncav_control))
                        continue

                    if self.scenario == 'highway':
                        if self.normal_behave or (vid not in self.hazard_ncav_ids) or \
                            (vid in self.hazard_ncav_ids and step_n < self.hzv_hardbrake_start):
                            # target a speed
                            ncav_speed = self.ncav_states[vid][3:5]
                            ncav_speed = math.sqrt(ncav_speed[0]**2+ncav_speed[1]**2)
                            ncav_acc_revise = np.random.rand()*0.04 - 0.02
                            if ncav_speed >= self.ucv_target_velocity:
                                ncav_target_throttle = search_throttle_value(self.poly_dict, 0., self.ucv_target_velocity - 0.1)
                            else:
                                ncav_target_throttle = search_throttle_value(self.poly_dict, 0., self.ucv_target_velocity + 0.1)
                            ncav_control = carla.VehicleControl(throttle=ncav_target_throttle + ncav_acc_revise, steer=ncav_steer)
                        else:
                            if self.hzv_hardbrake_start <= step_n < self.hzv_hardbrake_start + 12:
                                haz_ttl_revise = np.random.rand()*0.1 - 0.5
                                ncav_control = carla.VehicleControl(throttle=0., brake=self.ncav_brake+haz_ttl_revise, steer=ncav_steer)
                            else:
                                # base brake 0.6 - 0.9
                                #base_random_brake = np.random.rand()*0.2 + 0.7
                                # 0.5 ~ 1.0
                                #haz_brk_revise = np.random.rand()*0.2 - 0.1
                                #ncav_control = carla.VehicleControl(throttle=0., brake=base_random_brake+haz_brk_revise, steer=ncav_steer)
                                ncav_speed = self.ncav_states[vid][3:5]
                                ncav_speed = math.sqrt(ncav_speed[0]**2+ncav_speed[1]**2)
                                ncav_acc_revise = np.random.rand()*0.04 - 0.02
                                rand_target_speed = np.random.rand()*3 + 1.
                                if ncav_speed >= rand_target_speed:
                                    ncav_target_throttle = search_throttle_value(self.poly_dict, 0., rand_target_speed - 0.1)
                                else:
                                    ncav_target_throttle = search_throttle_value(self.poly_dict, 0., rand_target_speed + 0.1)
                                ncav_control = carla.VehicleControl(throttle=ncav_target_throttle + ncav_acc_revise, steer=ncav_steer)
                            # comment on crossing
                            '''
                                CROSSING:
                                0 ~ FSS: acc 0.82
                                normal_behave
                                    FSS ~ final
                                        - UCV brake 1.
                                train:  HAZV = 0
                                    FSS ~ final 
                                        - target speed (9 - 11)
                                test:   HAZV = 2
                                    FSS ~ final
                                        - target speed (7.5, 12.5)
                            '''
                    elif self.scenario == 'crossing':
                        if self.normal_behave:
                            random_brake = np.random.rand()*0.1 + 0.9
                            ncav_control = carla.VehicleControl(throttle=0., brake=random_brake, steer=ncav_steer)
                        else:
                            # at end stage, hard brake all UCV
                            if step_n>= 206:
                                ncav_control = carla.VehicleControl(throttle=0., brake=0.95, steer=ncav_steer)
                            else:
                                throttle_revise = np.random.rand()*0.04 - 0.02
                                ncav_speed = self.ncav_states[vid][3:5]
                                ncav_speed = math.sqrt(ncav_speed[0]**2+ncav_speed[1]**2)
                                if vid in self.hazard_ncav_ids:
                                    ncav_target_speed = self.hzv_target_velocity
                                else:
                                    ncav_target_speed = self.ucv_target_velocity
                                
                                if ncav_speed >= ncav_target_speed:
                                    ncav_target_throttle = search_throttle_value(self.poly_dict, 0., ncav_target_speed - 0.2)
                                else:
                                    ncav_target_throttle = search_throttle_value(self.poly_dict, 0., ncav_target_speed + 0.2)
                                
                                ncav_control = carla.VehicleControl(throttle=ncav_target_throttle + throttle_revise, steer=ncav_steer)
                    else:
                        raise NotImplementedError

                    #ncav_loc  = ncav.get_location()
                    #ncav_waypoint = self.map.get_waypoint(ncav_loc)
                    #ncav_next_waypoint = ncav_waypoint.next(10.)[0]
                    #ncav_steer = self.ncav_steer_controllers[i].run_step(ncav_next_waypoint)
                    
                    batch.append(carla.command.ApplyVehicleControl(ncav, ncav_control))
            
# ----- This part could be simplified ------ #            
        except (AttributeError, RuntimeError):
            traceback.print_exc()
            print("ENV: Early termination of episode during the run_step function.")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
            force_continue = True
        except IndexError:
            traceback.print_exc()
            print("ENV: Early termination of episode during the run_step function: IndexError.")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
            force_continue = True
        except:
            traceback.print_exc()
            print("ENV: Early termination of unexpected error.")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
            force_continue = True

        #self.client.apply_batch(batch)
        responses = self.client.apply_batch_sync(batch)
        self.world.tick()

        for response in responses:
            if response.has_error():
                print("Env error: ", response.actor_id, response.error)
        
        # 4. Also record state information and neighbor information
        all_car_info_dict = self._update_each_vehicle_true_state()

        # generate error if experiment requires
        # generate_error_new(cav_ids, all_ids, e_type, error_bound, scenario, current_step=None, target_start_end=None, error_pos=None)
        error = None
        old_error = None
        if self.e_type:
            if self.e_type == 'tgt_t':
                error_dict, old_error_dict = generate_error_new(cav_ids=self.CAV_agents_dict.keys(), all_ids=[v[0] for v in self.vehicle_list], \
                                        e_type=self.e_type, error_bound=[3,1,3,3], scenario=self.scenario, \
                                        current_step=step_n, target_start_end=self.target_start_end,\
                                        error_pos=self.error_pos)
            elif self.e_type == 'tgt_v':
                error_dict, old_error_dict = generate_error_new(cav_ids=self.CAV_agents_dict.keys(), all_ids=[v[0] for v in self.vehicle_list], \
                                        e_type=self.e_type, error_bound=[3,1,3,3], scenario=self.scenario, \
                                        error_pos=self.error_pos)
            else:
                error_dict, old_error_dict = generate_error_new(cav_ids=self.CAV_agents_dict.keys(), all_ids=[v[0] for v in self.vehicle_list], \
                                        e_type=self.e_type, error_bound=[3,1,3,3], scenario=self.scenario)
            error = error_dict
            old_error = old_error_dict
            print(error_dict)
            print(old_error_dict)

        if self.spec_view != 'none':
            loc_xs = [state[0] for state in self.state.values()] + [state[0] for state in self.ncav_states.values()]
            loc_ys = [state[1] for state in self.state.values()] + [state[1] for state in self.ncav_states.values()]
            #print("Mean and STD of Xs:", mean(loc_xs), stdev(loc_xs))
            #haz_y, haz_z = self.ncav_states[self.hazard_ncav_ids[0]][1:3]
            if self.spec_view=='side_up':
                if self.map_name == 'Town06' and self.scenario == 'crossing':
                    spec_transform = carla.Location(x=mean(loc_xs[:self.num_CAVs]), y=min(35, stdev(loc_xs)), z=min(50, stdev(loc_xs)))
                    spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
                elif self.scenario == 'highway':
                    #spec_transform = spec_rotation = None
                    #spec_transform = carla.Location(x=50., y=25, z=30)
                    #spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
                    spec_transform = carla.Location(x=mean(loc_xs[:self.num_CAVs])-10, y=mean(loc_ys[:self.num_CAVs])+30., z=40.)
                    spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
            elif self.spec_view == 'front_up':
                spec_transform = carla.Location(x=mean(loc_xs[:self.num_CAVs])+26, y=min(30, stdev(loc_xs)), z=min(30, stdev(loc_xs)))
                spec_rotation = carla.Rotation(pitch=-50, yaw=180, roll=0)
            elif self.spec_view == 'birdeye':
                if self.map_name == 'Town06' and self.scenario == 'crossing':
                    # move camera up and down
                    height_addon = (stdev(loc_xs) + stdev(loc_ys))/ 2.
                    spec_transform = carla.Location(x=0, y=45, z=50 + height_addon)
                    spec_rotation = carla.Rotation(pitch=-90, yaw=90.0)
                else:
                    spec_transform = carla.Location(x=mean(loc_xs[:self.num_CAVs]), y=min(30, stdev(loc_xs))+25, z=min(30, stdev(loc_xs))+20)
                    spec_rotation = carla.Rotation(pitch=-40, yaw=-90., roll=0.)
            
            if spec_transform and spec_rotation:
                self.spectator.set_transform(
                    carla.Transform(spec_transform, spec_rotation)
                )

        # 3. Get reward stats.
        rewards, reward_items, mean_rwd_items, flow_reward, dest_reward, cav_collision_dict \
            = self._get_reward(chosen_actions=action, true_actions=true_actions, avg_p=self.args.avg_coef, \
                col_r=True, include_env_collision=True, interpolate=((step_n+1) / self.default_steps_per_episode))
        done = (step_n >= self.episode_length)

        # from self.state to observation
        obs_dict = observation_from_state(self.state, self.CAV_agents_dict.keys(), self.max_num_car, self.brief_state, error=error, debug=False)
        
        self.cbf_safe_action_dict = {}
        self.cbf_solutions_dict = {}
        #risk_rs = {}
        dt = 1
        # close_threshold = 12, however, we include range=30 vehicles into consideration
        for agent in self.CAV_agents:
            error_input = None
            if self.e_type:
                error_input = (old_error['type'], old_error[agent.vehicle.id])
            cbf_state, neighbors, others = agent.detect_env(scenario=self.scenario, crossV_ET_range=(0.5, 1.5), debug=False,
                                                previous_action_dict=true_actions, error=error_input)

            # consider risk case by case
            #risk_r = 3. # 30 is the threshold we consider; 10 is an alert distance ; 3= (min(30,30))/10

            for k, nbr in neighbors.items():
                #print(agent.vehicle.id, k, nbr)
                _, x_diff, target_v, target_acc = nbr
                #x_diff_dt = x_diff + 0.5 * target_acc*pow(dt,2) + (target_v - cbf_state[3]*cos(cbf_state[2]))*dt
                #risk_r = min(risk_r, min(abs(x_diff), abs(x_diff_dt))/10.) 
                # !!!!!!! change this , add predicted distance in the next timestep
                
            #risk_rs[agent.vehicle.id] = risk_r
            # if (not self.disable_CBF) and step_n>=self.cav_force_straight_step:
            if not self.disable_CBF:
                cbf_start_time = time.time()
                allowed_behaviors, cbf_solution = self.robust_cbf(agent.vehicle.id, cbf_state, neighbors, \
                        other_vehicles=others, error_y=self.cbf_robustness)
                if self.debug:
                    print("CBF {} takes: {:.4f} second.".format(agent.vehicle.id, time.time()-cbf_start_time))
                self.cbf_safe_action_dict[agent.vehicle.id] = allowed_behaviors
                self.cbf_solutions_dict[agent.vehicle.id] = cbf_solution
        
        # if step_n < self.force_straight_step or self.disable_CBF:
        if self.disable_CBF:
            self.cbf_safe_action_dict = self._null_cbf_safe_action_dict()
            self.cbf_solutions_dict=None

        base_safe_actions = generate_safe_action(self.CAV_agents_dict.keys(), \
                self.discretize, mute_lane_change=self.mute_lane_change)
        assert set(obs_dict.keys())==set(base_safe_actions.keys())
        assert set(obs_dict.keys())==set(self.cbf_safe_action_dict.keys())

        safe_actions = combine_safe_actions(base_safe_actions, self.cbf_safe_action_dict)

        # if step_n >= self.force_straight_step:
        print("ANY Collision? "+("{:s}"*len(cav_collision_dict)).format(*[" | VID: {:d} - {:s}".\
                format(k,str(v)) for k,v in cav_collision_dict.items()]))
        for vid, safe_action in safe_actions.items():
            safe_action_idx = np.nonzero(safe_action)[0].tolist()
            safe_action = [str(Behavior(a).name) for a in safe_action_idx]
            print("Car {:d} safe actions:{}".format(vid, str(safe_action)))

        return obs_dict, rewards, done, safe_actions, all_car_info_dict, \
                true_actions, reward_items, mean_rwd_items, flow_reward, dest_reward, force_continue, error
        #return true_actions, done, reward, risk_rs, (mean_flow, mean_cols, mean_dest), \
        #        self.allowed_behaviors_dict, self.cbf_safe_action_dict

    def close(self):
        """
        todo
        """
        actor_list = [v[2] for v in self.vehicle_list] + self.colsensor_list
        self.client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        #for actor in actor_list:
        #    destroyed_sucessfully = actor.destroy()
        #    if not destroyed_sucessfully:
        #        print("destroyed_sucessfully")
