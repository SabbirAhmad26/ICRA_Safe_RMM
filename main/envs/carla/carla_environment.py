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
# from cvxopt import matrix, solvers, spdiag
from .controller import PIDLateralController
from .tools.misc import draw_points, angle_between_vec
from .env_utils import (
    #find_max_min_acc, 
    #find_acc, 
    search_throttle_value,
    # compute_safe_dist,
    # compute_trajectory,
    generate_error_new,
    observation_from_state,
    #combine_safe_actions,
    #spawn_from_config,
    spawn_from_config_new,
    generate_safe_action,
    compute_collision_penalty,
    reference_speed,
    combine_obs_dicts
)
#from .tools.gen_scenegraph import SceneGraph
import carla
from .enums import V_Type
from .enums import Behavior

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
    
    def __init__(self, args, num_CAVs = 1):
        # Connect to the Carla world and set up settings
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

        self.discretize = args.discretize
        assert self.discretize in [0,3,4,5]
        
        self.test_only = args.test_only
        self.debug = args.debug
        self.e_type = args.e_type

        # env params
        self.num_envs = 1
        self.episode_length = args.episode_length
        self.flow_reward_coef = self.args.flow_reward_coef
        self.cav_force_straight_step = self.args.cav_fs_step
        self.ucv_force_straight_step = self.args.ucv_fs_step
        self.temporal_length = self.args.temporal_length

        # RL params
        self._policies = None
        self.brief_state = args.brief_state
        if self.brief_state:
            self.base_state_length = 8
            self.max_num_car = 6
        else:
            self.base_state_length = 16
            self.max_num_car = 8
        self.observation_space = dict.fromkeys(range(num_CAVs), [self.base_state_length*self.max_num_car*self.temporal_length])
        self.share_observation_space = dict.fromkeys(range(num_CAVs), [self.base_state_length*self.max_num_car*num_CAVs*self.temporal_length])
        self.action_space = dict.fromkeys(range(num_CAVs), spaces.Discrete(3 + self.discretize))

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
        self.num_Vs = args.cars
        self.num_CAVs = num_CAVs
        #self.seq_length = seq_length
        self.vehicle_list    = []
        self.colsensor_list  = []  # todo: unused
        self.collision_queue = queue.Queue()
        self.CAV_dict = {}
        self.obs_list = []

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
        # destination info
        self.destination = []

        # collision flag
        self.done_collision = False

        with open('./new_polys.pkl', 'rb') as fp:
            poly_dict = pickle.load(fp)
            fp.close()
        self.poly_dict = poly_dict

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
            car_info_dict = {'id': vid, 'x': car_loc.x, 'y': car_loc.y, 'phi': np.deg2rad(car_rot.yaw),
                             'vel': kmh_velocity, 'road_id': wp.road_id,
                             'lane_id': wp.lane_id, 'jct_id': jct_id, #'phi': math.radians(car_rot.yaw), 
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

    def get_agents_dict(self):
        agents_dict = {}
        for vid, cav in self.CAV_dict.items():
            cav_lane_id = self.map.get_waypoint(cav.get_location()).lane_id
            # "command" will be updated by the path planner
            temp_cav = {'position': dict(x = 0, y = 0), 'theta': None, 'speed': None, "command": "lane_keeping", 'initial_position': dict(x = 0, y =0),
                             "previous_command": "lane_keeping", "reference_lane": None, "MPs" : [], "recommended_command": "lane_keeping",
                             "reference_lane_index": cav_lane_id, "previous_completion": 0, 'ellipsoid' :dict(a = 0.01, b =0.01),
                             "current_lane": None, "completion":0, "length_path":50, "lc_starting_time": -1, "lc_starting_time_2":-1,
                             "current_lane_index": cav_lane_id, 'timeout':0, "lane_keeping_count" : 20, 'acceleration' : 3.92,
                             'arrival_index': None, "ref_centerlane": None, "throttle": None, "steer": None, "conflicting_cars":dict(ip = [], ic_1=[], ic_2=[]),
                             "brake": None, "ref_theta": None, "ref_speed": None}

            agents_dict[vid] = temp_cav
        return agents_dict

    def _get_reward(self, avg_p=1.0, col_r=True, include_env_collision=True, interpolate=1.0): # total_col_intensity=0, locs=None)
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
        cav_ids = self.CAV_dict.keys()
        vids = [v[0] for v in self.vehicle_list]
        
        col_penalties, cav_collision_dict, accumulated_collision_penalty, has_collision = \
            compute_collision_penalty(collisions, cav_ids, vids, col_r, include_env_collision)

        self.done_collision = self.done_collision or has_collision
        
        # locs = []
        vels_rwds = {}
        dest_rwds = {}

        # spawn_locs are sorted in the increasing order of vehicle_ids
        assert len(self.destinations) == len(self.CAV_dict) <= len(self.spawn_locs)
        
        for vid in self.CAV_dict:
            vels_rwds[vid] = (0.2 * self.flow_reward_coef) * (self.speed_dict[vid] - reference_speed) / 3.6
            loc = self.state[vid][0:3]
            spawn, desti = self.spawn_dict[vid], self.desti_dict[vid]
            dest_rwds[vid] = (loc[0] - spawn[0]) / ((desti[0] - spawn[0]) * interpolate)

        # mean_flow_rwd = np.mean(list(vels_rwds.values()))
        # mean_dest_rwd = np.mean(list(dest_rwds.values()))
        # mean_cols_rwd = np.mean(list(col_penalties.values()))

        # this will be modified
        sact_rwds = dict.fromkeys(cav_ids, 0.)

        # rewards = avg_p * (mean_flow_rwd + mean_dest_rwd + mean_cols_rwd + 0) + \
        #          (1 - avg_p) * (flow_reward + loc_penalties + col_penalties + sact_reward)
        # return rewards, mean_flow_rwd, mean_cols_rwd, mean_dest_rwd, 0, cav_collision_dict

        return vels_rwds, dest_rwds, col_penalties, sact_rwds, cav_collision_dict

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
        # CAV info
        self.CAV_dict = {}
        self.state = {}
        self.speed_dict = {}
        # re_initiate ncav parameters
        self.ncav_vehicles = {}
        self.ncav_states = {}
        self.ncav_steer_controllers = {}
        self.hazard_ncav_ids = []
        self.spawn_dict = {}
        self.desti_dict = {}
        self.obs_list = []
        
        self.done_collision = False
        self.switch = False

        # !!! here it should change to scenario-based values
        self.lane_bounds = (0, 0) 
        if self.map_name=='Town05':
            self.lane_bounds = (1, 2) 
        elif self.map_name=='Town06':
            if self.scenario == 'highway':
                self.lane_bounds = (-3, -6)
            elif self.scenario == 'crossing':
                self.lane_bounds = (-4, -6)

        with self.collision_queue.mutex:
            self.collision_queue.queue.clear()

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
                    elif self.args.scene_n == 2:
                        config_dict = config["Highway-baseline-4"]    
                    else:
                        raise NotImplementedError
                else:
                    if self.args.scene_n == 0:
                        config_dict = config["Highway-0903"]
                    elif self.args.scene_n == 1:
                        config_dict = config["Highway-0909"]
                    elif self.args.scene_n == 2:
                        config_dict = config["Highway-baseline-4"]
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

        else:
            raise RuntimeError("Unrecognzied Map!")

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
        
        # Spawn CAVs at each given spawn point
        for i in range(self.num_Vs):
            loc = self.spawn_locs[i]
            transform = self.map.get_waypoint(carla.Location(x=loc[0], y=loc[1]), project_to_road=False).transform
            transform.location.z = transform.location.z + 0.2

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

                # we set by default that the last n (n = num_haz) vehicles are hazard vehicle
                if i >= self.num_Vs - self.num_HAZ:
                    self.hazard_ncav_ids.append(vehicle.id)

                if self.scenario == 'crossing':
                    if i >= self.num_Vs - self.num_HAZ:
                        print("NCAV {:d} target speed: {:.2f} m/s; Initial Spawn ({:.2f}, {:.2f})".\
                            format(vehicle.id, self.hzv_target_velocity, transform.location.x, transform.location.y))
                    else:
                        print("NCAV {:d} target speed: {:.2f} m/s; Initial Spawn ({:.2f}, {:.2f})".\
                            format(vehicle.id, self.ucv_target_velocity, transform.location.x, transform.location.y))
                else:
                    print("NCAV {:d} target speed: {:.2f} m/s; Initial Spawn ({:.2f}, {:.2f})".\
                        format(vehicle.id, self.ucv_target_velocity, transform.location.x, transform.location.y))

        markers = [str(i) for i in range(1,5)]

        for i, (vid, _, vehicle) in enumerate(self.vehicle_list[:self.num_CAVs]):
            self.CAV_dict[vid] = vehicle

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
        self.previous_lane_ids = {vid: all_car_info_dict[vid]['lane_id'] for vid in self.CAV_dict}

        # from self.state to observation
        obs_dict = observation_from_state(self.state, \
            self.CAV_dict.keys(), self.max_num_car, self.brief_state)

        if self.temporal_length > 1:
            self.obs_list = [deepcopy(obs_dict) for _ in range(self.temporal_length)]
            obs_dict = combine_obs_dicts(self.obs_list)

        safe_action_dict = generate_safe_action(self.CAV_dict.keys(), simple_n=3+self.discretize)

        return obs_dict, all_car_info_dict, safe_action_dict

    def step_cav_only(self, action_dict=None):
        try:
            batch = []
            for vid, cav in self.CAV_dict.items():    
                if not action_dict or vid not in action_dict:
                    action = [0., 0., 0.]
                    print("Using hardcode action")
                else:
                    action = action_dict[vid]

                cav_control = carla.VehicleControl(throttle=action[0], \
                                                   brake=action[1], \
                                                   steer=action[2])
                batch.append(carla.command.ApplyVehicleControl(cav, cav_control))

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
        obs_dict = observation_from_state(self.state, \
            self.CAV_dict.keys(), self.max_num_car, self.brief_state)

        if self.temporal_length > 1:
            self.obs_list.pop(0)
            self.obs_list.append(obs_dict)
            obs_dict = combine_obs_dicts(self.obs_list)

        # currently set all action as safe action
        safe_action_dict = generate_safe_action(self.CAV_dict.keys(), simple_n=3+self.discretize)

        return obs_dict, all_car_info_dict, safe_action_dict

    def step(self, action_dict=None, step_n=-1):
        """
        :param action: [throttle, brake, steering]
        :returns:
        """

        force_continue = False
        # 2. Create and apply a batch of Carla VehicleControls.
        #    (Must be applied as batch-- see Carla documentation.)
        # Runs one step of behavior planning, path planning, and control
        try:

            batch = []
            for vid, cav in self.CAV_dict.items():
                
                if not action_dict or vid not in action_dict:
                    action = [0.6, 0., 0.]
                    print("Using hardcode action")
                else:
                    action = action_dict[vid]
                #!!! replace this action with your control input.

                cav_control = carla.VehicleControl(throttle=action[0], \
                                                   brake=action[1], \
                                                   steer=action[2])
                batch.append(carla.command.ApplyVehicleControl(cav, cav_control))

            # ----- This part could be simplified ------ #
            # ----- (the logics for HAZV and NCAV could be deduced from all_car_info_dict) ----- #
            if self.map_name=='Town06':
                for vid, ncav in self.ncav_vehicles.items():
                    ncav_steer = 0
                    # --sabbir_control (True / False flag)                    
                    '''
                        you can change the specific control for each individual ucv here

                    '''
                    if self.args.sabbir_control:
                        pass

                    # set ncav with constant throttle value
                    elif step_n < self.ucv_force_straight_step: # == 66
                        force_throttle = 0.85
                        if self.scenario == 'crossing':
                            force_throttle = 0.85
                        ncav_control = carla.VehicleControl(throttle=force_throttle, steer=ncav_steer)
                        batch.append(carla.command.ApplyVehicleControl(ncav, ncav_control))
                        continue

                        '''
                            HIGHWAY:
                            0 - force_straight_step: acc max
                            FSS ~ final
                                HAZV and hazard: brake
                            FSS - 100: acc ucv_throttle
                            100 onwards
                                UCV or normal behave: drive within a range
                                HAZV and hazard: brake hard after 100 step 
                        '''
                    elif self.scenario == 'highway':
                        if self.normal_behave or (vid not in self.hazard_ncav_ids) or \
                            (vid in self.hazard_ncav_ids and step_n < self.hzv_hardbrake_start):
                            # target a speed
                            ncav_speed = self.ncav_states[vid][3:5]
                            ncav_speed = math.sqrt(ncav_speed[0]**2+ncav_speed[1]**2)
                            ncav_acc_revise = np.random.rand()*0.04 - 0.02
                            # To sabbir: we will try to maintain speed at this value
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
                            FSS: force_straight_steps=66
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
                            else: # force_straight_step <= step < 206
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
            
            else:
                raise NotImplementedError("This map {} hasn't been implemented.".format(self.map_name))
            
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

        # generate error if experiment requires
        # generate_error_new(cav_ids, all_ids, e_type, error_bound, scenario, current_step=None, target_start_end=None, error_pos=None)
        error_dict, old_error_dict = None, None
        if self.e_type:
            if self.e_type == 'tgt_t':
                error_dict, old_error_dict = generate_error_new(cav_ids=self.CAV_dict.keys(), all_ids=[v[0] for v in self.vehicle_list], \
                                        e_type=self.e_type, error_bound=[3,1,3,3], scenario=self.scenario, \
                                        current_step=step_n, target_start_end=self.target_start_end,\
                                        error_pos=self.error_pos)
            elif self.e_type == 'tgt_v':
                error_dict, old_error_dict = generate_error_new(cav_ids=self.CAV_dict.keys(), all_ids=[v[0] for v in self.vehicle_list], \
                                        e_type=self.e_type, error_bound=[3,1,3,3], scenario=self.scenario, \
                                        error_pos=self.error_pos)
            else:
                error_dict, old_error_dict = generate_error_new(cav_ids=self.CAV_dict.keys(), all_ids=[v[0] for v in self.vehicle_list], \
                                        e_type=self.e_type, error_bound=[3,1,3,3], scenario=self.scenario)

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
        vels_rwds, dest_rwds, col_penalties, sact_rwds, cav_collision_dict \
            = self._get_reward(avg_p=1.0, col_r=True, include_env_collision=True, interpolate=((step_n+1) / self.default_steps_per_episode))
        done = (step_n >= self.episode_length)

        # from self.state to observation
        obs_dict = observation_from_state(self.state, self.CAV_dict.keys(), self.max_num_car, self.brief_state, error=error_dict, debug=False)

        if self.temporal_length > 1:
            self.obs_list.pop(0)
            self.obs_list.append(obs_dict)
            obs_dict = combine_obs_dicts(self.obs_list)
        
        # currently set all action as safe action
        safe_action_dict = generate_safe_action(self.CAV_dict.keys(), simple_n=3+self.discretize)

        return obs_dict, done, all_car_info_dict, (vels_rwds, dest_rwds, col_penalties, sact_rwds), safe_action_dict, force_continue, error_dict

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
