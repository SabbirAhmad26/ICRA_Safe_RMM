#!/usr/bin/env python

import sys
sys.path.append('../')

import numpy as np
import math
import carla
from queue import Queue

from .behavior_planner import BehaviorPlanner
#from .path_planner import RoadOption, PathPlanner
from .path_planner_new import PathPlanner
from .tools.misc import (scalar_proj, 
    dot, norm, norm2d, cross, angle_between, 
    pseudo_vehicle_rotation, draw_points)
from .enums_old import Behavior, RoadOption



# input: current vehicle's location, heading direction and target_lane_waypoint
# the lane change finish condition is that 
#     dist_to_middle_of_lane < 0.5 and angle_between < 5.0 degree
def lane_change_finish(current_transform, target_lane_wp):
    target_fwd = target_lane_wp.transform.get_forward_vector()
    current_location = current_transform.location
    heading = current_transform.get_forward_vector()
    relative_dist_vec = current_location - target_lane_wp.transform.location

    # vehicle's perpendicular distance to the lane middle line
    dy = cross(relative_dist_vec, target_fwd, norm=True) / norm(target_fwd)
    # angle between the target wp's heading and vehicle's current heading vector
    phi = angle_between(target_fwd, heading, degree=True)

    return abs(phi)< 8. and dy < 0.6



class CAVBehaviorPlanner(BehaviorPlanner):
    """
    CAVBehaviorPlanner uses the connected behavior-planning algorithm to make
    lane change decisions
    
    Example usage:
        # vehicle, param_dict, carla world already instantiated
        CAV_agents_dict = {}
        agent = CAVBehaviorPlanner(
            1/20, 40, vehicle, param_dict, CAV_agents_dict, 1
            )
        CAV_agents_dict[agent.vehicle.id] = CAV_agents_dict
        agent.set_behavior(0) # 0 corresponds to "KEEP_LANE" enum

        # make sure to apply these controls as a batch
        # in the multi-vehicle, synchronous CARLA scenario,
        # since ApplyVehicleControl implicitly ticks
        agent_action = agent.run_step(debug=False)
        for ii in range(100):
            carla.command.ApplyVehicleControl(
                agent.vehicle, agent_action
            )
    """

    def __init__(self, dt, target_speed, vehicle, map_name, param_dict, CAV_agents_dict, \
        num_CAVs, marker=None, discretize=2, lane_change_cooling_time=60, \
        lane_bounds=None, remove_CBFA=False, spawn_transform=None):
        """
        todo

        :param dt:
        :param target_speed:
        :param vehicle:
        :param param_dict:
        :param CAV_agents_dict:
        :param num_CAVs:
        """
        # todo - Consider using Python 3 style super() without arguments
        super(CAVBehaviorPlanner, self).__init__(vehicle, dt, param_dict)
        self.dt = dt
        self.target_speed = target_speed
        self.marker = marker
        self.path_planner = PathPlanner(
            vehicle=self.vehicle,
            map_name = map_name,
            marker=marker,
            discretize=discretize,
            spawn_transform=spawn_transform,
            opt_dict={
                'dt': dt,
                'target_speed': target_speed
            }
        )
        self.F = param_dict['F']
        self.w = param_dict['w']
        self.theta_CL = param_dict['theta_CL']
        self.theta_CR = param_dict['theta_CR']
        self.eps = param_dict['eps']
        self.CAV_agents_dict = CAV_agents_dict
        self.num_CAVs = num_CAVs
        self.lane_bounds = lane_bounds
        self.map_name = map_name
        self.change_distance = param_dict['chg_distance']
        self.lane_change_cooling_time = lane_change_cooling_time
        self.cooling_timer = 0
        self.remove_CBFA = remove_CBFA
        
        if lane_bounds[0]==1:
            self.base_lane = 2
        else:
            self.base_lane = 3

        self.chg_hazard_l = False
        self.hazard_c = False
        self.chg_hazard_r = False

        self.switcher_step = 0  # cycles through 0, 1, ..., (Tds - 1) each timestep
        self.Qv_l = 0.9 * self.target_speed
        self.Qv_c = 0.9 * self.target_speed
        self.Qv_r = 0.9 * self.target_speed
        self.change_buf = Queue(maxsize=self.F)
        while not self.change_buf.full():
            self.change_buf.put(False)
        self.Qf = 0
        self.rCL = 0
        self.rCR = 0

        self.neighbor_left = []
        self.neighbor_current = []
        self.neighbor_right = []

        self.vehicle = vehicle
        self.behavior = Behavior.KEEP_LANE
        self.cbf_solution = None
        self.lane_id_before_junction = None

        self.closeneighbor_left = []
        self.closeneighbor_current = []
        self.closeneighbor_right = []
        #self.close_eps = 30
        self.close_eps = 12 # here we need to discuss how to set this value
        self.neighbor_threshold = 200.

        self.neighbor_info = np.zeros((self.num_CAVs-1, 2))
        #self.neighbor_info = np.zeros((self.num_CAVs, 2))

    '''
        This function supports the CBF of agent at each timestep
        It obtains vehicles from given vicinity, and they categorize them into several
        1. vehicle that is on current road
        2. vehicle that is not on current road, but a collision could happen
        3. vehicle that is not on current road and no collision could happen

        The condition for 2 and 3 could vary, being either
        1. the ego road direction and target road direction has no connection
        2. the ego vehicle directin and target vehicle direction has no connection 
    '''
    def detect_env(self, scenario='highway', crossV_ET_range=None, debug=False, consider_curvy=False, previous_action_dict=None, error=None):
        assert scenario in ['highway', 'crossing', 'merging']
        crossV_ET_low, crossV_ET_high = 0, math.inf
        if crossV_ET_range:
            crossV_ET_low, crossV_ET_high = crossV_ET_range
            assert (0 <= crossV_ET_low and crossV_ET_low < 1) and (1 < crossV_ET_high)
        if self.map_name == 'Town05':
            road_sections = [7, 6, 1140, 1193, 1236] # !!!!!!! this is hard coded
        elif self.map_name == 'Town06':
            if scenario == 'highway':
                road_sections = [213, 35, 36, 37, 38, 391] + [570, 575, 1149, 1037]
            elif scenario == 'crossing':
                road_sections = [12,13,14,15,16,25,28,29,30] + [764, 796, 797, 783, 453, 546, 545, 799, 735, 1083]
            elif scenario == 'merging':
                road_sections = [47,48,49,50,51,52] + [1206, 1216, 1164, 1175, 1176, 1180, 680, 652, 653, 415]
        
        if error:
            e_type, temp_car_error = error
        #return closest_front, closest_rear, 
        #   closest_left_front, closest_left_rear
        #   closest_right_front, closest_right_rear
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()

        # define the change distance for lane-change
        #self.change_distance = norm(vel) * 3.

        car_transform = self.vehicle.get_transform()
        car_rotation = car_transform.rotation
        heading = car_rotation.get_forward_vector()
        self.current_waypoint = self.map.get_waypoint(loc)
        fwd = self.current_waypoint.transform.get_forward_vector()
        speed = norm(vel)
        # from fwd to heading
        # angle = arctan( (fwd x heading) / (fwd * heading) )
        # angle = atan2( cross_product / dot_product )
        psi = angle_between(fwd, heading)
        #psi = math.atan2(heading.y*fwd.x-heading.x*fwd.y, heading.x*fwd.x+heading.y*fwd.y)
        cbf_state = [loc.x, loc.y, psi, speed]
        
        left_waypt = self.current_waypoint.get_left_lane()
        right_waypt = self.current_waypoint.get_right_lane()
        if left_waypt:
            left_left_waypoint = left_waypt.get_left_lane()
        if right_waypt:
            right_right_waypoint = right_waypt.get_right_lane()

        # keys: cf, cb, lf, lb, rf, rb
        # each element is: (relative longitude distance, longitude velocity)
        neighbors = {}
        # each element is: (loc_x, loc_y, v_x, v_y)
        others = []

        for other in self.world.get_actors().filter("*vehicle*"):
            # must be a different vehicle
            if self.vehicle.id == other.id:
                continue

            # if the error type is miss-detection and current vehicle happened to be neglected by agent
            if error and (e_type=='miss') and (other.id not in temp_car_error):
                continue

            other_loc = other.get_location()
            other_waypoint = self.map.get_waypoint(other_loc)
            other_fwd = other_waypoint.transform.get_forward_vector()
            other_heading = other.get_transform().get_forward_vector()
            other_velocity = other.get_velocity()
            other_speed = norm(other_velocity)
            if loc.distance(other_loc) > self.neighbor_threshold:
                continue
            # must be on the same segment of road as ego vehicle
            #if other_waypoint.road_id != self.current_waypoint.road_id:
            if  (other_waypoint.road_id in road_sections) and \
                (   (scenario in ['highway', 'merging']) or \
                    (scenario == 'crossing' and \
                        abs(angle_between(heading, other_heading, degree=True)) < 45.)):
                # could be positive or negative
                longi_dist = scalar_proj(other_loc-loc, fwd)

                if consider_curvy:
                    other_fwd_speed = scalar_proj(other_velocity, other_heading)
                    other_fwd_acc = scalar_proj(other.get_acceleration(), other_heading)
                else:
                    # only consider straight road
                    other_fwd_speed = scalar_proj(other_velocity, fwd)
                    other_fwd_acc = scalar_proj(other.get_acceleration(), fwd)

                if error and (e_type in ['noise', 'l_attack', 'f_attack', 'tgt_v', 'tgt_t']) and (other.id in temp_car_error):
                    longi_dist += temp_car_error[other.id][0]
                    other_fwd_speed += temp_car_error[other.id][1]

                nbr_state = [other.id, longi_dist, other_fwd_speed, other_fwd_acc]
                #if debug:
                #if other.id == self.vehicle.id-4:
                #    print("CBP detect env: Ego {} found other: {}, longi_dist: {}".format(self.vehicle.id, other.id, longi_dist))

                # Other is on LEFT lane
                if left_waypt and other_waypoint.lane_id == left_waypt.lane_id:
                    if longi_dist >= 0: # left front, 'lf'
                        if 'lf' in neighbors:
                            if longi_dist < neighbors['lf'][1]:
                                neighbors['lf'] = nbr_state
                        else:
                            neighbors['lf'] = nbr_state
                    else: # left back, 'lb'; 
                        if 'lb' in neighbors:
                            if longi_dist > neighbors['lb'][1]:
                                neighbors['lb'] = nbr_state
                        else:
                            neighbors['lb'] = nbr_state
                elif (not self.remove_CBFA) and left_waypt and left_left_waypoint and \
                        other_waypoint.lane_id==left_left_waypoint.lane_id:
                    # other is on the left_left and is changing right!
                    if previous_action_dict and (other.id in previous_action_dict) and \
                        (previous_action_dict[other.id]==Behavior.CHANGE_RIGHT):
                        # assume the vehicle is driving on left lane
                        if longi_dist >= 0: # left front, 'lf'
                            if 'lf' in neighbors:
                                if longi_dist < neighbors['lf'][1]:
                                    neighbors['lf'] = nbr_state
                            else:
                                neighbors['lf'] = nbr_state
                        else: # left back, 'lb'; 
                            if 'lb' in neighbors:
                                if longi_dist > neighbors['lb'][1]:
                                    neighbors['lb'] = nbr_state
                            else:
                                neighbors['lb'] = nbr_state

                # Other is on CURRENT lane
                elif other_waypoint.lane_id == self.current_waypoint.lane_id:
                    if longi_dist >= 0: # current front, 'cf'
                        if 'cf' in neighbors:
                            if longi_dist < neighbors['cf'][1]:
                                neighbors['cf'] = nbr_state
                        else:
                            neighbors['cf'] = nbr_state
                    else: # current back, 'cb'; 
                        if 'cb' in neighbors:
                            if longi_dist > neighbors['cb'][1]:
                                neighbors['cb'] = nbr_state
                        else:
                            neighbors['cb'] = nbr_state

                # Other is on RIGHT lane
                elif right_waypt and other_waypoint.lane_id == right_waypt.lane_id:
                    if longi_dist >= 0: # right front, 'rf'
                        if 'rf' in neighbors:
                            if longi_dist < neighbors['rf'][1]:
                                neighbors['rf'] = nbr_state
                        else:
                            neighbors['rf'] = nbr_state
                    else: # right back, 'rb'; 
                        if 'rb' in neighbors:
                            if longi_dist > neighbors['rb'][1]:
                                neighbors['rb'] = nbr_state
                        else:
                            neighbors['rb'] = nbr_state
                elif (not self.remove_CBFA) and right_waypt and right_right_waypoint and \
                        other_waypoint.lane_id==right_right_waypoint.lane_id:
                    # other is on the left_left and is changing right!
                    if  previous_action_dict and (other.id in previous_action_dict) and \
                        (previous_action_dict[other.id]==Behavior.CHANGE_LEFT):
                        if longi_dist >= 0: # right front, 'rf'
                            if 'rf' in neighbors:
                                if longi_dist < neighbors['rf'][1]:
                                    neighbors['rf'] = nbr_state
                            else:
                                neighbors['rf'] = nbr_state
                        else: # right back, 'rb'; 
                            if 'rb' in neighbors:
                                if longi_dist > neighbors['rb'][1]:
                                    neighbors['rb'] = nbr_state
                            else:
                                neighbors['rb'] = nbr_state

            else:
                # loc, fwd, other_loc, other_heading
                '''
                    connection: if two traj connect, returns the location; otherwise False
                    neighbor: flag, True if target vehicle is also driving on current lane
                    opposite: flag, True if target vehicle is driving from opposite direction AND on current lane
                    target_v: [pseudo_x, pseudo_y, pseudo_V_x, pseudo_V_y, pseudo_relative_distance]
                '''
                ego_direction = fwd
                if abs(angle_between(fwd, heading, degree=True)) > 45.:
                    ego_direction = heading
                
                #if debug:
                #    print("other ID: {}".format(other.id), other_loc, other_heading)
                #    temp_tgt_loc = carla.Location(x=other_loc.x+10*other_heading.x, y=other_loc.y+10*other_heading.y, z=2.)
                #    self.world.debug.draw_string(temp_tgt_loc, text='X', color=carla.Color(255,0,0,255), life_time=0.05)
                connection, neighbor, opposite, target_v = pseudo_vehicle_rotation(loc, ego_direction, other_loc, other_heading, other_speed)
                
                if not target_v: # did not detect a relevant crossing vehicle
                    continue
                
                longi_dist = target_v[-1]
                other_fwd_speed = norm2d(target_v[2], target_v[3])

                if error and (e_type in ['noise', 'l_attack', 'f_attack', 'tgt_v', 'tgt_t']) and (other.id in temp_car_error):
                    longi_dist += temp_car_error[other.id][0]
                    other_fwd_speed += temp_car_error[other.id][1]
                
                nbr_state = [other.id, longi_dist, other_fwd_speed, 0]

                if debug and (connection or neighbor):
                    pseudo_pt = np.array([target_v[:2]])
                    draw_points(self.world, pseudo_pt, sample=1, z=5.0, marker='*'+self.marker, color=carla.Color(255,0,0,255))
                    if connection:
                        print("connection", connection, neighbor, opposite)
                        con_pt = np.array([connection])
                        draw_points(self.world, con_pt, sample=1, z=5.0, marker='C', color=carla.Color(255,255,0,255))
                    print("PSEUDO-CBF: Car {} - pseudo distance: {}; pseudo speed: {}".format(other.id, longi_dist, other_fwd_speed))
                if connection: 
                    # here we compute the expected time for ego V and target V to reach the connection (potential collision)
                    # if the ratio is out-of_range (ex. [0.5, 1.5]), then we don't consider such vehicle to be a threat.
                    ego_ET2_connection = norm2d(loc.x - connection[0], loc.y - connection[1]) / max(speed, 0.1)
                    tgt_ET2_connection = norm2d(other_loc.x - connection[0], other_loc.y - connection[1]) / max(other_speed, 0.1)
                    ET_ratio = tgt_ET2_connection / max(ego_ET2_connection, 1.)
                    if crossV_ET_range and (crossV_ET_low <= ET_ratio ) and (ET_ratio <= crossV_ET_high):
                        if longi_dist >= 0: # current front, 'cf'
                            if 'cf' in neighbors:
                                if longi_dist < neighbors['cf'][1]:
                                    neighbors['cf'] = nbr_state
                            else:
                                neighbors['cf'] = nbr_state
                        else: # current back, 'cb'; 
                            if 'cb' in neighbors:
                                if longi_dist > neighbors['cb'][1]:
                                    neighbors['cb'] = nbr_state
                            else:
                                neighbors['cb'] = nbr_state
                elif neighbor and (not opposite):
                    if longi_dist >= 0: # current front, 'cf'
                        if 'cf' in neighbors:
                            if longi_dist < neighbors['cf'][1]:
                                neighbors['cf'] = nbr_state
                        else:
                            neighbors['cf'] = nbr_state
                    else: # current back, 'cb'; 
                        if 'cb' in neighbors:
                            if longi_dist > neighbors['cb'][1]:
                                neighbors['cb'] = nbr_state
                        else:
                            neighbors['cb'] = nbr_state
                elif opposite: # this could be discussed
                    continue

                
        return cbf_state, neighbors, others

    def left_change_conflict_detection(self):
        """
        detect whether there is a conflict with other vehicles on the left lane
        before starting a lane-changing

        :returns: Bool, todo
        """
        # i.e. Return True if any of our leftward neighbors
        # are a CAV and their 'discrrete_state' is CHANGELANELEFT
        # or is CHANGELANERIGHT
        for vehicle in self.closeneighbor_left:
            if vehicle.id in self.CAV_agents_dict.keys():
                # todo; why construct dict here?
                if self.CAV_agents_dict[vehicle.id].discrete_state() in {RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT}:
                    return True

        return False

    def right_change_conflict_detection(self):
        """
        detect whether there is a conflict with other vehicles on the right lane
        before starting a lane-changing

        :returns: Bool, todo
        """
        # i.e. Return True if any of our rightward neighbors
        # are a CAV and their 'discrrete_state' is CHANGELANELEFT
        # or is CHANGELANERIGHT
        for vehicle in self.closeneighbor_right:
            if vehicle.id in self.CAV_agents_dict.keys():
                # todo; why construct dict here?
                if self.CAV_agents_dict[vehicle.id].discrete_state() in {RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT}:
                    return True

        return False

    """
    todo

    :param debug:
    :type debug: Bool
    
    :returns: A VehicleControl to be applied on the given vehicle
    :rtype: carla.libcarla.VehicleControl
    """
    def run_step(self, scenario, debug=False, step_n=-1, force_straight_step=70, mute_lane_change=False, previous_speed=0., cbf_safe_action=None):
    
        # define the change distance for lane-change
        vel = self.vehicle.get_velocity()
        self.change_distance = max(norm(vel) * 2.5, 12)
        
        current_transform = self.vehicle.get_transform()
        current_loc = current_transform.location
        #if current_loc is not None:
        #    print("CAVBP from: ({:.2f}, {:.2f})".format(current_loc.x, current_loc.y))
        #else:
        #    print("CAVBP from NONE !!!!")
        self.current_waypoint = self.map.get_waypoint(current_loc)# 这个是carla标准的waypoint基于vehicle current location

        if not self.current_waypoint.is_junction:
            self.lane_id_before_junction = self.current_waypoint.lane_id
        #self.detect_nearby_vehicles()

        '''
        When to use cbf throttle and steering?
        0. emergency_stop
            - don't use cbf; directly set max braking
        1. ongoing change left or right
            - if hazard and abort the change_lane, don't use cbf; let PID do keep_lane
            - if no hazard, don't use cbf; use the PID
        2. ongoing lane_follow:
            - if decide to change lane:
                - use cbf_throttle and cbf_steering
            - if decide to keep_lane
                - use cbf_throttle and cbf_steering
            - if decide to acc/brk
                - use cbf_throttle and cbf_steering
        '''
        use_cbf_throttle = False
        use_cbf_steering = False
        #print("%%%%CBP 419")
        # location constraints - if go across the middle lane boundary, punish
        #hazard_zone = (current_loc.y < -90. or current_loc.y > -76.) # this only applies to one scenario

        #if self.hazard_c or self.behavior == Behavior.EMERGENCY_STOP or hazard_zone:
        if self.hazard_c or self.behavior == Behavior.EMERGENCY_STOP:
            self.behavior = Behavior.EMERGENCY_STOP
            # !!! be careful about this part, need to modify and remove the second output later
            if debug:
                print("++++CBP: CAV {}: Emergency stop!".format(self.vehicle.id))
            return self.emergency_stop(), self.behavior


        if self.discrete_state() == RoadOption.CHANGELANELEFT: # change_left ongoing
            #print("%%%%CBP 427")
            #if self.chg_hazard_l or self.left_change_conflict_detection():
            if  step_n <= force_straight_step \
                or self.chg_hazard_l \
                or mute_lane_change \
                or (cbf_safe_action and cbf_safe_action[1]==0):
                # Cancel the attempted lane change
                self.path_planner.set_lane_origin(self.change_distance, decision_step=step_n, debug=debug)
                if debug:
                    print("++++CBP: CAV {}: Left change aborted".format(self.vehicle.id))
                self.behavior = Behavior.KEEP_LANE
            else: # if no hazard, force to finish the change_left
                if lane_change_finish(current_transform, self.path_planner.target_waypoint): # the lane change has finished
                    if debug:
                        print("++++CBP: CAV {}: Left change Finished, set cooling timer.".format(self.vehicle.id))
                    self.cooling_timer = self.lane_change_cooling_time
                    self.path_planner.set_lane_origin(self.change_distance, decision_step=step_n, debug=debug)
                    self.behavior = Behavior.KEEP_LANE
                else:
                    self.behavior = Behavior.CHANGE_LEFT

        elif self.discrete_state() == RoadOption.CHANGELANERIGHT: # change_right ongoing
            #print("%%%%CBP 438")
            #if self.chg_hazard_r or self.right_change_conflict_detection():
            if  step_n<=force_straight_step \
                or self.chg_hazard_r \
                or mute_lane_change \
                or (cbf_safe_action and cbf_safe_action[2]==0):
                # Cancel the attempted lane change
                self.path_planner.set_lane_origin(self.change_distance, decision_step=step_n, debug=debug)
                if debug:
                    print("++++CBP: CAV {}: Right change aborted".format(self.vehicle.id))
                self.behavior = Behavior.KEEP_LANE
            else: # if no hazard, force to finish the change_right
                if lane_change_finish(current_transform, self.path_planner.target_waypoint): # the lane change has finished
                    if debug:
                        print("++++CBP: CAV {}: Right change Finished, set cooling timer.".format(self.vehicle.id))
                    self.cooling_timer = self.lane_change_cooling_time
                    self.path_planner.set_lane_origin(self.change_distance, decision_step=step_n, debug=debug)
                    self.behavior = Behavior.KEEP_LANE
                else:
                    self.behavior = Behavior.CHANGE_RIGHT

        elif (self.discrete_state() == RoadOption.LANEFOLLOW):
            #and self.path_planner.target_waypoint
            #and self.switcher_step == self.Tds - 1):

            # suppress the abrupt change of lanes -> if you happen to hit a change_left
            #print("%%%%CBP 451")
            lane_changed = False

            # special checking for vehicle in intersection whose lane_id is not correct
            temp_lane_id = self.lane_id_before_junction

            if self.map_name=='Town06':
                if scenario == 'crossing':
                    temp_road_id = self.current_waypoint.road_id
                    if temp_road_id in [14, 15]:
                        self.lane_bounds = (-4, -6)
                    elif temp_road_id in [12, 13, 453, 546, 545, 764, 796, 797, 783]:
                        self.lane_bounds = (-3, -5)
                    else:
                        self.lane_bounds = (-3, -5)
                # else:
                #     self.lane_bounds = (-3, -6)


            if self.map_name=='Town05' and self.current_waypoint.is_junction \
                and (temp_lane_id < self.lane_bounds[0] or temp_lane_id > self.lane_bounds[1]):
                if  current_loc.y > -86.0:
                    temp_lane_id = 2
                else:
                    temp_lane_id = 1


            # Check if we can change left
            if self.behavior == Behavior.CHANGE_LEFT:
                #lane_changed = False
                #print("%%%%CBP 465")
                if step_n > force_straight_step: # we force straight acc before 70 steps
                    change_left_condition = (temp_lane_id != self.lane_bounds[0] # 如果在左边lane，那么不能继续往左并线
                                             and not (self.current_waypoint.is_junction) 
                                             and str(self.current_waypoint.lane_change) in {'Left', 'Both'}
                                             and not self.chg_hazard_l
                                             and self.cooling_timer==0)
                                             #and not self.left_change_conflict_detection())
                    if change_left_condition:
                        #print("%%%%CBP 473 pass change left cond")
                        lane_changed = self.path_planner.set_lane_left(self.change_distance, decision_step=step_n, debug=debug)
                        # !!! need to discuss
                        use_cbf_throttle = lane_changed
                        use_cbf_steering = False  
                if not lane_changed:
                    if debug:
                        if step_n > force_straight_step and change_left_condition:
                            print("++++CBP: CAV {}: Left change abandoned; PP.set_lane_left() failed".format(self.vehicle.id))
                        else:
                            print("++++CBP: CAV {}: Left change abandoned; condition unsatisfied".format(self.vehicle.id))
                    self.behavior = Behavior.KEEP_LANE

            # Check if we can change right
            elif self.behavior == Behavior.CHANGE_RIGHT:
                #lane_changed = False
                #print("%%%%CBP 484")
                if step_n > force_straight_step:
                    change_right_condition = (temp_lane_id != self.lane_bounds[1]
                                              and not (self.current_waypoint.is_junction)
                                              and str(self.current_waypoint.lane_change) in {'Right', 'Both'}
                                              and not self.chg_hazard_r
                                              and self.cooling_timer==0)
                                              #and not self.right_change_conflict_detection())
                    if change_right_condition:
                        #print("%%%%CBP 492 pass change right cond")
                        lane_changed = self.path_planner.set_lane_right(self.change_distance, decision_step=step_n, debug=debug)
                        use_cbf_throttle = lane_changed
                        use_cbf_steering = False

                if not lane_changed:
                    if debug:
                        if step_n > force_straight_step and change_right_condition:
                            print("++++CBP: CAV {}: Right change abandoned; PP.set_lane_right() failed".format(self.vehicle.id))
                        else:
                            print("++++CBP: CAV {}: Right change abandoned; condition unsatisfied".format(self.vehicle.id))
                    self.behavior = Behavior.KEEP_LANE

            else:
                self.path_planner.set_lane_origin(max(self.change_distance,10.), decision_step=step_n, debug=debug)
                ## BRK_0, ACC_1, BRK_1, ACC_2, BRK_2,..., ACC_5, BRK_5
                ## !! and BRK as decrease velocity by 10 if not less than 0.
                if self.behavior.value >= 3:
                    #use_cbf_control = True
                    use_cbf_throttle = True
                    use_cbf_steering = False
                    if self.behavior.value % 2 == 0: # 4, 6, 8, 10, 12
                        self.path_planner.discretized_acc((self.behavior.value-4)//2)
                    else: # 3, 5, 7, 9, 11, 13; 3 means brake
                        self.path_planner.discretized_brk((self.behavior.value-4)//2)
                else:
                    #use_cbf_control = True
                    use_cbf_throttle = True
                    use_cbf_steering = False
                    self.behavior = Behavior.KEEP_LANE
                    
            # Update Qf and save most recent change_lane value
            self.Qf = self.Qf - self.change_buf.get() + lane_changed
            self.change_buf.put(lane_changed)
        else:
            if debug:
                print("++++CBP: CAV {} unrecognized road option: {}".format(self.vehicle.id, str(self.discrete_state)))
            self.path_planner.set_lane_origin(self.change_distance, decision_step=step_n, debug=debug)
            self.behavior = Behavior.KEEP_LANE
            #raise RuntimeError("Unrecognized road option: {}".format(str(self.discrete_state)))

        if self.cooling_timer > 0:
            self.cooling_timer -= 1

        use_cbf_throttle = use_cbf_throttle and (self.cbf_solution is not None)
        use_cbf_steering = use_cbf_steering and (self.cbf_solution is not None)
        if debug:
            print("++++CBP: vehicle {} behavior eventually is {}, {} at step {}".format(self.vehicle.id, str(self.behavior),\
                    str(self.discrete_state()), step_n))
        #print("Vehicle Ctrl: {}; use_cbf_throttle: {}; use_cbf_steering: {}".format(\
        #        self.vehicle.id, str(use_cbf_throttle), str(use_cbf_steering)))
        #if use_cbf_steering or use_cbf_throttle:
        #    print("Used CBF ctrl: {}, {}".format(str(self.discrete_state()), str(self.behavior)))
        #print(use_cbf_throttle, use_cbf_steering, self.cbf_solution)
        self.switcher_step = (self.switcher_step + 1) % self.Tds
        return self.path_planner.run_step(debug=debug, step_n=step_n, force_straight_step=force_straight_step, \
                previous_speed=previous_speed, use_cbf_throttle=use_cbf_throttle, \
                use_cbf_steering=use_cbf_steering, cbf_solution=self.cbf_solution), \
               self.behavior

    def set_behavior(self, behavior, set_step=-1, debug=False):
        """Setter method for self.behvior
        :param behavior: Behavior enum.
            0 : KEEP_LANE
            1 : CHANGE_LEFT
            2 : CHANGE_RIGHT
        :type behavior: int (Behavior enum)
            3 : accelerate
            4 : brake
        """
        if debug:
            print("ENV: vehicle {} behavior is set to {} at step {}".format(self.vehicle.id, str(behavior), set_step))
        self.behavior = behavior

    def set_cbf_solution(self, cbf_solution):
        self.cbf_solution = cbf_solution

    def detect_nearby_vehicles(self, debug=False):
        """
        todo
        :param debug:
        """
        #self.close_eps = 30 # threshold to determine the close neighbors

        left_waypt = self.current_waypoint.get_left_lane()
        right_waypt = self.current_waypoint.get_right_lane()

        self.chg_hazard_l = False  # there is a hazard on the left
        self.hazard_c = False  # there is a hazard ahead
        self.chg_hazard_r = False  # there is a hazard on the right

        nbrs_l, nbrs_c, nbrs_r = [], [], []    # nbrs means neighbors; left, current, right
        self.neighbor_left, self.neighbor_current, self.neighbor_right = [], [], []
        self.closeneighbor_left, self.closeneighbor_current, self.closeneighbor_right = [], [], []
        #self.neighbor_info = np.zeros((self.num_CAVs, 2))
        self.neighbor_info = np.zeros((self.num_CAVs-1, 2))
        neighbor_info_index = 0

        for other in self.world.get_actors().filter("*vehicle*"):
            # must be a different vehicle
            if self.vehicle.id == other.id:
                continue

            other_loc = other.get_location()
            other_waypoint = self.map.get_waypoint(other_loc)

            # must be on the same segment of road as ego vehicle
            if other_waypoint.road_id != self.current_waypoint.road_id:
                continue

            loc = self.vehicle.get_location()
            fwd = self.current_waypoint.transform.get_forward_vector()
            # scalar_proj = dot(u, v)/norm(v) 
            other_fwd_speed = scalar_proj(other.get_velocity(),
                                          other_waypoint.transform.get_forward_vector())

            # Other is on LEFT lane
            if left_waypt and other_waypoint.lane_id == left_waypt.lane_id:
                #Check if it's an eps-neighbor
                # self.eps = 150 (meter), means the other vehicle is within communication range
                if loc.distance(other_loc) < self.eps and dot(loc - other_loc, fwd) <= 0:
                    nbrs_l.append(other_fwd_speed)
                    self.neighbor_left.append(other)

                    # consider the neighbor info includes lane number and distance
                    self.neighbor_info[neighbor_info_index] = [self.base_lane - \
                        other_waypoint.lane_id, loc.distance(other_loc)]
                    neighbor_info_index += 1

                if loc.distance(other_loc) < self.close_eps:
                    self.closeneighbor_left.append(other)

                # Check if it's a hazard. Any one hazard should make the flag stay true
                #self.chg_hazard_l = (self.chg_hazard_l or
                #    abs(norm(other.get_location() - loc)) < self.theta_l)

            # Other is on CURRENT lane
            elif other_waypoint.lane_id == self.current_waypoint.lane_id:
                #Check if it's an eps-neighbor
                if loc.distance(other_loc) < self.eps and dot(loc - other_loc, fwd) <= 0:
                    nbrs_c.append(other_fwd_speed)
                    self.neighbor_current.append(other)

                    # consider the neighbor info includes lane number and distance
                    self.neighbor_info[neighbor_info_index] = [self.base_lane - \
                        other_waypoint.lane_id, loc.distance(other_loc)]
                    neighbor_info_index += 1

                if loc.distance(other_loc) < self.close_eps:
                    self.closeneighbor_current.append(other)

                # Check if it's a hazard. Any one hazard should make the flag stay true
                #self.hazard_c = (self.hazard_c or
                #    loc.distance(other_loc) < self.theta_c and dot(loc - other_loc, fwd) <= 0)

            # Other is on RIGHT lane
            elif right_waypt and other_waypoint.lane_id == right_waypt.lane_id:
                #Check if it's an eps-neighbor
                if loc.distance(other_loc) < self.eps and dot(loc - other_loc, fwd) <= 0:
                    nbrs_r.append(other_fwd_speed)
                    self.neighbor_right.append(other)

                    # consider the neighbor info includes lane number and distance
                    self.neighbor_info[neighbor_info_index] = [self.base_lane - \
                        other_waypoint.lane_id, loc.distance(other_loc)]
                    neighbor_info_index += 1

                if loc.distance(other_loc) < self.close_eps:
                    self.closeneighbor_right.append(other)

                # Check if it's a hazard. Any one hazard should make the flag stay true
                #self.chg_hazard_r = (self.chg_hazard_r or
                #    abs(norm(other.get_location() - loc)) < self.theta_r)

        self.Qv_l = sum(nbrs_l)/len(nbrs_l) if nbrs_l else 0.9*self.target_speed
        self.Qv_c = sum(nbrs_c)/len(nbrs_c) if nbrs_c else 0.9*self.target_speed
        self.Qv_r = sum(nbrs_r)/len(nbrs_r) if nbrs_r else 0.9*self.target_speed

        self.rCL = self.w*(self.Qv_l - self.Qv_c) - self.Qf
        self.rCR = self.w*(self.Qv_r - self.Qv_c) - self.Qf

        if debug:
            print("Here is the neighbor info:\n", self.neighbor_info, "\n **********************")
