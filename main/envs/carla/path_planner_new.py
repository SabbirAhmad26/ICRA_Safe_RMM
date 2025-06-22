#!/usr/bin/env python
"""
path_planner.py

todo
"""

# Companion code for the UConn undergraduate Honors Thesis "Evaluating Driving
# Performance of a Novel Behavior Planning Model on Connected Autonomous
# Vehicles" by Keyur Shah (UConn '20). Thesis was advised by Dr. Fei Miao;
# see http://feimiao.org/research.html.
#
# This code is meant for use with the autonomous vehicle simulator CARLA
# (https://carla.org/).
#
# Disclaimer: The CARLA project, which this project uses code from, follows the
# MIT license. The license is available at https://opensource.org/licenses/MIT.

#from enum import Enum
from collections import deque
import random
import copy
import math
import carla
from .enums_old import RoadOption
from .controller import VehiclePIDController
#from .mpc.controller2d import Controller2D
from .tools.misc import draw_waypoints, angle_between


# for intersection scenario, the two lane ids are: 1, 2
# for merging scenario, the three lane ids are: 2,3,4
'''
class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving
    from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6
'''

class PathPlanner:
    """
    PathPlanner implements the basic behavior of following a trajectory of
    waypoints that is generated on-the-fly.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.8

    def __init__(self, vehicle, map_name, marker='O', discretize=5, use_cbf=True, spawn_transform=None, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds.
                  This is typically fixed from server side using the
                  command-line arguments `-benchmark -fps=F`
                  In this case dt = 1/F, i.e. dt = 0.05, F = 20.
            target_speed -- desired cruise speed in Km/h
            sampling_radius -- search radius for next waypoints in seconds:
                e.g. 0.5 seconds ahead
            lateral_control_dict -- dictionary of arguments to setup the
                lateral PID controller, e.g. {'K_P':, 'K_D':, 'K_I':, 'dt'}
            longitudinal_control_dict -- dictionary of arguments to setup the
                longitudinal PID controller, e.g.  {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self.map_name = map_name
        self._marker = marker
        self.discretize = discretize # number of value ranges allowed for ACC/BRK
        self._use_cbf = use_cbf
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._spawn_transform = spawn_transform
        self._tb_command = None

        # course charted by path planner
        # contains tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=100)
        self._buffer_size = 5
        # immediate next few waypoints in the planned path (helpful for
        # controllers like MPC which use the next SEVERAL reference positions)
        # In our case, PID only uses the next SINGLE waypoint so it's unused
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        #self._print_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        """
        todo
        """
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self, opt_dict):
        # default params
        self._dt = 1.0 / 20.0
        # self._switch_timestep = 0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon # initially 5.6 s
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE # 5m
        args_lateral_dict = {
            'K_P': 5, # Keyur: 0.5  Local_planner: 1.95  Traffic_manager: 10
            'K_D': 0.001, # Keyur: 0.01  Local_planner: 0.2  Traffic_manager: 0
            'K_I': 1.0, # Keyur: 1.4  Local_planner: 0.07  Traffic_manager: 0.1
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 3.0, # Keyur: 1.0  Local_planner: 1.0  Traffic_manager: 5.0
            'K_D': 0.15, # Keyur: 0  Local_planner: 0  Traffic_manager: 0
            'K_I': 0.1, # Keyur: 1  Local_planner: 0.05  Traffic_manager: 0.1
            'dt': self._dt}

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                                        opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        if self._spawn_transform is not None:
            self._current_waypoint = self._map.get_waypoint(self._spawn_transform.location)
        else:
            self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        curr_x, curr_y = self._current_waypoint.transform.location.x,\
                        self._current_waypoint.transform.location.y
        print("++++++++PP: CAV {:d} Init Current: ({:.2f}, {:.2f})".format(self._vehicle.id, curr_x, curr_y))

        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral = args_lateral_dict,
                                                        args_longitudinal = args_longitudinal_dict)

        # compute initial waypoints
        ini_wpt = self._current_waypoint.next(self._sampling_radius)[0]
        x,y = ini_wpt.transform.location.x, ini_wpt.transform.location.y
        print("++++++++CAV {:d} Init next: ({:.2f}, {:.2f})".format(self._vehicle.id, x,y))
        self._waypoints_queue.append((ini_wpt, RoadOption.LANEFOLLOW, 0))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=50, decision_step=-1)

    def set_lane_left(self, distance_ahead, decision_step=-1, debug=False):
        """
        todo

        :param distance_ahead:
        :param debug:
        """
        current_loc = self._vehicle.get_location()
        #if current_loc is not None:
        #    print("SET LANE LEFT from: ({:.2f}, {:.2f})".format(current_loc.x, current_loc.y))
        #else:
        #    print("SET LANE LEFT from NONE !!!!")
        current_waypoint = self._map.get_waypoint(current_loc)
        self._waypoint_buffer.clear()
        self._waypoints_queue.clear()
        
        left_waypt = current_waypoint.get_left_lane()
        if left_waypt is None:
            self.set_lane_origin(distance_ahead, decision_step=decision_step, debug=debug)
            if debug:
                print("++++++++PP: CAV {:d} Failed to CHANGE LEFT, left waypoint does not exist: {:d}".\
                        format(self._vehicle.id, current_waypoint.lane_id))
            return False

        erroneous_wp=[]
        left_front_wpts = []
        for i in range(5):
            temp_wpt = (left_waypt.next(distance_ahead+i*2.)[0], RoadOption.CHANGELANELEFT, decision_step)
            left_front_wpts.append(temp_wpt)
            temp_wpt_loc = temp_wpt[0].transform.location
            erroneous_wp.append((temp_wpt_loc.y < -90. or temp_wpt_loc.y > -76.))
        if self.map_name=='Town05' and any(erroneous_wp):
            self.set_lane_origin(distance_ahead, decision_step=decision_step, debug=debug)
            if debug:
                print("++++++++PP: CAV {:d} Failed to CHANGE LEFT, error waypoints sampled.".\
                        format(self._vehicle.id))
            return False

        for i in range(5):
            self._waypoints_queue.append(left_front_wpts[i])
        if debug:
            print("++++++++PP: CAV {}; CHANGE LEFT from lane {} into {}; b&q: {}, {}".format(self._vehicle.id, \
                current_waypoint.lane_id, left_waypt.lane_id, len(self._waypoint_buffer), len(self._waypoints_queue)))
        return True

    def set_lane_right(self, distance_ahead, decision_step=-1, debug=False):
        """
        todo

        :param distance_ahead:
        :param debug:
        """
        current_loc = self._vehicle.get_location()
        #if current_loc is not None:
        #    print("SET LANE RIGHT from: ({:.2f}, {:.2f})".format(current_loc.x, current_loc.y))
        #else:
        #    print("SET LANE RIGHT from NONE !!!!")
        current_waypoint = self._map.get_waypoint(current_loc)
        #current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._waypoint_buffer.clear()
        self._waypoints_queue.clear()
        
        right_waypt = current_waypoint.get_right_lane()
        if right_waypt is None:
            self.set_lane_origin(distance_ahead, decision_step=decision_step, debug=debug)
            if debug:
                print("++++++++PP: CAV {:d} Failed to CHANGE RIGHT, right waypoint does not exist: {:d}".\
                        format(self._vehicle.id, current_waypoint.lane_id))
            return False

        erroneous_wp=[]
        right_front_wpts = []
        for i in range(5):
            temp_wpt = (right_waypt.next(distance_ahead+i*2.)[0], RoadOption.CHANGELANERIGHT, decision_step)
            right_front_wpts.append(temp_wpt)
            temp_wpt_loc = temp_wpt[0].transform.location
            erroneous_wp.append((temp_wpt_loc.y < -90. or temp_wpt_loc.y > -76.))
        if self.map_name=='Town05' and any(erroneous_wp):
            self.set_lane_origin(distance_ahead, decision_step=decision_step, debug=debug)
            if debug:
                print("++++++++PP: CAV {:d} Failed to CHANGE Right, error waypoints sampled.".\
                        format(self._vehicle.id))
            return False

        for i in range(5):
            self._waypoints_queue.append(right_front_wpts[i])
        if debug:
            print("++++++++PP: CAV {}; CHANGE RIGHT from lane {} into {}; b&q: {}, {}".format(self._vehicle.id, \
                current_waypoint.lane_id, right_waypt.lane_id, len(self._waypoint_buffer), len(self._waypoints_queue)))
        return True

    def set_lane_origin(self, distance_ahead, decision_step=-1, debug=False):
        """
        todo

        :param distance_ahead:
        :param debug:
        """
        current_loc = self._vehicle.get_location()
        #if current_loc is not None:
        #    print("SET LANE ORIGIN from: ({:.2f}, {:.2f})".format(current_loc.x, current_loc.y))
        #else:
        #    print("SET LANE ORIGIN from NONE !!!!")
        current_waypoint = self._map.get_waypoint(current_loc)
        #current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        
        self._waypoint_buffer.clear()
        self._waypoints_queue.clear()

        target_wpts = []
        erroneous_wp = []
        for i in range(5):
            temp_wpt = (current_waypoint.next(distance_ahead+i*2.)[0], RoadOption.LANEFOLLOW, decision_step)
            target_wpts.append(temp_wpt)
            temp_wpt_loc = temp_wpt[0].transform.location
            erroneous_wp.append((temp_wpt_loc.y < -90. or temp_wpt_loc.y > -76.))

        if self.map_name=='Town05' and any(erroneous_wp):
            for i in range(5):
                target_location_i = current_loc + carla.Location(x=distance_ahead+i*2.)
                target_waypoint_i = self._map.get_waypoint(target_location_i)
                self._waypoints_queue.append((target_waypoint_i, RoadOption.LANEFOLLOW, decision_step))
            if debug:
                print("++++++++PP: CAV {:d} Error waypoints sampled for set_lane_origin, hard code the target waypoint.".\
                        format(self._vehicle.id))
        else:
            for i in range(5):
                self._waypoints_queue.append(target_wpts[i])
            if debug:
                print("++++++++PP: CAV {}; Set back to the original lane {}; b&q: {}, {}".format(self._vehicle.id, \
                    current_waypoint.lane_id, len(self._waypoint_buffer), len(self._waypoints_queue)))

    def set_speed(self, speed):
        """
        todo

        :param speed:
        """
        self._target_speed = speed

    def discretized_acc(self, acc_level=0):
        """
        :param speed:
        """
        self._tb_command = ('t', acc_level)

    def discretized_brk(self, brk_level=0):
        """
        :param speed:
        """
        self._tb_command = ('b', brk_level)

    def _compute_next_waypoints(self, k=1, decision_step=-1, debug=False):  # Adds k new waypoints to queue
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        #min_x, max_x, min_y, max_y = 2000., -2000., 2000., -2000.
        #print("Sample radius: {:.2f}".format(self._sampling_radius))

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0: # fixes a bug # DEBUG:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                road_options_list = _retrieve_options(
                    next_waypoints, last_waypoint)
                if RoadOption.STRAIGHT in road_options_list:
                    road_option = RoadOption.STRAIGHT
                else:
                    road_option = random.choice(road_options_list)

                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option, decision_step))
            #next_wpt_loc = next_waypoint.transform.location
            #x,y = next_wpt_loc.x, next_wpt_loc.y
            #min_x = min(min_x, x)
            #max_x = max(max_x, x)
            #min_y = min(min_y, y)
            #max_y = max(max_y, y)
        if debug:
            print("++++++++PP: CAV", self._vehicle.id, "compute new waypoints, queue len: ", len(self._waypoints_queue))
        #print("Sampled Waypoints range - x:[{:.2f}, {:.2f}]; y: [{:.2f}, {:.2f}]".\
        #    format(min_x, max_x, min_y, max_y))


    # this might be the reason for start-up cooling process
    # need validation
    def run_step(self, debug=False, step_n=-1, force_straight_step=70, previous_speed=0., brk_range=0.6,\
                 use_cbf_throttle=False, use_cbf_steering=False, cbf_solution=None):
        # not enough waypoints in the horizon? => add more!
        # queue存了所有的可能的waypoint
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=20, decision_step=step_n, debug=debug)

        if len(self._waypoints_queue) == 0 and len(self._waypoint_buffer) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # current vehicle waypoint
        vehicle_transform = self._vehicle.get_transform()
        self._current_waypoint = self._map.get_waypoint(vehicle_transform.location)

        # Buffering the first few waypoints
        # 你接下来马上要执行的几个waypoints
        if not self._waypoint_buffer:
            i = 0
            while i < self._buffer_size:
                if not self._waypoints_queue:
                    if debug:
                        #print("++++++++PP: Vehicle: {} must resample waypoints to queue".format(self._vehicle.id))
                        pass
                    self._compute_next_waypoints(k=20, decision_step=step_n, debug=debug)
                temp_wp = self._waypoints_queue.popleft()
                distance_to_waypoint = temp_wp[0].transform.location.distance(vehicle_transform.location)
                if distance_to_waypoint >= self._min_distance:
                    self._waypoint_buffer.append(temp_wp)
                    i += 1
        
        if not self._waypoint_buffer:
            raise RuntimeError("PP: Vehicle {} cannot have valid waypoint in buffer".format(self._vehicle.id))

        # target waypoint
        self.target_waypoint, self._target_road_option, decision_step= self._waypoint_buffer[0]
        if debug:
            #print("++++++++PP: Vehicle: {}; Step: {}; TRO: {}; Decision made step: {}".format(self._vehicle.id, step_n, \
            #            str(self._target_road_option), decision_step))
            pass
        
        tar_x, tar_y = self.target_waypoint.transform.location.x, \
                        self.target_waypoint.transform.location.y
        #print("CAV {:d} current Target: ({:.2f}, {:.2f})".\
        #    format(self._vehicle.id, tar_x, tar_y ))

        if step_n >=0 and step_n <= force_straight_step:
            # this is covered by below force acceleration
            control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)
        else:
            # !!! while < 10km/h always want to accelerate.
            #if previous_speed < 10.:
            #    previous_speed = self._target_speed
            previous_speed = max(10, previous_speed)
            control = self._vehicle_controller.run_step(previous_speed, self.target_waypoint)
        if self._tb_command is not None:
            tb, level = self._tb_command
            
            if tb=='t':
                #lower_bound = level * ((1 - brk_range)/ self.discretize)
                #upper_bound = (level+1) * ((1 - brk_range) / self.discretize)
                mu = brk_range + (level+0.5) * ((1 - brk_range) / self.discretize)
                sigma = (1 - brk_range) / (2*self.discretize)
                #control.throttle = random.uniform(lower_bound, upper_bound) + brk_range
                control.throttle = random.gauss(mu, sigma)
                control.brake = 0.
            else:
                if level==-1:
                    control.throttle = 0.
                    control.brake = random.gauss(-0.25, 0.25)
                else:
                    #lower_bound = level * (brk_range/ self.discretize)
                    #upper_bound = (level+1) * (brk_range / self.discretize)
                    mu = (level+0.5) * (brk_range / self.discretize)
                    sigma = brk_range / (2*self.discretize)
                    #control.throttle = random.uniform(lower_bound, upper_bound)
                    control.throttle = random.gauss(mu, sigma)
                    control.brake = 0.

            self._tb_command = None

        if use_cbf_throttle:
            if cbf_solution[0] >= 0:
                control.throttle = cbf_solution[0]
                control.brake = 0.
            else:
                control.throttle = 0.
                control.brake = abs(cbf_solution[0])
        if use_cbf_steering:
            control.steer = cbf_solution[1]
        
        if step_n >=0 and step_n <= force_straight_step:
            control.throttle = 0.92
            control.brake = 0.
            control.steer = 0.
        

        # purge the queue of obsolete waypoints
        max_index = -1

        for i, (waypoint, _, _) in enumerate(self._waypoint_buffer):
            distance_to_waypoint = waypoint.transform.location.distance(vehicle_transform.location)
            if distance_to_waypoint < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        #if debug:
        #    print("++++++++PP agent {}: poped {}; left in Buffer {}; left in queue {}".format(self._vehicle.id, \
        #        max_index+1, len(self._waypoint_buffer), len(self._waypoints_queue)))

        if debug:
            draw_waypoints(self._vehicle.get_world(), self._waypoint_buffer, marker=self._marker, draw_str=True)
        return control


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and
    the multiple waypoints present in list_waypoints. The result is encoded as
    a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of
        multiple options.
    :param current_waypoint: current active waypoint.
    :return: list of RoadOption enums representing the type of connection from
        the active waypoint to eachcandidate in list_waypoints.
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the begining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options

def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint
    (current_waypoint) and a target waypoint (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
            LEFT = 1
            RIGHT = 2
            STRAIGHT = 3
            LANEFOLLOW = 4
            CHANGELANELEFT = 5
            CHANGELANERIGHT = 6
    """
    next_fwd = next_waypoint.transform.get_forward_vector()
    curr_fwd = current_waypoint.transform.get_forward_vector()
    diff_vec = next_waypoint.transform.location - current_waypoint.transform.location

    #diff_angle_of_direction = math.atan2(next_fwd.y*curr_fwd.x-next_fwd.x*curr_fwd.y, next_fwd.x*curr_fwd.x+next_fwd.y*curr_fwd.y)
    diff_angle_of_direction = angle_between(curr_fwd, next_fwd)
    diff_angle_of_direction = math.degrees(diff_angle_of_direction)

    #diff_angle_of_position = math.atan2(diff_vec.y*curr_fwd.x-diff_vec.x*curr_fwd.y, diff_vec.x*curr_fwd.x+diff_vec.y*diff_vec.y)
    diff_angle_of_position = angle_between(curr_fwd, diff_vec)
    diff_angle_of_position = math.degrees(diff_angle_of_position)

    if abs(diff_angle_of_position) <5.0:
        # 移动方向和当前朝向接近 没有换lane
        if diff_angle_of_direction >45.: # turn left
            return RoadOption.LEFT
        elif diff_angle_of_direction < -45.: # turn right
            return RoadOption.RIGHT
        else:
            return RoadOption.LANEFOLLOW
    else:
        # waypoint 换lane了
        if diff_angle_of_direction > 45.:
            return RoadOption.LEFT
        elif diff_angle_of_direction < -45.:
            return RoadOption.RIGHT
        else:
            if diff_angle_of_position > 0:
                return RoadOption.CHANGELANELEFT
            else:
                return RoadOption.CHANGELANERIGHT

