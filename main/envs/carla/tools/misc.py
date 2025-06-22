#!/usr/bin/env python

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

""" Module with auxiliary functions. """

import os
import glob
import sys
import math
import numpy as np
import networkx as nx
'''
try:
    os_version = 'win-amd64' if os.name == 'nt' else 'linux-x86_64'
    sys.path.append(glob.glob(f'/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.10-py3.7-{os_version}.egg')[0])
except IndexError:
    pass
'''
import carla


def get_data_from_graph(graph, state_len=7, adj='adj'):
    mat = None
    if adj == 'adj':
        mat = nx.adjacency_matrix(graph)
    elif adj == 'lap':
        mat = nx.laplacian_matrix(graph)
    elif adj == 'norm_lap':
        mat = nx.normalized_laplacian_matrix(graph)

    data = []
    node_dict = {k: v for v, k in enumerate(graph.nodes)}
    for node in graph.nodes:
        if 'state' in node.attr:
            data.append(node.attr['state'])
        else:
            data.append([0]*state_len)
    data = np.array(data)

    edges = []
    for n1, n2 in graph.edges:
        edges.append([node_dict[n1], node_dict[n2]])
    edges = np.array(edges).transpose()

    return data, edges, mat


def print_xys(locs):
    n = len(locs)
    loc_str = ""
    for i in range(n):
        temp_fmt_str = "({:.1f},{:.1f}); ".format(locs[i][0], locs[i][1])
        loc_str += temp_fmt_str
        if i%10 == 9:
            loc_str+="\n"
    loc_str += "Finished"
    print(loc_str)

# points is a 2d numpy array
def draw_points(world, points, sample=8, z=0.5, marker='*', color=carla.Color(255,0,0,255)):
    step = points.shape[0] // sample
    for i in range(0, points.shape[0], step):
        x,y = points[i]
        draw_loc = carla.Location(x=x, y=y, z=z)
        world.debug.draw_string(draw_loc, text=marker, color=color, life_time=0.05)


def draw_waypoints(world, waypoints, z=0.5, marker='O', draw_str=False):
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    for w, *_ in waypoints:
        if draw_str:
            
            world.debug.draw_string(w.transform.location, marker, draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0, a=255), life_time=0.05)
            
#            world.debug.draw_string(w.transform.location, 'O')
#            world.debug.draw_point(w.transform.location, size=0.1, color=(255,0,0), life_time=-1.0)
        
        else:            
            t = w.transform
            begin = t.location + carla.Location(z=z)
            angle = math.radians(t.rotation.yaw)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=0.05)


def get_speed(vehicle, no_z = False, mps=False):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :param no_z: use x and y components only
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()

    if no_z:
        vel_value = math.sqrt(vel.x ** 2 + vel.y ** 2)
    else:
        vel_value = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    if mps:
        return vel_value
    return 3.6 * vel_value


def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_transform: transform (pose) of the target object
    :param current_transform: transform (pose) of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2
    location_1, location_2:   carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]

def norm(v):
    return (v.x**2 + v.y**2 + v.z**2)**0.5

def norm2d(x,y):
    return (x**2 + y**2)**0.5

def dot(u, v):
    return u.x*v.x + u.y*v.y + u.z*v.z

def cross(u, v, norm=False):
    cross_prod = [u.y*v.z-u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y-u.y*v.x]
    if norm:
        return math.sqrt(cross_prod[0]**2+cross_prod[1]**2+cross_prod[2]**2)
    return cross_prod

def scalar_proj(u, v):
    return dot(u, v) / norm(v)

# the angle between u and v (from u to v); -\pi ~ \pi
def angle_between(u, v, degree=False):
    if degree:
        return math.degrees(math.atan2(v.y*u.x - v.x*u.y, u.x*v.x + u.y*v.y))
    return math.atan2(v.y*u.x - v.x*u.y, u.x*v.x + u.y*v.y)


def angle_between_vec(u,v):
    if (v[0]==0 and v[1]==0) or (u[0]==0 and u[1]==0):
        return 0.5 * math.pi
    return math.atan2(v[1]*u[0]-v[0]*u[1], u[0]*v[0]+u[1]*v[1])


'''
    p is the ego vehicle's location, the road direction specifically
    q is the target vehicle's location, the heading direction

    perpendicular: (p_fwd_x, p_fwd_y) -> (p_fwd_y, - p_fwd_x)
    p_x + p_fwd_y * hp = q_x + q_fwd_x * dq
    p_y + (- p_fwd_x) * hp = q_y + q_fwd_y * dq
    p_x * p_xy + p_fwd_y * hp * p_xy = q_x * p_xy + q_fwd_x * dq * p_xy
    (p_y + p_x * p_xy) = (q_y + q_x * p_xy) + (q_fwd_y + q_fwd_x * p_xy) * dq
    dq = (p_y - q_y + (p_x - q_x) * p_xy) / (q_fwd_y + q_fwd_x * p_xy)

    !!! (q_fwd_y + q_fwd_x * p_xy) = 0  
            <==> q_fwd_x * p_xy = - q_fwd_y 
            <==> p_xy * q_xy = -1, which is impossible

    p_xy = p_fwd_x / p_fwd_y
    p_x + p_fwd_x * dp = q_x + q_fwd_x * dq 
    p_y + p_fwd_y * dp = q_y + q_fwd_y * dq
    ==>
        p_y * p_xy + p_fwd_y * dp * p_xy = q_y * p_xy + q_fwd_y * dq * p_xy
        (p_x - p_y * p_xy) = (q_x - q_y * p_xy) + (q_fwd_x - q_fwd_y * p_xy) * dq 
        (q_fwd_x - q_fwd_y * p_xy) * dq = (p_x - p_y * p_xy) - (q_x - q_y * p_xy)
'''
def pseudo_vehicle_rotation(p, p_fwd, q, q_fwd, speed_q, min_dist = 3., max_dist = 300.):
    p_x, p_y, p_fwd_x, p_fwd_y = p.x, p.y, p_fwd.x, p_fwd.y
    q_x, q_y, q_fwd_x, q_fwd_y = q.x, q.y, q_fwd.x, q_fwd.y

    assert (pow(p_fwd_x,2)+pow(p_fwd_y,2) > 0) and (pow(q_fwd_x,2)+pow(q_fwd_y,2) > 0)
    norm_fwd_p = norm2d(p_fwd_x, p_fwd_y)
    norm_fwd_q = norm2d(q_fwd_x, q_fwd_y)
    p_fwd_x, p_fwd_y = p_fwd_x / norm_fwd_p, p_fwd_y / norm_fwd_p
    q_fwd_x, q_fwd_y = q_fwd_x / norm_fwd_q, q_fwd_y / norm_fwd_q
    # assume dp and dq are distance to the connection for p and q respectively
    '''
        p_xy = p_fwd_x / p_fwd_y
        p_yx = p_fwd_y / p_fwd_x
        
        p_x + p_fwd_x * dp = q_x + q_fwd_x * dq 
        p_y + p_fwd_y * dp = q_y + q_fwd_y * dq

        p_y * p_xy + p_fwd_y * dp * p_xy = q_y * p_xy + q_fwd_y * dq * p_xy
        (p_x - p_y * p_xy) = (q_x - q_y * p_xy) + (q_fwd_x - q_fwd_y * p_xy) * dq 
        (q_fwd_x - q_fwd_y * p_xy) * dq = (p_x - p_y * p_xy) - (q_x - q_y * p_xy)

    '''
    connection = False
    neighbor = False # whether the target neighbor is a "neighbor" on the current road
    opposite = False
    targe_v = False
    if p_fwd_x == 0:
        if q_fwd_x == 0: # two lines are parallel
            if abs(p_x - q_x) < min_dist: # two vehicle are driving on the same lane
                if p_fwd_y * q_fwd_y > 0: # two vehicles are driving on same lane and same direction
                    # in this case add target v as neighbor
                    neighbor = True
                elif (q_x-p_x)*p_fwd_x + (q_y-p_y)*p_fwd_y > 0: # two vehicles are driving on same lane and opposite direction !!!
                    # in this case add target v as emergency opposite hazard
                    neighbor = True
                    opposite = True
            #else: # two vehicles are driving in parallel and far enough
        else:
            dq = (p_x - q_x) / (q_fwd_x)
            dp = (q_y + q_fwd_y * dq - p_y) / p_fwd_y
            if dp <= max_dist: # two trajectories connect
                connection = (p_x, p_y + p_fwd_y * dp)
    elif p_fwd_y == 0:
        if q_fwd_y == 0: # two lines are parallel
            if abs(p_y - q_y) < min_dist: # two vehicle are driving on the same lane
                if p_fwd_x * q_fwd_x > 0: # two vehicles are driving on same lane and same direction
                    neighbor = True
                elif (q_x-p_x)*p_fwd_x + (q_y-p_y)*p_fwd_y > 0: # two vehicles are driving on same lane and opposite direction !!!
                    neighbor = True
                    opposite = True
            #else: # two vehicles are driving in parallel and far enough
        else:
            dq = (p_y - q_y) / q_fwd_y
            dp = (q_x + q_fwd_x * dq - p_x) / p_fwd_x
            if dp <= max_dist: # two trajectories connect
                connection = (p_x + p_fwd_x * dp, p_y)
    else:
        p_xy = p_fwd_x / p_fwd_y
        # two lines are in parallel
        if q_fwd_x * p_fwd_y == q_fwd_y * p_fwd_x: 
            # pq_dist = math.sqrt((p_x - q_x)**2 + (p_y - q_y)**2)
            # we find the perpendicular distance between two lines

            dq = (p_y - q_y + (p_x - q_x) * p_xy) / (q_fwd_y + q_fwd_x * p_xy)
            hp = (q_x + q_fwd_x * dq - p_x) / p_fwd_y # distance between two parallel lines
            # two vehicle are driving on the same lane
            if abs(dp) < min_dist: 
                # two vehicles are driving on same lane and same direction
                if p_fwd_x * q_fwd_x + p_fwd_y * q_fwd_y > 0: 
                    neighbor = True
                # two vehicles are driving on same lane and opposite direction !!!
                elif (q_x-p_x)*p_fwd_x + (q_y-p_y)*p_fwd_y > 0: 
                    neighbor = True
                    opposite = True

        else:
            dq = (p_x - q_x - (p_y - q_y)*p_xy) / (q_fwd_x - q_fwd_y * p_xy)
            dp = (q_x + q_fwd_x * dq - p_x) / p_fwd_x
            # then the potential collision place will be in front of the ego's trajectory
            if dp > 0 and dq > - min_dist: 
                connection = (p_x + p_fwd_x * dp, p_y + p_fwd_y * dp)
            # else: in this case the potential collision will happen behind the ego, which is okay.

    target_v = None
    if connection:
        pseudo_relative_distance = dp - dq
        pseudo_position = [p_x + p_fwd_x * pseudo_relative_distance, p_y + p_fwd_y * pseudo_relative_distance]
        pseudo_velocity = [p_fwd_x * speed_q, p_fwd_y * speed_q]
        target_v = pseudo_position + pseudo_velocity + [pseudo_relative_distance]
    elif neighbor:
        pseudo_relative_distance = (q_x-p_x)*p_fwd_x + (q_y-p_y)*p_fwd_y
        pseudo_position = [p_x + p_fwd_x * pseudo_relative_distance, p_y + p_fwd_y * pseudo_relative_distance]
        pseudo_velocity = [q_fwd_x * speed_q, q_fwd_y * speed_q]
        target_v = pseudo_position + pseudo_velocity + [pseudo_relative_distance]
    
    return connection, neighbor, opposite, target_v