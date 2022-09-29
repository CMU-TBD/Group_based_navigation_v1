import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from img_process import DrawGroupShape
from grouping import Grouping
from group_shape_generation import GroupShapeGeneration

import copy

# PRE-PROCESSING:
# Gets input options from user
def get_flags():
    x = input("Groups? (y/n): ")
    group_flag = False
    laser_flag = False
    pred_flag = False
    react_flag = False
    pred_method = None
    if(x=='y'):
        group_flag = True
        x = input("Laser Scans? (y/n): ")
        if(x=='y'):
            laser_flag = True
    x = input("Prediction? (y/n): ")
    if(x=='y'):
        pred_flag = True
        if (group_flag):
            pred_method = "auto"
        else:
            x = input("sgan/linear (1/2): ")
            x = int(x)
            if(x==1):
                pred_method = "sgan"
            elif(x==2):
                pred_method = "linear"
    else:
        pred_method = "nopred"
    x = input("Reactive Agents? (y/n): ")
    if (x == 'y'):
        react_flag = True
    return (group_flag, laser_flag, pred_flag, react_flag,
            pred_method)

def convert_dataset_id(dataset, dataset_idx):
    if (dataset == 'eth') and (dataset_idx == 0):
        return 'eth', 0
    if (dataset == 'eth') and (dataset_idx == 1):
        return 'hotel', 1
    if (dataset == 'ucy') and (dataset_idx == 0):
        return 'zara1', 2
    if (dataset == 'ucy') and (dataset_idx == 1):
        return 'zara2', 3
    if (dataset == 'ucy') and (dataset_idx == 2):
        return 'univ', 4
    raise Exception('impossible dataset names/idx')

# for creating a pedestrian dictionary
def ped_dicts(filename):
    file = open(filename,'r')
    lines = file.readlines()
    dictionary = {}
    # [frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y ]
    ped_ids = {}

    for line in lines:
        curr = line.split()
        frame_id = float(curr[0])
        x_pos = float(curr[2])
        y_pos = float(curr[4])
        pedestrian_id = float(curr[1])

        if frame_id in dictionary:
            dictionary[frame_id].append([x_pos, y_pos])
        else:
            dictionary[frame_id] = [[x_pos, y_pos]]

        if pedestrian_id in ped_ids:
            ped_ids[pedestrian_id].append([x_pos, y_pos])
        else:
            ped_ids[pedestrian_id] = [[x_pos, y_pos]]

    file.close()
    return dictionary, ped_ids


# VISUALIZATION:
def draw_individual_space(pos, vel, const):
    num_ped = np.shape(pos)[0]
    all_boundary = []
    for i in range(num_ped):
        boundary = GroupShapeGeneration.draw_social_shapes([pos[i]], [vel[i]], const)
        boundary.append(boundary[0])
        all_boundary.append(np.array(boundary))
    return all_boundary

def anim_frame(groups, pred, laser, occupied_points, obs_vel, ped_pos, scan_pts, start_config, robot_path, goal_config, has_ped, const, time_steps=None):
    curr_frame = []
    curr_frame.append(plt.scatter(start_config[0], start_config[1], c='g', s=10))
    curr_frame.append(plt.scatter(robot_path[:, 0], robot_path[:, 1], c='y', s=10))
    curr_frame.append(plt.scatter(goal_config[0], goal_config[1], c='m', s=10))
    if has_ped:
        if(groups):
            if laser:
                curr_frame.append(plt.scatter(scan_pts[:, 0], scan_pts[:, 1], c='y', s=10))

            if pred:
                group_boundaries = occupied_points[0]
            else:
                group_boundaries = occupied_points
            curr_frame.append(plt.scatter(group_boundaries[:, 0], 
                                          group_boundaries[:, 1], c='k', s=3))
            """
            if pred:
                for i in range(time_steps):
                    group_boundaries = occupied_points[1 + i]
                    curr_frame.append(plt.scatter(group_boundaries[:, 0], 
                                                  group_boundaries[:, 1], c='b', s=3))
            """
        else:
            if pred:
                boundary = draw_individual_space(occupied_points[:, 0, :],
                                                 obs_vel[:, 0, :], const)
                for bdry in boundary:
                    curr_frame += plt.plot(bdry[:, 0], bdry[:, 1], c='k', lw=3)
                """
                for i in range(time_steps):
                    boundary = draw_individual_space(occupied_points[:, i + 1, :],
                                                     obs_vel[:, i + 1, :], const)
                    for bdry in boundary:
                        curr_frame += plt.plot(bdry[:, 0], bdry[:, 1], c='b', lw=3)
                """
            else:
                boundary = draw_individual_space(occupied_points, obs_vel, const)
                boundary = np.array(boundary)
                curr_frame.append(plt.scatter(boundary[:, 0], boundary[:, 1], c='k', s=3))
            
        curr_frame.append(plt.scatter(ped_pos[:, 0], ped_pos[:, 1], c='r', s=10))
    return curr_frame

def visualize_peds(ped_pos,pred):
    if(pred):
        for tmp in ped_pos:
            curr = tmp[0]
            next = tmp[1:]
            plt.scatter(curr[0], curr[1], c='r')
            plt.scatter(next[:, 0], next[:, 1], c='y')
    else:
        plt.scatter(ped_pos[:,0],ped_pos[:,1],c='r')

# PROPAGATION:
def linear_propagate_peds_once(curr, vel, dt):
    curr[0]+=vel[0]*dt
    curr[1]+=vel[1]*dt
    return curr

def create_model_input(msg,frame,dt=0.1,length=8):
    curr_frame_people = msg.video_position_matrix[frame]
    curr_frame_velocity = msg.video_velocity_matrix[frame]
    num_ppl = np.shape(curr_frame_people)[0]
    ped_ids = []
    idx2id = {}
    for i in range(num_ppl):
        id = msg.video_pedidx_matrix[frame][i]
        ped_ids.append(id)
        idx2id[i] = id
    ans = np.zeros((num_ppl,length,2),dtype=np.float32)
    ans_vel = np.zeros((num_ppl,length,2),dtype=np.float32)
    start_frame = frame - length + 1
    for i in range(num_ppl):
        start_frame_i = msg.people_start_frame[idx2id[i]]
        # curr_start = max(start_frame,start_frame_i)
        if (start_frame_i <= start_frame):
            curr_start = start_frame
        else:
            curr_start = start_frame_i
        curr_vel_i = msg.people_velocity_complete[idx2id[i]][curr_start - start_frame_i]
        for j in range(curr_start,frame):
            ans[i][j-start_frame][0] = msg.people_coords_complete[idx2id[i]][j-start_frame_i][0]
            ans[i][j-start_frame][1] = msg.people_coords_complete[idx2id[i]][j-start_frame_i][1]
            ans_vel[i][j-start_frame][0]=msg.people_velocity_complete[idx2id[i]][j-start_frame_i][0]
            ans_vel[i][j-start_frame][1]=msg.people_velocity_complete[idx2id[i]][j-start_frame_i][1]
        ans[i][length - 1][0] = curr_frame_people[i][0]
        ans[i][length - 1][1] = curr_frame_people[i][1]
        ans_vel[i][length - 1][0] = curr_frame_velocity[i][0]
        ans_vel[i][length - 1][1] = curr_frame_velocity[i][1]
        for k in range(curr_start-1,start_frame-1,-1):
            ans[i][k-start_frame][0] = ans[i][k-start_frame+1][0] - curr_vel_i[0]*dt
            ans[i][k-start_frame][1] = ans[i][k-start_frame+1][1] - curr_vel_i[1]*dt
            ans_vel[i][k-start_frame][0] = curr_vel_i[0]
            ans_vel[i][k-start_frame][1] = curr_vel_i[1]

    return ans, ans_vel

def propagate_peds(msg, frame, dt, tf,one_step_propagater=linear_propagate_peds_once):
    pos_matrix = np.array(msg.video_position_matrix[frame])
    vel_matrix = np.array(msg.video_velocity_matrix[frame])
    num_ppl = np.shape(pos_matrix)[0]
    time_steps = int(tf/dt)
    ans = []
    for i in range(num_ppl):
        tmp = [pos_matrix[i]]
        vel = vel_matrix[i]
        curr = pos_matrix[i]
        for j in range(time_steps):
            next = one_step_propagater(np.copy(curr), vel, dt)
            curr = next
            tmp.append(np.copy(curr))
        ans.append(tmp)
    return np.array(ans)

def advance(current_pos,next_pos,v,dt):
    [dy,dx] =  [(next_pos[1] - current_pos[1]), (next_pos[0] - current_pos[0])]
    slope_angle = np.arctan2(dy,dx)
    vx = v*np.cos(slope_angle)
    vy = v*np.sin(slope_angle)
    dx = vx*dt
    dy = vy*dt
    current_pos = list(current_pos)
    current_pos[0]+=dx
    current_pos[1]+=dy
    return current_pos

# GENERATE PATH:
def generate_straight_path(start_config, goal_config, step_size):
    start_config = np.array(start_config)
    goal_config = np.array(goal_config)
    dist = np.linalg.norm(goal_config-start_config)
    if(dist==0):
        return start_config, 0
    steps = dist//step_size + 1
    waypoints = np.array([np.linspace(start_config[i], goal_config[i], int(steps)) for i in range(2)]).transpose()
    return waypoints, dist

#COLLISION CHECKERS:
def get_min_ped_dist(ped_poses, config):
    min_dist = 10000
    for pos in ped_poses:
        dist = ((pos[0]-config[0])**2 + (pos[1]-config[1])**2)**0.5
        if(dist<=min_dist):
            min_dist = dist
    return min_dist

def path_checker(start_config, goal_config, ped_pos, step_size, thresh, collision_checker, path_generator=generate_straight_path):
    path, length = path_generator(start_config,goal_config,step_size)
    if length==0:
        return 0, 0
    num_collisions = collision_checker(ped_pos,path,thresh)
    return num_collisions, length

def at_goal(start_config, end_config, final_thresh):
    diff = np.array(end_config) - np.array(start_config)
    dist = np.linalg.norm(diff, ord=2)
    if(dist<final_thresh):
        #print("Within threshold of goal, current position = ", start_config)
        return True
    else:
        return False


# FIND LEAST DISTANCE BETWEEN CONFIG AND POINTS
def find_least_dist(config, points):
    if len(points) == 0:
        return 1e+9, None
    diff = points - config
    dist = np.linalg.norm(diff, axis=1)
    return np.min(dist), np.argmin(dist)

# COMBINE CURRENT AND PREDICTED PEDESTRIAN POSITIONS
def combine_current_and_predicted_pos(curr_peds, predicted):
    num_ppl = np.shape(curr_peds)[0]
    time_steps = np.shape(predicted)[1]
    current = np.reshape(curr_peds,(num_ppl,1,2))
    ans = current.tolist()
    predicted = predicted.tolist()
    for i in range(num_ppl):
        for j in range(time_steps):
            ans[i].append(predicted[i][j])
    return np.array(ans)

# Special coordinate transform from pixel to metric
def inv_coordinate_transform(msg, vertices):
    if msg.dataset == 'ucy':
        tmp = copy.deepcopy(vertices[0,:])
        vertices[0,:] = vertices[1, :] - msg.frame_width / 2
        vertices[1,:] = msg.frame_height / 2 - tmp
    vertices = np.append(vertices, np.ones((1, np.shape(vertices)[1])), axis=0)
    vertices = np.matmul(msg.H, vertices)
    vertices = [vertices[0,:] / vertices[2,:], vertices[1,:] / vertices[2,:]]
    return vertices

# GROUP BASED OPERATIONS
def get_frame_groups(msg, positions, velocities, laser_flag, const):
    if (msg.dataset == "ucy") and (msg.flag == 2):
        pos = 1.5
        ori = 15
        vel = 0.5
        params = {'position_threshold': pos,
                  'orientation_threshold': ori / 180.0 * np.pi,
                  'velocity_threshold': vel,
                  'velocity_ignore_threshold': 0.5}
        group_ids = Grouping.grouping(positions, velocities, params = params)
    else:
        group_ids = Grouping.grouping(positions, velocities)
    group_vertices = GroupShapeGeneration.draw_all_social_spaces(group_ids, positions, velocities, 
                                                                 laser_flag, const)
    dgs = DrawGroupShape(msg)
    canvas = np.zeros((msg.frame_height, msg.frame_width, 3), dtype=np.uint8)
    for v in group_vertices:
        canvas = dgs.draw_group_shape(v, canvas, center=False, aug=False)
    img = canvas[:, :, 0] / 255
    return img

def frame_to_vertices(msg, frame):
    laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    frame = signal.convolve2d(frame, laplacian, mode='same')
    frame = np.clip(np.abs(frame), 0, 1)
    vertices = np.array(np.nonzero(frame))
    vertices = inv_coordinate_transform(msg, vertices)
    """
    for i in range(1, msg.frame_height - 1):
        for j in range(1, msg.frame_width - 1):
            if (frame[i, j] == 1) and (not (
                (frame[i-1, j] == 1) and
                (frame[i, j-1] == 1) and
                (frame[i+1, j] == 1) and
                (frame[i, j+1] == 1))):
                coord = np.array([[i], [j], [1.0]], dtype=np.float32)
                coord = np.matmul(msg.H, coord)
                vertices.append([coord[0][0] / coord[2][0], coord[1][0] / coord[2][0]])
    """
    return np.transpose(np.array(vertices), (1,0))

# Simulator Helper Functions
def get_pref_velocity(pos, goal, spd_limit):
    vel = np.array(goal) - np.array(pos)
    dist = np.linalg.norm(vel)
    if not (dist == 0):
        vel = vel / np.linalg.norm(vel) * spd_limit
    else:
        vel = [0, 0]
    return (vel[0], vel[1])

# Metrics Related
def estimate_path_length(path):
    path = np.array(path)
    rel_path = path[1:, :] - path[:-1, :]
    return np.sum(np.linalg.norm(rel_path, axis=1))

def estimate_path_irregularity(path):
    path = np.array(path)
    rel_path = path[1:, :] - path[:-1, :]
    rel_ang = np.arctan2(rel_path[:, 1], rel_path[:, 0])
    if len(rel_ang) < 2:
        return 0
    else:
        change_ang = np.abs(((rel_ang[1:] - rel_ang[:-1]) + np.pi) % (2 * np.pi) - np.pi)
        return np.mean(change_ang)

# Simulate Laser Scans
def ped_to_scans(robo_pos, ped_pos, ped_vel):
    num_ped = len(ped_pos)

    # SICK LMS511 2D Lidar
    ang_res = 0.25 * np.pi / 180
    det_range = 40 #Basically Inf
    noise_limit = 0.05
    ped_radius = 0.5
    r_sq = ped_radius ** 2

    laser_pos = []
    laser_vel = []
    ang = 0
    while ang < (2 * np.pi):
        if not (ang % (np.pi / 2) == 0):
            min_dist = det_range
            laser_x = None
            laser_y = None 
            min_idx = None
            for i in range(num_ped):
                a = ped_pos[i][0] - robo_pos[0]
                b = ped_pos[i][1] - robo_pos[1]
                A = 1 + np.tan(ang) ** 2
                B = -2 * (a + b * np.tan(ang))
                C = a ** 2 + b ** 2 - r_sq
                check_root = round(B ** 2 - 4 * A * C, 12)
                if check_root >= 0:
                    x1 = (-B - np.sqrt(check_root)) / (2 * A)
                    y1 = x1 * np.tan(ang)
                    x2 = (-B + np.sqrt(check_root)) / (2 * A)
                    y2 = x2 * np.tan(ang)
                    mag1 = np.sqrt(x1 ** 2 + y1 ** 2)
                    mag2 = np.sqrt(x2 ** 2 + y2 ** 2)
                    if mag1 < mag2:
                        append_x = x1
                        append_y = y1
                        dist = mag1
                    else:
                        append_x = x2
                        append_y = y2
                        dist = mag2
                    noise = np.random.uniform(-noise_limit, noise_limit)
                    append_x += noise * np.cos(ang)
                    append_y += noise * np.sin(ang)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                        laser_x = append_x + robo_pos[0]
                        laser_y = append_y + robo_pos[1]
            if not (laser_x == None):
                laser_pos.append([laser_x, laser_y])
                laser_vel.append([ped_vel[min_idx][0], ped_vel[min_idx][1]]) 

        ang += ang_res
    return np.array(laser_pos), np.array(laser_vel)

def ped_series_to_scans(robo_pos, ped_pos_series, ped_vel_series):
    # output time X pts X coord
    time_steps = np.shape(ped_pos_series)[1]
    pos_series = []
    vel_series = []
    for i in range(time_steps):
        pos_scan, vel_scan = ped_to_scans(robo_pos, 
                                          ped_pos_series[:, i, :], 
                                          ped_vel_series[:, i, :])
        pos_series.append(pos_scan)
        vel_series.append(vel_scan)
    return pos_series, vel_series
