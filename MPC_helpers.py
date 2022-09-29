import numpy as np
import matplotlib.pyplot as plt
from img_process import DrawGroupShape
from grouping_helpers import visualize_groups
from general_helpers import find_least_dist, inv_coordinate_transform
from group_shape_generation import GroupShapeGeneration as GSG

interval = 1

# GENERATE ROLLOUTS
def generate_rollouts(start_config, time_horizon, num_rollouts, vel):
    len_horizon = np.shape(time_horizon)[0]
    angles = np.linspace(np.radians(-180), np.radians(180), num_rollouts, endpoint=True)
    rollouts = np.zeros((num_rollouts * 9, np.shape(time_horizon)[0], 2), dtype=np.float32)
    rollouts[:, 0] = start_config
    dt = time_horizon[1] - time_horizon[0]
    R1 = vel * dt * (len_horizon - 1) / (np.pi / 2)

    for i in range(1, len_horizon):
        rollouts[:num_rollouts, i, 0] = start_config[0] + (vel * dt * i * np.sin(angles[:]))
        rollouts[:num_rollouts, i, 1] = start_config[1] + (vel * dt * i * np.cos(angles[:]))
        rollouts[num_rollouts:(2*num_rollouts), i, 0] = \
            start_config[0] + (2/3 * vel * dt * i * np.sin(angles[:]))
        rollouts[num_rollouts:(2*num_rollouts), i, 1] = \
            start_config[1] + (2/3 * vel * dt * i * np.cos(angles[:]))
        rollouts[(2*num_rollouts):(3*num_rollouts), i, 0] = \
            start_config[0] + (1/3 * vel * dt * i * np.sin(angles[:]))
        rollouts[(2*num_rollouts):(3*num_rollouts), i, 1] = \
            start_config[1] + (1/3 * vel * dt * i * np.cos(angles[:]))

        ang = (vel * dt * i) / (2 * R1)
        L = 2 * R1 * np.sin(ang)
        rollouts[(3*num_rollouts):(4*num_rollouts), i, 0] = \
            start_config[0] + (L * np.sin(angles[:] + ang))
        rollouts[(3*num_rollouts):(4*num_rollouts), i, 1] = \
            start_config[1] + (L * np.cos(angles[:] + ang))
        ang = (2/3 * vel * dt * i) / (2 * R1)
        L = 2 * R1 * np.sin(ang)
        rollouts[(4*num_rollouts):(5*num_rollouts), i, 0] = \
            start_config[0] + (L * np.sin(angles[:] + ang))
        rollouts[(4*num_rollouts):(5*num_rollouts), i, 1] = \
            start_config[1] + (L * np.cos(angles[:] + ang))
        ang = (1/3 * vel * dt * i) / (2 * R1)
        L = 2 * R1 * np.sin(ang)
        rollouts[(5*num_rollouts):(6*num_rollouts), i, 0] = \
            start_config[0] + (L * np.sin(angles[:] + ang))
        rollouts[(5*num_rollouts):(6*num_rollouts), i, 1] = \
            start_config[1] + (L * np.cos(angles[:] + ang))

        ang = (vel * dt * i) / (2 * R1)
        L = 2 * R1 * np.sin(ang)
        rollouts[(6*num_rollouts):(7*num_rollouts), i, 0] = \
            start_config[0] + (L * np.sin(angles[:] - ang))
        rollouts[(6*num_rollouts):(7*num_rollouts), i, 1] = \
            start_config[1] + (L * np.cos(angles[:] - ang))
        ang = (2/3 * vel * dt * i) / (2 * R1)
        L = 2 * R1 * np.sin(ang)
        rollouts[(7*num_rollouts):(8*num_rollouts), i, 0] = \
            start_config[0] + (L * np.sin(angles[:] - ang))
        rollouts[(7*num_rollouts):(8*num_rollouts), i, 1] = \
            start_config[1] + (L * np.cos(angles[:] - ang))
        ang = (1/3 * vel * dt * i) / (2 * R1)
        L = 2 * R1 * np.sin(ang)
        rollouts[(8*num_rollouts):, i, 0] = \
            start_config[0] + (L * np.sin(angles[:] - ang))
        rollouts[(8*num_rollouts):, i, 1] = \
            start_config[1] + (L * np.cos(angles[:] - ang))


    return rollouts

def generate_rollouts_old(start_config, time_horizon, num_rollouts, vel):
    angles = np.linspace(np.radians(-180), np.radians(180), num_rollouts, endpoint=True)
    rollouts = np.zeros((num_rollouts, np.shape(time_horizon)[0], 2), dtype=np.float32)
    rollouts[:, 0] = start_config
    dt = time_horizon[1] - time_horizon[0]
    for i in range(1, np.shape(time_horizon)[0]):
        rollouts[:, i, 0] = rollouts[:, i - 1, 0] + (vel * dt * np.sin(angles[:]))
        rollouts[:, i, 1] = rollouts[:, i - 1, 1] + (vel * dt * np.cos(angles[:]))
    return rollouts


# FINDING DISTANCE OF ROLLOUTS TO GROUPS

def check_inside_groups(msg, rollout_pt, group_frame):
    dgs = DrawGroupShape(msg)
    rollout_pt_pix = dgs.coordinate_transform(rollout_pt)
    y, x = rollout_pt_pix
    if ((x >= 0) and (x < msg.frame_height) and
        (y >= 0) and (y < msg.frame_width) and
        (group_frame[x, y] > 0)):
        return True
    else:
        return False

def rollout_groups(msg, rollout, group_frames, groups_boundaries):
    time_steps = np.shape(rollout)[0]
    dists = np.ones(time_steps)*(1e+9)
    hit_idx = time_steps
    for i in range(time_steps):
        if check_inside_groups(msg, rollout[i], group_frames[0]):
            hit_idx = min(hit_idx, i)
        dists[i], _ = find_least_dist(rollout[i], groups_boundaries)
    return dists, hit_idx

def rollout_groups_pred(msg, rollout, group_frames, groups_boundaries):
    time_steps = np.shape(rollout)[0]
    dists = np.ones(time_steps)*(1e+9)
    hit_idx = time_steps
    for i in range(time_steps):
        if check_inside_groups(msg, rollout[i], group_frames[i]):
            hit_idx = min(hit_idx, i)
        dists[i], _ = find_least_dist(rollout[i], groups_boundaries[i])
    return dists, hit_idx

# FINDING DISTANCE OF ROLLOUT TO PEDESTRIANS

def check_inside_PS(pt_pos, ped_pos, ped_vel, const):
    num_ped = len(ped_pos)
    for i in range(num_ped):
        rel_pos = pt_pos - ped_pos[i]
        dist = np.linalg.norm(rel_pos)
        ori = np.arctan2(ped_vel[i][1], ped_vel[i][0])
        rel_ang = np.arctan2(rel_pos[1], rel_pos[0]) - ori
        boundary_dist = GSG.boundary_dist(ped_vel[i], rel_ang, const)
        if dist <= boundary_dist:
            return True

    return False

def rollout_peds_pred(rollout, ped_pos, ped_vel, space_const):
    time_steps = np.shape(rollout)[0]
    dists = np.ones(time_steps) * (1e+9)
    hit_idx = time_steps
    assert(time_steps <= np.shape(ped_pos)[1])
    for i in range(time_steps):
        ped_pos_curr = ped_pos[:, i, :]
        ped_vel_curr = ped_vel[:, i, :]
        dists[i], idx = find_least_dist(rollout[i], ped_pos_curr)
        if not (idx == None):
            min_ped_vel = ped_vel_curr[idx]
            min_ped_ori = np.arctan2(min_ped_vel[1], min_ped_vel[0])
            rel_pos = rollout[i] - ped_pos_curr[idx]
            rel_ang = np.arctan2(rel_pos[1], rel_pos[0]) - min_ped_ori
            boundary_dist = GSG.boundary_dist(min_ped_vel, rel_ang, space_const)
            if dists[i] <= boundary_dist:
                hit_idx = min(hit_idx, i)

    return dists, hit_idx

def rollout_peds(rollout, ped_pos, ped_vel, space_const):
    time_steps = np.shape(rollout)[0]
    dists = np.ones(len(rollout), dtype=np.float32) * (1e+9)
    hit_idx = time_steps
    for i in range(time_steps):
        dists[i], idx = find_least_dist(rollout[i], ped_pos)
        if not (idx == None):
            min_ped_vel = ped_vel[idx]
            min_ped_ori = np.arctan2(min_ped_vel[1], min_ped_vel[0])
            rel_pos = rollout[i] - ped_pos[idx]
            rel_ang = np.arctan2(rel_pos[1], rel_pos[0]) - min_ped_ori
            boundary_dist = GSG.boundary_dist(min_ped_vel, rel_ang, space_const)
            if dists[i] <= boundary_dist:
                hit_idx = min(hit_idx, i)
    return dists, hit_idx

# EVALUATING COST FUNCTION

def min_dist_cost_func(dists, hit_idx):
    cost = 0
    gamma = 0.9
    discount = 1
    for i, d in enumerate(dists):
        if i >= hit_idx:
            d = -d
        #cost += np.exp(-d)
        cost += np.exp(-d) * discount
        discount *= gamma
    return cost

def evaluate_rollouts(msg, robo_pos, rollouts, occupied_points, ped_vels, group_frames, coll_thresh, goal_config, mpc_w, ps_const, groups=False, pred=False, coll_flag=False, has_obstacles=True):
    # group_frames only used in groups for inside detection    

    num_rollouts = np.shape(rollouts)[0]
    costs = np.zeros(num_rollouts, dtype=np.float32)
    min_dist_weight = mpc_w
    end_dist_weight = 1 - min_dist_weight
    for i in range(num_rollouts):
        if has_obstacles:
            if(groups):
                if(pred):
                    min_dists, hit_idx = rollout_groups_pred(msg, rollouts[i], 
                                                             group_frames, occupied_points)
                else:
                    min_dists, hit_idx = rollout_groups(msg, rollouts[i], 
                                                        group_frames, occupied_points)
            else:
                if(pred):
                    min_dists, hit_idx = rollout_peds_pred(rollouts[i], occupied_points, 
                                                           ped_vels, ps_const)
                else:
                    min_dists, hit_idx = rollout_peds(rollouts[i], occupied_points, 
                                                      ped_vels, ps_const)
            min_dist_cost = min_dist_cost_func(min_dists, hit_idx)
        else:
            min_dist_cost = 0
            hit_idx = np.shape(rollouts)[1]
        if hit_idx == 0:
            end_dist_cost = np.linalg.norm(goal_config - robo_pos)
        else:
            end_dist_cost = np.linalg.norm(goal_config - rollouts[i, hit_idx - 1])
        costs[i] = min_dist_weight * min_dist_cost + end_dist_weight * end_dist_cost
    return costs

#VISUALIZATION:

def visualize_rollouts(rollouts, lowest_cost_ind):
    for i in range (len(rollouts)):
        if(i!=lowest_cost_ind):
            plt.plot(rollouts[i][:,0], rollouts[i][:,1], 'b')
        else:
            plt.plot(rollouts[i][:,0], rollouts[i][:,1], 'y')

def visualize_frame(rollouts, lowest_cost_ind, ped_pos, group_boundaries, start_config, end_config, pred=False, has_ped=True):
    visualize_rollouts(rollouts, lowest_cost_ind)
    if has_ped:
        if len(np.shape(group_boundaries)) != 0:
            visualize_groups(group_boundaries, pred)
            plt.scatter(ped_pos[:, 0], ped_pos[:, 1], c='r')
        else:
            if (not pred):
                plt.scatter(ped_pos[:, 0], ped_pos[:, 1], c='r')
            else:
                for tmp in ped_pos:
                    curr = tmp[0]
                    next = tmp[1:]
                    plt.scatter(curr[0],curr[1],c='r')
                    plt.scatter(next[:,0], next[:,1],c='y')

    plt.plot(start_config[0], start_config[1], 'go')
    plt.plot(end_config[0], end_config[1], 'mo')
    plt.show()

