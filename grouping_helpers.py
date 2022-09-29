import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import general_helpers as gh

def group_collisions(groups_vertices,cur_poses,thresh):
    num_collisions = 0
    for group_vertices in groups_vertices:
        path_obj = Path(group_vertices)
        occ_arr = path_obj.contains_points(cur_poses, radius=thresh)
        occ_arr = 1*np.array(occ_arr)
        num_collisions+=np.sum(occ_arr)
    return num_collisions

def interpolate_between_group_vertices(vertices, step_size=0.1):
    vertices.append(vertices[0])
    ans = []
    for i in range(np.shape(vertices)[0] - 1):
        ans.append(gh.generate_straight_path(vertices[i],vertices[i+1],step_size)[0])
    ret = np.concatenate(ans,axis=0)
    return ret

def get_points_in_groups(group_vertices, grid_points):
    path_obj = Path(group_vertices)
    occ_grid = path_obj.contains_points(grid_points)
    ans = []
    for i in range(np.shape(occ_grid)[0]):
        if(occ_grid[i] == True):
            ans.append(grid_points[i])
    return ans

def make_dict_pedidx2arridx(msg_obj,frame):
    pedidx_to_arridx = {}
    ped_idx = msg_obj.video_pedidx_matrix[frame]
    for i in range(np.shape(ped_idx)[0]):
        pedidx_to_arridx[ped_idx[i]] = i
    return pedidx_to_arridx

def advance_group_once(vertices, dt, avg_vel):
    for i in range(np.shape(vertices)[0]):
        vertices[i][0] += avg_vel[0]*dt
        vertices[i][1] += avg_vel[1]*dt
    return vertices

def advance_group(tf, dt, avg_vel, group_vertices):
    time_steps = int(tf/dt)
    ans = [np.copy(group_vertices)]
    curr = np.copy(group_vertices)
    for i in range(time_steps):
        next = advance_group_once(curr, dt, avg_vel)
        ans.append(np.copy(next))
        curr = next
    return ans

def advance_groups(msg, frame, groups_vertices, groups_members, tf, dt):
    pedidx_to_arridx = make_dict_pedidx2arridx(msg, frame)
    ped_vels = msg.video_velocity_matrix[frame]
    ans = []
    for group_vertices, group_members in zip(groups_vertices, groups_members):
        avg_vel = [0, 0]
        for member in group_members:
            avg_vel[0] += ped_vels[pedidx_to_arridx[member]][0]
            avg_vel[1] += ped_vels[pedidx_to_arridx[member]][1]
        avg_vel[0] /= np.shape(group_members)[0]
        avg_vel[1] /= np.shape(group_members)[0]
        curr = advance_group(tf, dt, avg_vel, group_vertices)
        ans.append(curr)
    return ans

def get_groups(gs_gen, group_ids, frame, edge_step_size):
    group_ids_unique = list(set(group_ids))
    groups_vertices=[]
    groups_members=[]
    for group_id in group_ids_unique:
        raw_group_vertices, group_members = gs_gen.generate_group_shape(frame, group_id, None)
        interpolated_group_vertices = interpolate_between_group_vertices(raw_group_vertices, step_size=edge_step_size)
        groups_vertices.append(interpolated_group_vertices)
        groups_members.append(group_members)
    return groups_vertices, groups_members

def visualize_groups(groups_vertices, prediction=False):
    if(prediction):
        for groups_vertices_current in groups_vertices:
            plt.scatter(group_vertices_current[:, 0], group_vertices_current[:, 1], s=1, c='k')
    else:
        plt.scatter(groups_vertices[:, 0], groups_vertices[:, 1], s=1, c='k')
