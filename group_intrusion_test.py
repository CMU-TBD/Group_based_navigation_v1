import numpy as np
from message import Message
from data_loader import DataLoader
import general_helpers as gh
import MPC_helpers as mpc_h
import matplotlib.pyplot as plt

from copy import deepcopy

import rvo2
import pysocialforce as psf

import pickle

def sim_step(robo_curr, robo_goal, dt, robo_max_v, t_horizon, ped_pos, ped_vel, ped_goals):

    #RVOSimulator (float timeStep, float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed)
    ped_max_spd = 1.75
    sim = rvo2.PyRVOSimulator(dt, 2.5, 10, t_horizon, 2, 0.5, ped_max_spd)

    robot = sim.addAgent((robo_curr[0], robo_curr[1]))
    sim.setAgentMaxSpeed(robot, robo_max_v)
    sim.setAgentPrefVelocity(robot, gh.get_pref_velocity(robo_curr, robo_goal, robo_max_v))

    num_ped = len(ped_pos)
    ped_list = []
    for i in range(num_ped):
        ped = sim.addAgent((ped_pos[i][0], ped_pos[i][1]))
        ped_spd = np.linalg.norm(np.array(ped_vel[i]))
        sim.setAgentVelocity(ped, (ped_vel[i][0], ped_vel[i][1]))
        sim.setAgentMaxSpeed(ped, max(ped_spd, ped_max_spd))
        sim.setAgentPrefVelocity(ped, gh.get_pref_velocity(ped_pos[i], ped_goals[i], ped_spd))
        ped_list.append(ped)

    sim.doStep()

    ped_pos_new = np.array([sim.getAgentPosition(ped) for ped in ped_list])
    ped_vel_new = np.array([sim.getAgentVelocity(ped) for ped in ped_list])

    return ped_pos_new, ped_vel_new

def evaluate(dataset, dataset_idx, init_start_config, init_end_config, start_frame, time_limit, 
             traj_playback, react_flag, laser_flag):

    # rollout params
    dt = 0.1
    time_steps = 8
    tf_horizon = dt*time_steps
    v = 1.75

    collision_radius = 0.5
    final_thresh = 1.0
    const_init = 0.35 # group shape size
    if (dataset == "ucy") and (dataset_idx == 2):
        const_init = 0.25
    const_min = 0.05

    raw_msg = Message()
    data = DataLoader(dataset, dataset_idx, 1 / dt)
    raw_msg = data.update_message(raw_msg)

    time_limit = time_limit + start_frame 

    msg = deepcopy(raw_msg) 
    const = const_init

    start_config = init_start_config
    end_config = init_end_config

    if react_flag:
        ped_pos_current = []
        ped_vel_current = []
        ped_idxes_current = []
        ped_idxes = []
        ped_goals_current = []

    group_intrusions = []
    for frame in range(start_frame, time_limit):
        if frame >= msg.total_num_frames:
            ped_pos_current = []
            ped_vel_current = []
        elif react_flag:
            # remove any pedestrian who reaches goal
            ped_pos_new = []
            ped_vel_new = []
            ped_idxes_new = []
            ped_goals_new = []
            for i in range(len(ped_pos_current)):
                if (not gh.at_goal(ped_pos_current[i], ped_goals_current[i], final_thresh)):
                    ped_pos_new.append(ped_pos_current[i])
                    ped_vel_new.append(ped_vel_current[i])
                    ped_idxes_new.append(ped_idxes_current[i])
                    ped_goals_new.append(ped_goals_current[i])

            # check for new pedestrians
            new_idxes_current = msg.video_pedidx_matrix[frame]
            for i, idx in enumerate(new_idxes_current):
                if (not (idx in ped_idxes)):
                    ped_idxes.append(idx)
                    ped_pos_new.append(msg.video_position_matrix[frame][i])
                    ped_vel_new.append(msg.video_velocity_matrix[frame][i])
                    ped_idxes_new.append(idx)
                    ped_goals_new.append(msg.people_coords_complete[idx][-1])

            ped_pos_current = np.array(ped_pos_new,dtype=np.float32)
            ped_vel_current = np.array(ped_vel_new,dtype=np.float32)
            ped_idxes_current = ped_idxes_new
            ped_goals_current = np.array(ped_goals_new,dtype=np.float32)

            # Modify msg
            msg.video_position_matrix[frame] = ped_pos_current
            msg.video_velocity_matrix[frame] = ped_vel_current
            msg.video_pedidx_matrix[frame] = ped_idxes_current
            for i, idx in enumerate(ped_idxes_current):
                st_frame = msg.people_start_frame[idx]
                end_frame = msg.people_end_frame[idx]
                if (frame > end_frame):
                    msg.people_coords_complete[idx].append(ped_pos_current[i])
                    msg.people_velocity_complete[idx].append(ped_vel_current[i])
                    msg.people_end_frame[idx] = frame
                else:
                    msg.people_coords_complete[idx][frame - st_frame] = ped_pos_current[i]
                    msg.people_velocity_complete[idx][frame - st_frame] = ped_vel_current[i]
        else:
            ped_pos_current = np.array(msg.video_position_matrix[frame],dtype=np.float32)
            ped_vel_current = np.array(msg.video_velocity_matrix[frame],dtype=np.float32)

        num_ppl = len(ped_pos_current)
        if num_ppl == 0:
            group_intrusions.append(0)
        else:
            groups_frame_current = gh.get_frame_groups(msg,
                                                       ped_pos_current,
                                                       ped_vel_current,
                                                       laser_flag,
                                                       const)
            if mpc_h.check_inside_groups(msg, start_config, groups_frame_current):
                group_intrusions.append(1)
            else:
                group_intrusions.append(0)

        if not (frame == (time_limit - 1)):
            start_config = traj_playback[frame - start_frame + 1]

        if react_flag:
            ped_pos_current, ped_vel_current = sim_step(start_config, end_config, 
                                                        dt, v, tf_horizon,
                                                        ped_pos_current, ped_vel_current, 
                                                        ped_goals_current)

    return group_intrusions



if __name__ == "__main__":

    experiments = [(False, False, "ped_nopred.txt"), 
                   (False, False, "ped_linear.txt"),
                   (False, False, "ped_sgan.txt"),
                   (False, False, "group_nopred.txt"),
                   (False, False, "group_auto.txt"), 
                   (False, True, "group_auto_laser.txt"), 
                   (True, False, "ped_nopred_react.txt"),
                   (True, False, "ped_linear_react.txt"),
                   (True, False, "ped_sgan_react.txt"),
                   (True, False, "group_nopred_react.txt"),
                   (True, False, "group_auto_react.txt"),
                   (True, True, "group_auto_laser_react.txt"), 
                  ]


    for exp in experiments:
        print(exp[2])
        is_react, laser_flag, exp_name = exp
        pfile_name = "results/" + exp_name
        output_fname = "group_intrusion_rst/" + exp_name
        with open(pfile_name, "rb") as fp:
            cases = pickle.load(fp)
        num_cases = len(cases)

        all_results = []
        for i, case in enumerate(cases):
            print([i, num_cases], end='\r')

            if not (case[1][0] == 0):
                dataset = case[0][0]
                dataset_idx = case[0][1]
                init_start_config = case[0][2]
                init_end_config = case[0][3]
                start_frame = case[0][4]

                traj_playback = case[1][5][0]
                time_limit = len(traj_playback)

                metrics = evaluate(dataset, 
                                   dataset_idx, 
                                   init_start_config, 
                                   init_end_config, 
                                   start_frame, 
                                   time_limit,
                                   traj_playback,
                                   is_react,
                                   laser_flag)
                all_results.append([case[0], metrics])

            with open(output_fname, "wb") as f:
                pickle.dump(all_results, f)
