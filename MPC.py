import numpy as np
from message import Message
from data_loader import DataLoader
from group_shape_prediction import GroupShapePrediction
import general_helpers as gh
import grouping_helpers as gp_h
import MPC_helpers as mpc_h
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from time import time
from copy import deepcopy

import sys
sys.path.append('sgan')
from scripts.inference import SGANInference as sgan

import rvo2
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

def run_trial(dataset, dataset_idx, init_start_config, init_end_config, start_frame, time_limit, 
              args, mpc_w):

    group_flag, laser_flag, pred_flag, react_flag, pred_method = args

    # Meta debug flags
    show_plot_flag = False
    include_when_fail = False

    # rollout params
    num_rollouts = 12
    time_steps = 8
    dt = 0.1
    tf_horizon = dt*time_steps
    v = 1.75
    time_horizon = np.arange(0, tf_horizon + dt, dt)

    collision_radius = 0.5
    final_thresh = 1.0
    const_init = 0.35 # group shape size
    if (dataset == "ucy") and (dataset_idx == 2):
        const_init = 0.25
    const_min = 0.05

    num_eval = 1

    if pred_method == "sgan":
        dt_name, _ = gh.convert_dataset_id(dataset, dataset_idx)
        path = "sgan/models/sgan-models/" + dt_name  + "_" + str(time_steps) + "_model.pt"
        model = sgan(path)

    raw_msg = Message()
    data = DataLoader(dataset, dataset_idx, 1 / dt)
    raw_msg = data.update_message(raw_msg)

    time_limit = time_limit + start_frame 

    # Metrics related
    end_status = []
    min_ped_distances = []
    time_duration = []    
    path_lengths = []
    path_irregularities = []
    all_robo_paths = []
    for test_num in range(num_eval):
        #print('Test number #{}'.format(test_num+1))
        time_out = True
        min_ped_dist = 10000
        msg = deepcopy(raw_msg) 
        const = const_init

        if (group_flag) and (pred_flag) and (pred_method == "auto"):
            _, dt_num = gh.convert_dataset_id(dataset, dataset_idx)
            path = "checkpoints/model_fpsfix_" + str(dt_num) + ".pth"
            gsp = GroupShapePrediction(msg, path)

        start_config = init_start_config
        end_config = init_end_config
        robot_path = [start_config]

        if react_flag:
            ped_pos_current = []
            ped_vel_current = []
            ped_idxes_current = []
            ped_idxes = []
            ped_goals_current = []

        for frame in range(start_frame, time_limit):
            print([frame, time_limit], end='\r')
            if (gh.at_goal(start_config, end_config, final_thresh) == True):
                time_out = False
                end_status.append(0) # 0 = success
                min_ped_distances.append(min_ped_dist)
                time_duration.append((frame - start_frame) * dt)
                path_lengths.append(gh.estimate_path_length(robot_path))
                path_irregularities.append(gh.estimate_path_irregularity(robot_path))
                all_robo_paths.append(robot_path)
                break

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
            rollouts = mpc_h.generate_rollouts(start_config, time_horizon, num_rollouts, v)
            group_frames_collection = None

            if(num_ppl == 0):
                has_ped = False
                obstacle_points = []
                obstacle_vels = []
                ped_pos = []
                ped_pos_current = []
                ped_pos_alt = []
                ped_vel_alt = []
                groups_vertices = []
                groups_vertices_current = []
            else:
                min_ped_dist_curr = gh.get_min_ped_dist(ped_pos_current, start_config)
                if (min_ped_dist_curr < min_ped_dist):
                    min_ped_dist = min_ped_dist_curr
                if (min_ped_dist < collision_radius):
                    # print("Collision took place!!!")
                    time_out = False
                    end_status.append(1) # 1 = collision
                    if include_when_fail:
                        min_ped_distances.append(min_ped_dist)
                        time_duration.append((frame - start_frame) * dt)
                        path_lengths.append(gh.estimate_path_length(robot_path))
                        path_irregularities.append(gh.estimate_path_irregularity(robot_path))
                    all_robo_paths.append(robot_path)
                    break

                ped_pos_alt = ped_pos_current
                ped_vel_alt = ped_vel_current
                if group_flag:
                    # Shrink group space if already inside group space
                    const = const_init
                    if laser_flag:
                        ped_pos_alt, ped_vel_alt = gh.ped_to_scans(start_config, 
                                                                   ped_pos_current,
                                                                   ped_vel_current)
                    groups_frame_current = gh.get_frame_groups(msg, 
                                                               ped_pos_alt, 
                                                               ped_vel_alt, 
                                                               laser_flag,
                                                               const)
                    while mpc_h.check_inside_groups(msg, start_config, groups_frame_current):
                        const = max(const - 0.1, const_min)
                        groups_frame_current = gh.get_frame_groups(msg, 
                                                                   ped_pos_alt, 
                                                                   ped_vel_alt,
                                                                   laser_flag, 
                                                                   const)
                        if const == const_min:
                            break
                    groups_vertices_current = gh.frame_to_vertices(msg, groups_frame_current)
                    group_frames_collection = [groups_frame_current]
                    if(pred_flag):
                        groups_vertices = [groups_vertices_current]
                        if(pred_method=="auto"):
                            inputs_pos, inputs_vel = gh.create_model_input(msg,frame,dt=dt,
                                                                           length=time_steps)
                            if (laser_flag):
                                inputs_pos, inputs_vel = gh.ped_series_to_scans(start_config,
                                                                                inputs_pos,
                                                                                inputs_vel)
                                groups_frames_predicted = gsp.laser_predict(inputs_pos, 
                                                                            inputs_vel, 
                                                                            const)
                            else:
                                groups_frames_predicted = gsp.predict(inputs_pos, 
                                                                      inputs_vel, 
                                                                      const)
                            for pred_frame in groups_frames_predicted:
                                group_frames_collection.append(pred_frame)
                                groups_vertices.append(gh.frame_to_vertices(msg, pred_frame))
                        else:
                            raise Exception('Undefined prediction method!')
                    else:            
                        groups_vertices = groups_vertices_current
                    obstacle_points = groups_vertices
                    obstacle_vels = []
                else:
                    # Shrink personal space if already inside personal space
                    const = const_init
                    while mpc_h.check_inside_PS(start_config, ped_pos_current, ped_vel_current, 
                                                const):
                        const = max(const - 0.1, const_min)
                        if const == const_min:
                            break

                    groups_vertices=None
                    groups_vertices_current=None
                    if(pred_flag):
                        if(pred_method=="sgan"): # PREDICTION METHOD
                            input_to_sgan, _ = gh.create_model_input(msg,frame,dt=dt,
                                                                     length=time_steps)
                            ped_pos_predicted = model.evaluate(input_to_sgan)
                            ped_pos = gh.combine_current_and_predicted_pos(ped_pos_current,
                                                                           ped_pos_predicted)

                        elif(pred_method=="linear"):
                            ped_pos = gh.propagate_peds(msg,frame,dt,tf_horizon)

                        obstacle_points = ped_pos
                        ped_vel_predicted = (ped_pos[:, 1:, :] - ped_pos[:, :-1, :]) / dt
                        obstacle_vels = np.concatenate((np.expand_dims(ped_vel_current, 1), 
                                                        ped_vel_predicted), axis=1)
                    else:
                        ped_pos = ped_pos_current
                        obstacle_points = ped_pos_current
                        obstacle_vels = ped_vel_current
                has_ped = True

            costs = mpc_h.evaluate_rollouts(msg, 
                                            start_config, 
                                            rollouts, 
                                            obstacle_points, 
                                            obstacle_vels,
                                            group_frames_collection, 
                                            collision_radius, 
                                            end_config,
                                            mpc_w, 
                                            const,
                                            groups=group_flag, 
                                            pred=pred_flag, 
                                            has_obstacles=has_ped)
            lowest_cost_ind = np.argmin(costs)

            start_config = rollouts[lowest_cost_ind][1]
            robot_path.append(start_config)
            if react_flag:
                ped_pos_current, ped_vel_current = sim_step(start_config, end_config, 
                                                            dt, v, tf_horizon,
                                                            ped_pos_current, ped_vel_current, 
                                                            ped_goals_current)

        if (time_out == True):
            end_status.append(2) # 2 = timeout
            if include_when_fail:
                min_ped_distances.append(min_ped_dist)
                time_duration.append((frame - start_frame) * dt)
                path_lengths.append(gh.estimate_path_length(robot_path))
                path_irregularities.append(gh.estimate_path_irregularity(robot_path))
            all_robo_paths.append(robot_path)

    success = 0
    for st in end_status:
        if st == 0:
            success += 1
    success_rate = success / len(end_status) * 100    
    if success == 0:
        return (0, 0, 0, 0, 0, all_robo_paths)
    else:
        min_ped_distance = np.min(np.array(min_ped_distances))
        avg_duration = np.mean(np.array(time_duration))
        avg_path_length = np.mean(path_lengths)
        avg_path_irregularity = np.mean(path_irregularities)
        return (success_rate, min_ped_distance, avg_duration, 
                avg_path_length, avg_path_irregularity, all_robo_paths)



if __name__ == "__main__":

    args = gh.get_flags()
    group_flag, laser_flag, pred_flag, react_flag, pred_method = args

    pfile_name = "test_cases/all.txt"
    with open(pfile_name, "rb") as fp:
        cases = pickle.load(fp)
    cases = [cases[0]]

    if group_flag:
        exp_name = "group_" + pred_method
    else:
        exp_name = "ped_" + pred_method
    if laser_flag:
        exp_name += "_laser"
    if react_flag:
        w = 0.3
        exp_name += "_react"
    else:
        w = 0.65

    fname = "results/" + exp_name + ".txt"
    num_cases = len(cases)
    all_results = []
    for i, case in enumerate(cases):
        print([i, num_cases])

        dataset = case[0]
        dataset_idx = case[1]
        init_start_config = case[2]
        init_end_config = case[3]
        start_frame = case[4]
        time_limit = case[5]

        start_time = time()
        metrics = run_trial(dataset, 
                            dataset_idx, 
                            init_start_config, 
                            init_end_config, 
                            start_frame, 
                            time_limit,
                            args,
                            w)
        end_time = time()
        print("Time taken for the past trial: ", end_time - start_time)
        all_results.append((case, metrics))
        with open(fname, "wb") as f:
            pickle.dump(all_results, f)

