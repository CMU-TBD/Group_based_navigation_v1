import numpy as np
import torch
import cv2
from grouping import Grouping
from group_shape_generation import GroupShapeGeneration
from img_process import ProcessImage, DrawGroupShape
from model import ConvAutoencoder
import general_helpers as gh

class GroupShapePrediction(object):

    def __init__(self, msg, path):
        # No need to do grouping here for msg
        self.msg = msg

        self.cuda = torch.device('cuda:0')
        ckpt = path
        self.model = ConvAutoencoder()

        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()
        self.model.to(self.cuda)

        print('Model initialized!')

        return

    def _load_parameters(self):
        # Initialize parameters to prepare for DBSCAN

        pos = 2.0
        ori = 30
        vel = 1.0
        params = {'position_threshold': pos,
                  'orientation_threshold': ori / 180.0 * np.pi,
                  'velocity_threshold': vel,
                  'velocity_ignore_threshold': 0.5}
        return params

    def _predict_sequence(self, input_sequence, pred_length):
        confidence_threshold = 0.5
        
        inputs = np.transpose(np.array(input_sequence), (3, 0, 1, 2))
        inputs_tensor = np.expand_dims(inputs, 0)
        inputs_tensor = torch.tensor(inputs_tensor, dtype=torch.float32, device=self.cuda)
        outputs_tensor = self.model(inputs_tensor)
        outputs = outputs_tensor.data.cpu().numpy()
        output_sequence = np.transpose(outputs[0, :, :, :, :], (1, 2, 3, 0))

        for i in range(pred_length):
            output_sequence[i] = np.round(output_sequence[i] >= confidence_threshold)
        return output_sequence

    def _predict_from_vertices(self, vertice_sequence, pred_seq_length):
        dgs = DrawGroupShape(self.msg)
        dgs.set_center(vertice_sequence)
        dgs.set_aug(angle=0)
        img_sequence = []
        for i, v in enumerate(vertice_sequence):
            canvas = np.zeros((self.msg.frame_height, self.msg.frame_width, 3), dtype=np.uint8)
            img = dgs.draw_group_shape(v, canvas, center=True, aug=False)
            img_sequence.append(img)

        pimg = ProcessImage(self.msg, img_sequence)
        for i, img in enumerate(img_sequence):
            img_sequence[i] = pimg.process_image(img, debug=False)

        pred_img_sequence = self._predict_sequence(img_sequence, pred_seq_length)

        group_pred_img_sequence = []
        for i, img in enumerate(pred_img_sequence):
            #img = np.round(np.repeat(img, 3, axis=2)) * 255
            img = np.round(np.repeat(img, 3, axis=2))
            pred_img = pimg.reverse_process_image(img, debug=True)
            pred_img = dgs.reverse_move_center_img(pred_img)
            group_pred_img_sequence.append(pred_img[:, :, 0])
        return group_pred_img_sequence

    def _compile_group_pred(self, all_pred_img_sequences, pred_length, num_groups):
        fnl_pred_img_sequence = []
        for i in range(pred_length):
            canvas = np.zeros((self.msg.frame_height, self.msg.frame_width), dtype=np.uint8)
            for j in range(num_groups):
                img = all_pred_img_sequences[j][i]
                img = np.round(img)
                canvas += img
            fnl_pred_img_sequence.append(np.clip(canvas, 0, 1))
        return fnl_pred_img_sequence

    def predict(self, positions, velocities, const):
        if (self.msg.dataset == "ucy") and (self.msg.flag == 2):
            pos = 1.5
            ori = 15
            vel = 0.5
            params = {'position_threshold': pos,
                      'orientation_threshold': ori / 180.0 * np.pi,
                      'velocity_threshold': vel,
                      'velocity_ignore_threshold': 0.5}
        else:
            params = self._load_parameters()
        
        position_array = []
        velocity_array = []
        num_people = len(positions)

        if num_people == 0:
            raise Exception('People Needed!')

        seq_length = len(positions[0])
        pred_seq_length = 8
        for i in range(num_people):
            position_array.append(positions[i][-1])
            velocity_array.append(velocities[i][-1])    
        labels = Grouping.grouping(position_array, velocity_array, params)

        all_labels = np.unique(labels)
        num_groups = len(all_labels)
        all_pred_img_sequences = []
        for ei, curr_label in enumerate(all_labels):
            group_positions = []
            group_velocities = []
            for i, l in enumerate(labels):
                if l == curr_label:
                    group_positions.append(positions[i])
                    group_velocities.append(velocities[i])
            
            vertice_sequence = []
            for i in range(seq_length):
                frame_positions = []
                frame_velocities = []
                for j in range(len(group_positions)):
                    frame_positions.append(group_positions[j][i])
                    frame_velocities.append(group_velocities[j][i])
                vertices = GroupShapeGeneration.draw_social_shapes(frame_positions, 
                                                                   frame_velocities,
                                                                   False,
                                                                   const)
                vertice_sequence.append(vertices)

            group_pred_img_sequence = self._predict_from_vertices(vertice_sequence, pred_seq_length)
            all_pred_img_sequences.append(group_pred_img_sequence)

        return self._compile_group_pred(all_pred_img_sequences, pred_seq_length, num_groups)

    def laser_predict(self, positions, velocities, const):
        if (self.msg.dataset == "ucy") and (self.msg.flag == 2):
            pos = 1.5
            ori = 15
            vel = 0.5
            params = {'position_threshold': pos,
                      'orientation_threshold': ori / 180.0 * np.pi,
                      'velocity_threshold': vel,
                      'velocity_ignore_threshold': 0.5}
        else:
            params = self._load_parameters()

        # Nearest geo-center way of building history
        time_steps = len(positions)
        group_pos_series = []
        group_vel_series = []
        group_centers = []
        group_vel_centers = []
        # Get group scan pts, vels, centers & center_vels for each frame
        for i in range(time_steps):
            pos = positions[i]
            vel = velocities[i]
            labels = Grouping.grouping(pos, vel, params)
            all_labels = np.unique(labels)
            num_groups = len(all_labels)
            all_group_pos = []
            all_group_vel = []
            centers = []
            vel_centers = []
            for j, curr_label in enumerate(all_labels):
                group_positions = []
                group_velocities = []
                center_x = 0
                center_y = 0
                center_vx = 0
                center_vy = 0
                for k, l in enumerate(labels):
                    if curr_label == l:
                        group_positions.append(pos[k])
                        group_velocities.append(vel[k])
                        center_x += pos[k][0]
                        center_y += pos[k][1]
                        center_vx += vel[k][0]
                        center_vy += vel[k][1]
                all_group_pos.append(group_positions)
                all_group_vel.append(group_velocities)
                num_members = len(group_positions)
                center_x /= num_members
                center_y /= num_members
                center_vx /= num_members
                center_vy /= num_members
                centers.append(np.array([center_x, center_y]))
                vel_centers.append(np.array([center_vx, center_vy]))
            group_pos_series.append(all_group_pos)
            group_vel_series.append(all_group_vel)
            group_centers.append(centers)
            group_vel_centers.append(vel_centers)

        temp_threshold = 2.5 / 10 #m/s / fps
        num_curr_groups = len(group_pos_series[-1])
        pred_seq_length = 8
        all_pred_img_sequences = []
        for i in range(num_curr_groups):
            position_seq = [group_pos_series[-1][i]]
            velocity_seq = [group_vel_series[-1][i]]
            config = group_centers[-1][i]
            break_idx = None
            save_idx = i
            # search nearest centers for each prev frame
            for j in range(time_steps-2, -1, -1):
                points = group_centers[j]
                min_dist, min_idx = gh.find_least_dist(config, points)
                if min_dist > temp_threshold:
                    break_idx = j
                    break
                else:
                    position_seq.append(group_pos_series[j][min_idx])
                    velocity_seq.append(group_vel_series[j][min_idx])
                    config = group_centers[j][min_idx]
                    save_idx = min_idx
            # if discrepancy, linear back-prop
            if not (break_idx == None):
                position_last = group_pos_series[break_idx + 1][save_idx]
                velocity_last = group_vel_series[break_idx + 1][save_idx]
                vel = group_vel_centers[break_idx + 1][save_idx]
                for j in range(break_idx, -1, -1):
                    position_last = list(np.array(position_last) - vel / 10)
                    position_seq.append(position_last)
                    velocity_seq.append(velocity_last)

            vertice_sequence = []
            for j in range(time_steps-1, -1, -1):
                vertices = GroupShapeGeneration.draw_social_shapes(position_seq[j],
                                                                   velocity_seq[j],
                                                                   True,
                                                                   const)
                vertice_sequence.append(vertices)

            group_pred_img_sequence = self._predict_from_vertices(vertice_sequence, pred_seq_length)
            all_pred_img_sequences.append(group_pred_img_sequence)

        return self._compile_group_pred(all_pred_img_sequences, pred_seq_length, num_curr_groups)
