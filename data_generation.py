import cv2
import numpy as np
from message import Message
from data_loader import DataLoader
from grouping import Grouping
from group_shape_generation import GroupShapeGeneration
from img_process import ProcessImage, DrawGroupShape

class DataGeneration(object):

    def __init__(self, history, offset, train_set, test_set):
        self.fps = 10

        self.history = history
        self.og_history = history
        #self.history += 1
        self.history += 8
        self.offset = offset
        self.history *= offset
        self.msg_group = self._get_msg_group()
        self.train_idx = train_set
        self.test_idx = test_set

        self.train_prob = self._calculate_set_prob(self.train_idx)
        self.test_prob = self._calculate_set_prob(self.test_idx)
        self.group_labels, self.frame_labels = self._get_labels()

        self.num_train_data = 0
        self.num_test_data = 0
        for e in self.train_idx:
            self.num_train_data += len(self.group_labels[e])
        for e in self.test_idx:
            self.num_test_data += len(self.group_labels[e])
        return

    def _get_msg_group(self):
        msg_group = []
        dataset_list = ['eth', 'eth', 'ucy', 'ucy', 'ucy']
        dataset_idx_list = [0, 1, 0, 1, 2]
        #dataset_list = ['eth']
        #dataset_idx_list = [0]

        for i in range(len(dataset_list)):
            dataset = dataset_list[i]
            dataset_idx = dataset_idx_list[i]
            msg = Message()
            data = DataLoader(dataset, dataset_idx, self.fps)
            msg = data.update_message(msg)
            gp = Grouping(msg, self.history)
            msg = gp.update_message(msg)
            msg_group.append(msg)

        return msg_group

    def _calculate_set_prob(self, set_idx):
        set_prob = []
        for i in set_idx:
            msg = self.msg_group[i]
            gp_labels = msg.video_labels_matrix
            set_prob.append(len(self._get_unique_labels(gp_labels)))
        set_prob = np.array(set_prob)
        set_prob = set_prob / np.sum(set_prob)
        return set_prob

    def _get_labels(self):
        # group_labels: all groups ids that exist longer than history
        # frame_labels: valid frames for the groups to sample from (accounting for history)

        group_labels = []
        frame_labels = []
        for msg in self.msg_group:
            msg_group_labels = []
            msg_frame_labels = []
            tmp_frame_labels = []
            labels = msg.video_labels_matrix
            max_label = np.max(self._get_unique_labels(labels))
            for i in range(max_label + 1):
                tmp_frame_labels.append([])
            for i, sub_list in enumerate(labels):
                for elem in sub_list:
                    tmp_frame_labels[elem].append(i)
            
            for i, sub_list in enumerate(tmp_frame_labels):
                sub_list = np.unique(sub_list)
                if not(len(sub_list) < self.history):
                    msg_group_labels.append(i)
                    msg_frame_labels.append(sub_list[(self.history - 1):None])
            group_labels.append(msg_group_labels)
            frame_labels.append(msg_frame_labels)
        return group_labels, frame_labels
   
    def _get_unique_labels(self, labels):
        all_labels = []
        for sub_list in labels:
            all_labels += sub_list
        return np.unique(all_labels)

    def generate_sample(self, from_train=True, debug=False):
        if from_train:
            idx = np.random.choice(self.train_idx, p=self.train_prob)
        else:
            idx = np.random.choice(self.test_idx, p=self.test_prob)
        msg = self.msg_group[idx]
        shape_gen_class = GroupShapeGeneration(msg)

        group_pool = self.group_labels[idx]
        #print(len(group_pool))
        frame_pool = self.frame_labels[idx]
        if (len(group_pool) == 0):
            raise Exception('No valid groups exist!')
        group_idx = np.random.choice(range(len(group_pool)))
        group = group_pool[group_idx]
        frame = np.random.choice(frame_pool[group_idx])
        img_seq = self._generate_img_sequence(shape_gen_class, msg, group, frame, debug, from_train)
        #return np.array(img_seq[:-1]), np.array(img_seq[-1])
        return np.array(img_seq[:self.og_history]), np.array(img_seq[self.og_history:])

    def generate_cases_all_groups(self, case_num):
        msg = self.msg_group[case_num]
        shape_gen_class = GroupShapeGeneration(msg)
        group_pool = self.group_labels[case_num]
        frame_pool = self.frame_labels[case_num]
        
        num_groups = len(group_pool)
        input_cases = []
        output_cases = []
        for i in range(num_groups):
            group = group_pool[i]
            frame = np.random.choice(frame_pool[i])
            img_seq = self._generate_img_sequence(shape_gen_class, msg, group, frame, False, True)
            input_cases.append(np.array(img_seq[:self.og_history]))
            output_cases.append(np.array(img_seq[self.og_history:]))    

        return input_cases, output_cases

    def _generate_img_sequence(self, shape_gen_class, msg, group, frame, debug, from_train):
        norm_ang = False

        vertice_sequence = []
        all_group_info = []
        for i in range(frame - self.history + 1, frame + 1, self.offset):
            vertices, group_info = shape_gen_class.generate_group_shape(i, group)
            vertice_sequence.append(vertices)
            all_group_info.append(group_info)

        dgs = DrawGroupShape(msg)
        dgs.set_center(vertice_sequence[:self.og_history])
        if norm_ang:
            # This is still bugged
            velocities = all_group_info[self.og_history - 1][1]
            avg_vel = dgs.coordinate_transform(np.mean(np.array(velocities), axis=0))
            aug_ang = np.arctan2(avg_vel[1], avg_vel[0]) / np.pi * 180
        else:
            aug_ang = None

        dgs.set_aug(angle=aug_ang)
        img_sequence = []
        for i, v in enumerate(vertice_sequence):
            canvas = np.zeros((msg.frame_height, msg.frame_width, 3), dtype=np.uint8)
            img = dgs.draw_group_shape(v, canvas, center=True, aug=from_train)
            img_sequence.append(img)

        pimg = ProcessImage(msg, img_sequence[:-1])
        for i, img in enumerate(img_sequence):
            img_sequence[i] = pimg.process_image(img, debug)

        return img_sequence

