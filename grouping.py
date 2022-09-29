import copy
import numpy as np
from sklearn.cluster import DBSCAN

class Grouping(object):

    # This class uses DBSCAN (a clustering algorithm) to 
    # group pedestrians into groups. 
    # This class also records group splits and merges events.
    # Group memberships are stored in video_labels_matrix.
    # Split and merge events are stored in video_dynamics_matrix.
    #
    # video_labels_matrix: 3d irregular list
    # ---- 1st Dimension indicates frames
    # ---- 2nd Dimension indicates people
    # ---- 3rd Dimension indicates the group membership of each person
    # ---- (video_position_matrix[i][j][0] means the unique group membership id
    # ---- of preson j in frame i)
    #
    # video_dynamics_matrix: 1d tuple list (despite the name, it's not 3d!)
    # Each element is a tuple with 5 elements (a,b,c,d,e)
    # ---- a: If it's 1 then it's a merge. If it's -1 then it's a split.
    # ---- b: The frame number when the event happens.
    # If the event is a merge:
    # ---- c: The first group membership id of the group before the merge.
    # ---- d: The second group membership id of the group before the merge.
    # ---- e: The group membership id of the group after the merge.
    # If the event is a split:
    # ---- c: The group membership id of the group before the split.
    # ---- d: The first group membership id of the group after the split.
    # ---- e: The second group membership id of the group after the split.
    # (Being first or second doesn't matter.)

    def __init__(self, msg, history):
        # Initialization
        # Inputs:
        # msg: message class object (should have data loaded first)
        # history: how many frames to look before the split and merge events
        #          (In this class, it is only used to check if we have complete information of 
        #           the group(s) involved in the split/merge before the action.)

        if msg.if_processed_data:
            self.total_num_frames = msg.total_num_frames
            self.video_position_matrix = msg.video_position_matrix
            self.video_velocity_matrix = msg.video_velocity_matrix
            self.video_pedidx_matrix = msg.video_pedidx_matrix
        else:
            raise Exception('Data has not been loaded yet!')

        self._load_parameters(history)

        self.num_groups = 0
        self.video_labels_matrix = [] 
        self.video_dynamics_matrix = []

        self._social_grouping()
        return 

    def update_message(self, msg):
        # Store everything computed from grouping into the message
        if msg.if_processed_group:
            raise Exception('Grouping already performed!')

        msg.if_processed_group = True
        msg.video_labels_matrix = self.video_labels_matrix
        msg.video_dynamics_matrix = self.video_dynamics_matrix
        msg.num_groups = self.num_groups
        return msg

    def _load_parameters(self, history):
        # Initialize parameters to prepare for DBSCAN
        # Inputs:
        # history: from __init__

        pos = 2.0
        ori = 30
        vel = 1.0
        self.param = {'position_threshold': pos,
                      'orientation_threshold': ori / 180.0 * np.pi,
                      'velocity_threshold': vel,
                      'velocity_ignore_threshold': 0.5,
                      'label_history_threshold': history}
        return

    @staticmethod
    def _DBScan_grouping(labels, properties, standard):
        # DBSCAN clustering
        # Inputs:
        # labels: the input labels. This will be destructively updated to 
        #         reflect the group memberships after DBSCAN.
        # properties: the input that clustering is based on.
        #             Could be positions, velocities or orientation.
        # standard: the threshold value for clustering.

        max_lb = max(labels)
        for lb in range(max_lb + 1):
            sub_properties = []
            sub_idxes = []
            # Only perform DBSCAN within groups (i.e. have the same membership id)
            for i in range(len(labels)):
                if labels[i] == lb:
                    sub_properties.append(properties[i])
                    sub_idxes.append(i)
        
            # If there's only 1 person then no need to further group
            if len(sub_idxes) > 1:
                db = DBSCAN(eps = standard, min_samples = 1)
                sub_labels = db.fit_predict(sub_properties)
                max_label = max(labels)

                # db.fit_predict always return labels starting from index 0
                # we can add these to the current biggest id number to create 
                # new group ids.
                for i, sub_lb in enumerate(sub_labels):
                    if sub_lb > 0:
                        labels[sub_idxes[i]] = max_label + sub_lb
        return labels

    @classmethod
    def grouping(cls, position_array, velocity_array, params=None):
        if params == None:
            pos = 2.0
            ori = 30
            vel = 1.0
            params = {'position_threshold': pos,
                      'orientation_threshold': ori / 180.0 * np.pi,
                      'velocity_threshold': vel,
                      'velocity_ignore_threshold': 0.5}

        num_people = len(position_array)
        vel_orientation_array = []
        vel_magnitude_array = []
        for [vx, vy] in velocity_array:
            velocity_magnitude = np.sqrt(vx ** 2 + vy ** 2)
            if velocity_magnitude < params['velocity_ignore_threshold']:
                # if too slow, then treated as being stationary
                vel_orientation_array.append((0.0, 0.0))
                vel_magnitude_array.append((0.0, 0.0))
            else:
                vel_orientation_array.append((vx / velocity_magnitude, vy / velocity_magnitude))
                vel_magnitude_array.append((0.0, velocity_magnitude)) # Add 0 to fool DBSCAN
        # grouping in current frame (three passes, each on different criteria)
        labels = [0] * num_people
        labels = cls._DBScan_grouping(labels, vel_orientation_array,
                                  params['orientation_threshold'])
        labels = cls._DBScan_grouping(labels, vel_magnitude_array,
                                  params['velocity_threshold'])
        labels = cls._DBScan_grouping(labels, position_array,
                                  params['position_threshold'])
        return labels

    def _check_history(self, label, frame_idx):
        # Check if a group membership id exists from (frame_idx - history) to frame_idx
        # We only consider split/mrege to be valid if the participating groups
        # exist in the duration of the history leading to the action frame. 
        # Inputs:
        # label: the goup membership id to be checked
        # frame_idx: the frame that the split/merge event occurs

        history = self.param['label_history_threshold']
        if frame_idx < history:
            return False
        for i in range(frame_idx - history, frame_idx):
            if not (label in self.video_labels_matrix[i]):
                return False
        return True

    def _social_grouping(self):
        # Perform social grouping and split/merge identification using a
        # frame-by-frame based approach. For each frame, group assignments
        # are performed first. Then the groups are compared with groups from
        # the previous frame to determine whether they are new groups or existing 
        # groups. Lastly, also by comparing with groups fro the previous frame,
        # we can know whether a split or merge occurs.
        #
        # Group memberships are stored in video_labels_matrix
        # Split/Merge event info are stored in video_dynamics_matrix

        prev_labels = []
        prev_pedidx = []
        for i in range(self.total_num_frames):
            # get grouping criterion (inputs for DBSCAN)
            position_array = self.video_position_matrix[i]
            velocity_array = self.video_velocity_matrix[i]
            pedidx_array = self.video_pedidx_matrix[i]
            num_people = len(position_array)
            if not (num_people > 0):
                prev_labels = []
                prev_pedidx = []
                self.video_labels_matrix.append([])
                continue

            labels = self.grouping(position_array, velocity_array, self.param)

            # Check temporal consistency (cross frame comparison)
            if i == 0:
                temporal_labels = copy.deepcopy(labels)
            else:
                temporal_labels = [-1] * num_people
                # Get the temporal labels (labels from the previous frame) 
                for j in range(num_people):
                    curr_idx = pedidx_array[j]
                    for k in range(len(prev_labels)):
                        if prev_pedidx[k] == curr_idx:
                            temporal_labels[j] = prev_labels[k]

                # Figure out new groups
                for j in range(num_people):
                    curr_label = temporal_labels[j]
                    reference_label = labels[j]
                    # new group or join current group
                    if curr_label == -1:
                        found_group = False
                        for k in range(num_people):
                            if (labels[k] == reference_label) and (temporal_labels[k] != -1):
                                change_to_label = temporal_labels[k]
                                found_group = True
                        if not found_group:
                            change_to_label = max(self.num_groups, max(temporal_labels)) + 1
                        for k in range(j, num_people):
                            if labels[k] == reference_label:
                                temporal_labels[k] = change_to_label

                # resolve splits and merges
                dynamics_array = []
                for j in range(num_people):
                    curr_label = temporal_labels[j]
                    reference_label = labels[j]
                    for k in range(num_people):
                        if (temporal_labels[k] != curr_label) and \
                           (labels[k] == reference_label):  #merges
                            change_to_label = max(self.num_groups, max(temporal_labels)) + 1
                            if (self._check_history(temporal_labels[k], i)) and \
                               (self._check_history(curr_label, i)):
                                dynamics_array.append((1, curr_label, temporal_labels[k], j))
                            for l in range(num_people):
                                if labels[l] == reference_label:
                                    temporal_labels[l] = change_to_label
                        if (temporal_labels[k] == curr_label) and \
                           (labels[k] != reference_label): #splits
                            change_to_label_1 = max(self.num_groups, max(temporal_labels)) + 1
                            change_to_label_2 = max(self.num_groups, max(temporal_labels)) + 2
                            if self._check_history(curr_label, i):
                                dynamics_array.append((-1, curr_label, j, k))
                            for l in range(num_people):
                                if (labels[l] == labels[k]):
                                    temporal_labels[l] = change_to_label_1
                                if (labels[l] == reference_label):
                                    temporal_labels[l] = change_to_label_2

                for info in dynamics_array:
                    if info[0] == -1:
                        self.video_dynamics_matrix.append((-1, i, info[1],
                                temporal_labels[info[2]], temporal_labels[info[3]]))
                    elif info[0] == 1:
                        self.video_dynamics_matrix.append((1, i, info[1], info[2],
                                temporal_labels[info[3]]))

            prev_labels = temporal_labels
            prev_pedidx = pedidx_array
            self.num_groups = max(self.num_groups, max(temporal_labels))
            self.video_labels_matrix.append(temporal_labels)

        print('Social Grouping done!')
        return
