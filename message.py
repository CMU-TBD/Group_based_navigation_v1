import numpy as np

class Message(object):
    # This class defines a message system that stores information.
    # These information are needed by all the other codes.

    def __init__(self):
        self.clear_message()
        return

    def clear_message(self):
        self.fname = ""
        self.total_num_frames = 0
        self.has_video = True
        self.frame_width = 0
        self.frame_height = 0
        self.dataset = ""
        self.flag = 0

        self.video_position_matrix = []
        self.video_velocity_matrix = []
        self.video_pedidx_matrix = []
        self.video_labels_matrix = []
        self.video_dynamics_matrix = []

        self.people_start_frame = []
        self.people_end_frame = []
        self.people_coords_complete = []
        self.people_velocity_complete = []

        self.frame_id_list = []
        self.person_id_list = []
        self.x_list = []
        self.y_list = []
        self.vx_list = []
        self.vy_list = []
        self.H = []

        self.num_groups = 0

        self.if_processed_data = False
        self.if_processed_group = False        

        return

