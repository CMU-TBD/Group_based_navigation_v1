import cv2
import numpy as np

class DataLoader():

    # This class imports data from eth and ucy datasets (25 FPS)
    # The data are stored in two formats:
    # ---If using people centered format: 
    # ---(indices are matched - the ith person is the same person)
    # ------people_start_frame: record the start frame of each person 
    # ------                    when the person first makes its appearance
    # ------                    (people_start_frame[i] means the start frame of ith person)
    # ------people_end_frame: record the end frame of each person 
    # ------                  when the person last makes its appearance
    # ------                  (people_end_frame[i] means the end frame of ith person)
    # ------people_coords_complete: the coordinates of each person throughout its appearance
    # ------                        (people_coords_complete[i][j][0] means the x coordinate 
    # ------                         of person i in frame j+people_start_frame[i])
    # ------people_velocity_complete: the coordinates of each person throughout its appearance
    # ------                        (people_velocity_complete[i][j][0] means the x velocity 
    # ------                         of person i in frame j+people_start_frame[i])
    # ---If using frame centered format: 
    # ---(indices are matched - the ith frame is the same frame 
    # ---                       and the jth person is the same person)
    # ---They are not really matrices but it's just a legacy naming problem...
    # ------video_position_matrix: A 3D irregular list
    # ------                       1st Dimension indicates frames
    # ------                       2nd Dimension indicates people
    # ------                       3rd Dimension indicates coordinates of each person
    # ------                       (video_position_matrix[i][j][0] means the x coordinate
    # ------                        of person j in frame i)
    # ------video_velocity_matrix: A 3D irregular list
    # ------                       1st Dimension indicates frames
    # ------                       2nd Dimension indicates people
    # ------                       3rd Dimension indicates velocities of each person
    # ------                       (video_velocity_matrix[i][j][0] means the x velocity
    # ------                        of person j in frame i)
    # ------video_pedidx_matrix: A 3D irregular list
    # ------                     1st Dimension indicates frames
    # ------                     2nd Dimension indicates pedestrian_id
    # ------                     (video_pedidx_matrix[i][j] means the index id
    # ------                      of person j in frame i)
    #

    def __init__(self, dataset = 'eth', flag = 0, target_fps = 15):
        # Initialize data processor
        # Inputs:
        # dataset: can only be 'eth' or 'ucy'
        # flag: can only be 0 or 1 for 'eth'
        #       and 0-5 for 'ucy'
        # target_fps: desired fps for the labels
        #
        # dataset - flag       dataset name
        # eth - 0              ETH
        # eth - 1              HOTEL
        # ucy - 0              ZARA1
        # ucy - 1              ZARA2
        # ucy - 2              UNIV (or UNIV1)
        # ucy - 3              ZARA3
        # ucy - 4              UNIV2
        # ucy - 5              ARXIE (not recommended)

        self.dataset = dataset
        self.flag = flag

        self.video_position_matrix = [[]]
        self.video_velocity_matrix = [[]]
        self.video_pedidx_matrix = [[]]

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

        if dataset == 'eth':
            in_fps = 15
            read_success = self._read_eth_data(flag)
        elif dataset == 'ucy':
            in_fps = 25
            read_success = self._read_ucy_data(flag)
        else:
            print('dataset argument must be \'eth\' or \'ucy\'')
            read_success = False
        
        if read_success:
            self._organize_frame()
            if not (in_fps == target_fps):
                self._frame_matching(in_fps, target_fps)
            self._data_processing()
        else:
            raise Exception('Wrong inputs to the data loader!')

        return 

    def update_message(self, msg):
        # Store everything obtained/computed from the datasets into the message

        #if msg.if_processed_data:
        #    raise Exception('Data already loaded!')

        msg.if_processed_data = True
        msg.dataset = self.dataset
        msg.flag = self.flag
    
        msg.fname = self.fname
        msg.total_num_frames = self.total_num_frames
        msg.has_video = self.has_video
        msg.frame_width = self.frame_width
        msg.frame_height = self.frame_height

        msg.video_position_matrix = self.video_position_matrix
        msg.video_velocity_matrix = self.video_velocity_matrix
        msg.video_pedidx_matrix = self.video_pedidx_matrix

        msg.people_start_frame = self.people_start_frame
        msg.people_end_frame = self.people_end_frame
        msg.people_coords_complete = self.people_coords_complete
        msg.people_velocity_complete = self.people_velocity_complete

        msg.frame_id_list = self.frame_id_list
        msg.person_id_list = self.person_id_list
        msg.x_list = self.x_list
        msg.y_list = self.y_list
        msg.vx_list = self.vx_list
        msg.vy_list = self.vy_list
        msg.H = self.H

        return msg

    def _read_eth_data(self, flag):
        # Read data from the ETH dataset
        # Data is stored into x_list, y_list, vx_list, vy_list
        # Associated person_id and frames are stored into person_id_list, frame_id_list
        # Inputs:
        # flag - the flag argument from __init__
        # Returns:
        # True if data reading is successful
    
        if flag == 0:
            folder = 'seq_eth'
        elif flag == 1:
            folder = 'seq_hotel'
        else:
            print('Flag for \'eth\' should be 0 or 1')
            return False

        # Create a VideoCapture object and get basic video information
        self.fname = 'ewap_dataset/' + folder + '/' + folder + '.avi'
        cap = cv2.VideoCapture(self.fname)
        if (cap.isOpened()== False):
          print("Error opening video stream or file")
          return False
        self.total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(3))
        self.frame_height = int(cap.get(4))
        cap.release()
        self.has_video = True

        # Read Homography matrix
        fname = 'ewap_dataset/' + folder + '/H.txt'
        with open(fname) as f:
          for line in f:
            line = line.split(' ')
            real_line = []
            for elem in line:
              if len(elem) > 0:
                real_line.append(elem)
            real_line[-1] = real_line[-1][:-1]
            h1, h2, h3 = real_line
            self.H.append([float(h1), float(h2), float(h3)])

        f.close()

        # Read the data from text file 
        fname = 'ewap_dataset/'+ folder + '/obsmat.txt'

        with open(fname) as f:
          for line in f:
            line = line.split(' ')
            real_line = []
            for elem in line:
              if len(elem) > 0:
                real_line.append(elem)
            real_line[-1] = real_line[-1]
            frame_id, person_id, x, z, y, vx, vz, vy = real_line
            self.frame_id_list.append(int(round(float(frame_id))))
            self.person_id_list.append(int(round(float(person_id))))

            x = float(x)
            y = float(y)
            vx = float(vx)
            vy = float(vy)
            self.x_list.append(x)
            self.y_list.append(y)
            self.vx_list.append(vx)
            self.vy_list.append(vy)

        f.close()

        #print('File reading done!')
        return True

    def _read_ucy_data(self, flag):
        # Read data from the UCY dataset
        # Data is stored into x_list, y_list
        # Associated person_id and frames are stored into person_id_list, frame_id_list
        # Different from ETH, UCY don't provide velocity labels,
        # so vx_list and vy_list are generated via linear interpolations.
        # Inputs:
        # flag - the flag argument from __init__
        # Returns:
        # True if data reading is successful

        if flag == 0:
            folder = 'zara'
            source = 'crowds_zara01'
        elif flag == 1:
            folder = 'zara'
            source = 'crowds_zara02'
        elif flag == 2:
            folder = 'university_students'
            source = 'students003'
        elif flag == 3:
            folder = 'zara'
            source = 'crowds_zara03'
        elif flag == 4:
            folder = 'university_students'
            source = 'students001'
        elif flag == 5:
            folder = 'arxiepiskopi'
            source = 'arxiepiskopi1'
            print('Warning: bad data used!')
        else:
            print('Flag for \'ucy\' should be 0 - 5')
            return False

        # Create a VideoCapture object and read from input file
        if (flag == 3) or (flag == 4):
            # zara3 and univ2 don't have videos to read
            if flag == 3:
                alt_source = 'crowds_zara01'
            else:
                alt_source = 'students003'
            self.fname = 'ucy_dataset/' + folder + '/' + alt_source + '.avi'
            # Dummy video to capture frame information
            # ZARA1 ZARA2 ZARA3 have the same frame width and height
            # UNIV1 and UNIV2 also have the same frame width and height
            # number of frames will be determined by the last frame that contains information
            cap = cv2.VideoCapture(self.fname)
            if (cap.isOpened()== False):
              print("Error opening video stream or file")
              return False
            self.total_num_frames = -1
            self.has_video = False
        else:
            self.fname = 'ucy_dataset/' + folder + '/' + source + '.avi'
            cap = cv2.VideoCapture(self.fname)
            if (cap.isOpened()== False):
              print("Error opening video stream or file")
              return False
            self.total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.has_video = True
        self.frame_width = int(cap.get(3))
        self.frame_height = int(cap.get(4))
        cap.release()

        # Obtain the approximate H matrix 
        offx = 17.5949
        offy = 9.665722
        pts_img = np.array([[476, 117], [562, 117], [562, 311],[476, 311]])
        pts_wrd = np.array([[0, 0], [1.81, 0], [1.81, 4.63],[0, 4.63]])
        pts_wrd[:,0] += offx
        pts_wrd[:,1] += offy
        self.H, status = cv2.findHomography(pts_img, pts_wrd)

        # Read the data from text file 
        fname = 'ucy_dataset/' + folder + '/data_' + folder + '/' + source + '.vsp'
        with open(fname) as f:
          person_id = 0
          mem_x = 0
          mem_y = 0
          mem_frame_id = 0
          record_switch = True
          for idx, line in enumerate(f):
            line = line.split()
            if len(line) == 6:
              person_id += 1
              record_switch = False
              if idx > 0:
                vx = self.vx_list[-1]
                vy = self.vy_list[-1]
                self.vx_list.append(vx)
                self.vy_list.append(vy)
            if (len(line) == 8) or (len(line) == 4):
              x = float(line[0])
              y = float(line[1])
              frame_id = int(line[2])
              # convert units from pixels to meters using the approximated H
              pt = np.matmul(self.H, [[x], [y], [1.0]]) 
              x, y = (pt[0][0] / pt[2][0]), (pt[1][0] / pt[2][0])
              self.x_list.append(x)
              self.y_list.append(y)
              self.frame_id_list.append(frame_id)
              self.person_id_list.append(person_id)
              if record_switch:
                # velocity information not included from dataset
                # obtained by linear interpolation
                vx = (x - mem_x) / (frame_id - mem_frame_id)
                vy = (y - mem_y) / (frame_id - mem_frame_id)
                self.vx_list.append(vx * 25)
                self.vy_list.append(vy * 25)
              else:
                record_switch = True
              mem_x = x
              mem_y = y
              mem_frame_id = frame_id

          vx = self.vx_list[-1]
          vy = self.vy_list[-1]
          self.vx_list.append(vx)
          self.vy_list.append(vy)

        f.close()

        #print('File reading done!')
        return True

    def _organize_frame(self):
        # Connect paths for each individual person
        # For each person, the frame ids, the associated path 
        # coordinates and velocities are stored into person_*_complete

        num_people = max(self.person_id_list)
        for i in range(num_people):
          got_start = False
          prev_frame = 0
          curr_frame = 0
          person_coords = []
          person_velocity = []
          mem_x = 0
          mem_y = 0
          mem_vel_x = 0
          mem_vel_y = 0

          for j in range(len(self.frame_id_list)):
            if self.person_id_list[j] == (i + 1):
              curr_frame = self.frame_id_list[j]
              if not got_start:
                got_start = True
                self.people_start_frame.append(curr_frame)
                person_coords.append((self.x_list[j], self.y_list[j]))
                person_velocity.append((self.vx_list[j], self.vy_list[j]))
              else:
                num_frames_interpolated = curr_frame - prev_frame
                for k in range(num_frames_interpolated):
                  # for frames with missing information,
                  # linear interpolation is performed on the two neighboring frames
                  ratio = float(k + 1) / float(num_frames_interpolated)
                  diff_x = ratio * (self.x_list[j] - mem_x)
                  diff_y = ratio * (self.y_list[j] - mem_y)
                  person_coords.append((mem_x + diff_x, mem_y + diff_y))

                  diff_vx = ratio * (self.vx_list[j] - mem_vel_x)
                  diff_vy = ratio * (self.vy_list[j] - mem_vel_y)
                  person_velocity.append((mem_vel_x + diff_vx, mem_vel_y + diff_vy))

              mem_x, mem_y = self.x_list[j], self.y_list[j]
              mem_vel_x, mem_vel_y = self.vx_list[j], self.vy_list[j]
              prev_frame = curr_frame

          if got_start:
            self.people_end_frame.append(curr_frame)
            self.people_coords_complete.append(person_coords)
            self.people_velocity_complete.append(person_velocity)

        if self.total_num_frames == -1:
            self.total_num_frames = max(self.people_end_frame)

        #print('Frame organizing done!')
        return

    def _frame_matching(self, in_fps, out_fps):
        # Converts information stored in people_* arrays
        # from their original fps to a desired fps
        # Inputs:
        # in_fps - the original fps of the dataset labels
        # out_fps - desired fps to convert to

        total_time = self.total_num_frames / in_fps
        new_total_frames = int(np.floor(total_time * out_fps))
        num_people = len(self.people_start_frame)
        for i in range(num_people):
            curr_start = self.people_start_frame[i]
            curr_end = self.people_end_frame[i]
            # make sure the pedestrian exists in its new start frame and end frame
            new_start = int(np.ceil(curr_start / in_fps * out_fps))
            new_end = int(np.floor(curr_end / in_fps * out_fps))
            new_pos_array = []
            new_vel_array = []
            # bilinear interpolation to obtain pos and vel at new frames (timestamps)
            for j in range(new_start, new_end + 1):
                timestamp = j / out_fps
                old_frame_idx = timestamp * in_fps
                if abs(round(old_frame_idx) - old_frame_idx) < 1e-10:
                    idx = int(round(old_frame_idx)) - curr_start
                    new_pos_array.append(self.people_coords_complete[i][idx])
                    new_vel_array.append(self.people_velocity_complete[i][idx])
                else:
                    prev_idx = int(np.floor(old_frame_idx)) - curr_start
                    next_idx = int(np.ceil(old_frame_idx)) - curr_start
                    ratio = (old_frame_idx - curr_start) - prev_idx # / 1
                    new_pos = np.array(self.people_coords_complete[i][prev_idx]) * (1 - ratio) + \
                              np.array(self.people_coords_complete[i][next_idx]) * ratio
                    new_vel = np.array(self.people_velocity_complete[i][prev_idx]) * (1 - ratio) + \
                              np.array(self.people_velocity_complete[i][next_idx]) * ratio
                    new_pos_array.append((new_pos[0], new_pos[1]))
                    new_vel_array.append((new_vel[0], new_vel[1]))
            self.people_start_frame[i] = new_start
            self.people_end_frame[i] = new_end
            self.people_coords_complete[i] = new_pos_array
            self.people_velocity_complete[i] = new_vel_array

        return

    def _data_processing(self):
        # Precompute 3d video arrays and store them into video_*_matrix
        # (not really matrices but too lazy to fix now)
        # Each array contains information frame by frame
        # Entries in the array that share the same index points to the same person
        # pedidx highlights which person the entry corresponds to

        for i in range(self.total_num_frames):
            position_array = []
            velocity_array = []
            pedidx_array = []
            curr_frame = i + 1

            for j in range(len(self.people_start_frame)):
                curr_start = self.people_start_frame[j]
                curr_end = self.people_end_frame[j]
                if (curr_start <= curr_frame) and (curr_frame <= curr_end):
                    x, y = self.people_coords_complete[j][curr_frame - curr_start]
                    vx, vy = self.people_velocity_complete[j][curr_frame - curr_start]

                    position_array.append((float(x), float(y)))
                    velocity_array.append((float(vx), float(vy)))
                    pedidx_array.append(j)

            if len(position_array) > 0:
                self.video_position_matrix.append(position_array)
                self.video_velocity_matrix.append(velocity_array)
                self.video_pedidx_matrix.append(pedidx_array)
            else:
                self.video_position_matrix.append([])
                self.video_velocity_matrix.append([])
                self.video_pedidx_matrix.append([])

        #print('Initial data processing done!')
        return

