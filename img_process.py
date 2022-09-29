import cv2
import numpy as np
from message import Message
from PIL import Image

class ProcessImage(object):

    def __init__(self, msg, img_seq):
        self.frame_width = msg.frame_width
        self.frame_height = msg.frame_height
        frame_width = self.frame_width
        frame_height = self.frame_height
        left = frame_width / 2
        up = frame_height / 2
        right = frame_width / 2
        low = frame_height / 2
        for img in img_seq:
            im = Image.fromarray(np.uint8(img))
            bbox = im.getbbox()
            if bbox == None:
                continue
            if bbox[0] < left:
                left = bbox[0]
            if bbox[1] < up:
                up = bbox[1]
            if bbox[2] > right:
                right = bbox[2]
            if bbox[3] > low:
                low = bbox[3]
        upper_left = (left, up)
        lower_right = (right, low)
        upper_left_p = (frame_width - upper_left[0], frame_height - upper_left[1])
        lower_right_p = (frame_width - lower_right[0], frame_height - lower_right[1])
        self.bbox_crop = (min(upper_left[0], lower_right[0], upper_left_p[0], lower_right_p[0]),
                          min(upper_left[1], lower_right[1], upper_left_p[1], lower_right_p[1]),
                          max(upper_left[0], lower_right[0], upper_left_p[0], lower_right_p[0]),
                          max(upper_left[1], lower_right[1], upper_left_p[1], lower_right_p[1]))
        self.box_size = (self.bbox_crop[2] - self.bbox_crop[0],
                         self.bbox_crop[3] - self.bbox_crop[1])
        self.process_scale = 224.0 / max(self.box_size)
        self.new_size = (int(self.box_size[0] * self.process_scale),
                         int(self.box_size[1] * self.process_scale))
        return

    def process_image(self, img, debug=False):
        im = Image.fromarray(np.uint8(img))
        im = im.crop(self.bbox_crop)
        paste_im = im.resize(self.new_size)
        im = Image.new('RGB', (224, 224))
        im.paste(paste_im, (112 - paste_im.size[0] // 2, 112 - paste_im.size[1] // 2))
        if debug:
            return np.array(im)
        else:
            im = np.array(im) / 255.0
            im = im[:, :, 0]
            im = np.expand_dims(im, axis = 2)
            return im
    
    def reverse_process_image(self, img, debug=False):
        im = Image.fromarray(np.uint8(img))
        new_crop = (112 - self.new_size[0] // 2, 
                    112 - self.new_size[1] // 2,
                    112 + int(np.ceil(self.new_size[0] / 2)), 
                    112 + int(np.ceil(self.new_size[1] / 2)))
        im = im.crop(new_crop)
        paste_im = im.resize(self.box_size)
        im = Image.new('RGB', (self.frame_width, self.frame_height))
        im.paste(paste_im, (self.bbox_crop[0], self.bbox_crop[1]))
        if debug:
            return np.array(im)
        else:
            im = np.array(im) / 255.0
            im = im[:, :, 0]
            im = np.expand_dims(im, axis = 2)
            return im

class DrawGroupShape(object):

    def __init__(self, msg):
        self.H = msg.H
        self.dataset = msg.dataset
        self.frame_width = msg.frame_width
        self.frame_height = msg.frame_height
        self.center_set = False
        self.aug_set = False
        return

    def coordinate_transform(self, coord):
        # Transform the coordinates from metric space into pixel space
        # Units are now pixels instead of meters after the transformation.

        pt = np.matmul(np.linalg.inv(self.H), [[coord[0]], [coord[1]], [1.0]])
        x = pt[0][0] / pt[2][0]
        y = pt[1][0] / pt[2][0]
        if self.dataset == 'ucy':
            tmp_y = y
            y = self.frame_width / 2 + x
            x = self.frame_height / 2 - tmp_y
        x = int(round(x))
        y = int(round(y))
        return (y, x)

    def set_center(self, vertice_seq):
        self.center_set = True
        vertices = vertice_seq[-1]
        center = [0, 0]
        for v in vertices:
            center[0] += v[0]
            center[1] += v[1]
        center[0] = center[0] / float(len(vertices))
        center[1] = center[1] / float(len(vertices))
        center = self.coordinate_transform(center)
        self.center_offset = (self.frame_width / 2 - center[0],
                              self.frame_height / 2 - center[1])
        return
        
    def set_aug(self, angle=None, trans=None):
        self.aug_set = True
        if angle is None:
            self.aug_angle = np.random.choice(360)
        else:
            self.aug_angle = angle
        if trans is None:
            self.aug_trans = (0, 0)
        else:
            self.aug_trans = trans
        return

    def move_center(self, coord):
        x = coord[0] + self.center_offset[0]
        y = coord[1] + self.center_offset[1]
        return (int(x), int(y))

    def reverse_move_center(self, coord):
        x = coord[0] - self.center_offset[0]
        y = coord[1] - self.center_offset[1]
        return (int(x), int(y))

    def reverse_move_center_img(self, img):
        M = np.array([[1, 0, -self.center_offset[0]], [0, 1, -self.center_offset[1]]])
        rst = cv2.warpAffine(img, M, (self.frame_width, self.frame_height))
        return rst

    def aug_transform(self, coord):
        x = coord[0] + self.aug_trans[0]
        y = coord[1] + self.aug_trans[1]
        x -= self.frame_width / 2
        y -= self.frame_height / 2
        nx = np.cos(self.aug_angle) * x - np.sin(self.aug_angle) * y
        ny = np.sin(self.aug_angle) * x + np.cos(self.aug_angle) * y
        nx += self.frame_width / 2
        ny += self.frame_height / 2
        return (int(nx), int(ny))

    def reverse_aug_transform(self, coord):
        x = coord[0]
        y = coord[1]
        x -= self.frame_width / 2
        y -= self.frame_height / 2
        nx = np.cos(-self.aug_angle) * x - np.sin(-self.aug_angle) * y
        ny = np.sin(-self.aug_angle) * x + np.cos(-self.aug_angle) * y
        nx += self.frame_width / 2
        ny += self.frame_height / 2
        return (int(nx) - self.aug_trans[0], int(ny) - self.aug_trans[1])

    def reverse_aug_transform_img(self, img):
        M = np.array([[1.0, 0, -self.aug_trans[0]], 
                      [0, 1.0, -self.aug_trans[1]]])
        img = cv2.warpAffine(img, M, (self.frame_width, self.frame_height))
        M = cv2.getRotationMatrix2D((self.frame_width / 2, self.frame_height / 2), 
                                    -self.aug_angle / np.pi * 180, 1)
        img = cv2.warpAffine(img, M, (self.frame_width, self.frame_height))
        return img

    def draw_group_shape(self, vertices, frame, center=False, aug=False):
        convex_hull_vertices = []
        for i, elem in enumerate(vertices):
            coord = self.coordinate_transform(elem)
            if center and self.center_set:
                coord = self.move_center(coord)
            elif center and (not self.center_set):
                print('Warning! Centering parameters not set so not performed!')
            if aug and self.aug_set:
                coord = self.aug_transform(coord)
            elif aug and (not self.aug_set):
                print('Warning! Augmentation parameterss not set so not performed!')
            convex_hull_vertices.append(coord)
        cv2.fillConvexPoly(frame, np.array(convex_hull_vertices), (255, 255, 255))
        return frame

