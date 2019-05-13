# encoding:utf8
import torch
import torch.utils.data as data_util
import numpy as np
import json
import random
import os
import os.path
import cv2
import mmcv

from utils.video2img import video2imgage


'''
生成（vid, label, num_frames）格式的数据
'''
def video_to_frames(root, output_dir):
    # sp = open(split_file, 'r')
    # vid_list = sp.read().split('\n')[:-1]
    # vid_list = [x.split('.')[0] for x in vid_list]
    # for vid in vid_list:
    #     frame_list = os.path.join(root, vid)
    #     if not os.path.exists(frame_list):
    #         vid_frames = os.path.join(output_dir, vid)
    video2imgage(root, output_dir)
    return output_dir


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

image_tmpl="img_{:05d}.jpg"
flow_tmpl="{}_{:05d}.jpg"

def load_rgb_frames(image_dir, vid, nf):
  frames = []
  for i in range(nf):
      img = cv2.imread(os.path.join(image_dir, vid, ))
  
  for i in range(start, start+num, 3):
    img = cv2.imread(os.path.join(image_dir, vid, image_tmpl.format(i)))#[:, :, [2, 1, 0]]
    # print(os.path.join(image_dir, vid, image_tmpl.format(i)))
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num, 3):
        imgx = cv2.imread(os.path.join(image_dir, vid, flow_tmpl.format('flow_x', i)), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, flow_tmpl.format('flow_y', i)), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(output_dir, split_file, mapping_file, gth_dir, numclass):
    dataset = []

    #  生成label与实体的 dict
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    sp = open(split_file, 'r')
    vid_list = sp.read().split('\n')[:-1]
    for vid in vid_list:
        # 生成视频label list
        gth_list = open(os.path.join(gth_dir, vid), 'r')
        gth_list = gth_list.read().split('\n')[:-1]
        gth_list = [actions_dict[x] for x in gth_list]

        video = vid.split('.')[0]
        vid_frame_path = os.path.join(output_dir, video)
        vid_frames = open(vid_frame_path, 'r')
        vid_frames_list = vid_frames.read()
        vid_frames.close()

        num_frames = min(len(vid_frames_list), len(gth_list))
        dataset.append((video, gth_list[:num_frames], num_frames))
    return dataset




class Merl(data_util.Dataset):
    def __init__(self, split_file, root, output_dir, mode, mapping_file, gth_dir, numclass, transforms=None, feature_dir='', num=0):
        self.root = video_to_frames(root, output_dir)

        self.split_file = split_file
        self.transforms = transforms
        self.data = make_dataset(output_dir, split_file, mapping_file, gth_dir, numclass)


    def __getitem__(self, index):
        vid, label, nf = self.data[index]
        clips = []

        if os.path.exists(os.path.join(self.feature_dir, vid + '.npy')):
            return 0, 0, vid
        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, nf)
        else:
            imgs = load_flow_frames(self.root, vid)

        imgs = self.transforms(imgs)
        clips.append(video_to_tensor(imgs))


        return clips, torch.from_numpy(label), vid


    def __len__(self):
        return len(self.data)

