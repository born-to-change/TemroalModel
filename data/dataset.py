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
    if not os.path.exists(os.path.join(output_dir)):
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

def load_rgb_frames(image_dir, vid, start, num):

    frames = []
    labels = []
    for i in range(start, start+num):

        img = cv2.imread(os.path.join(image_dir, vid+'_crop', str(i)+'.jpg'))  # [:, :, [2, 1, 0]]
        # print(os.path.join(image_dir, vid, image_tmpl.format(i)))
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)




def load_flow_frames(image_dir, vid, start, num):
    frames = []

    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid+'_crop', flow_tmpl.format('flow_x', i)), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid+'_crop', flow_tmpl.format('flow_y', i)), cv2.IMREAD_GRAYSCALE)

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


def make_dataset(output_dir, split_file, mapping_file, gth_dir, numclass=6):
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
        vid_frame_path = os.path.join(output_dir, video+'_crop')
        vid_frames_list = os.listdir(vid_frame_path)
        # vid_frames = open(vid_frame_path, 'r')
        # vid_frames_list = vid_frames.read()
        # vid_frames.close()

        num_frames = min(len(vid_frames_list), len(gth_list))

        label = np.zeros((6, num_frames), np.float32)
        for fr, cls in enumerate(gth_list):
            label[cls, fr] = 1  # binary classification

        dataset.append((video, label, num_frames))

    return dataset




class Merl(data_util.Dataset):
    def __init__(self, split_file, root, output_dir, mode, mapping_file, gth_dir, numclass, transforms=None, save_dir='', num=0):
        self.root = output_dir
        self.save_dir = save_dir
        self.mode = mode
        self.split_file = split_file
        self.transforms = transforms
        self.data = make_dataset(output_dir, split_file, mapping_file, gth_dir, numclass)


    def __getitem__(self, index):
        vid, label, nf = self.data[index]

        start_f = random.randint(1,nf-65)

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, 64)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, 64)
        label = label[:, start_f:start_f+64]

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label)

        # clips = torch.FloatTensor()
        # gt = []
        # if os.path.exists(os.path.join(self.save_dir, vid + '.npy')):
        #     return 0, 0, vid
        #
        # label_dict={}
        # for x in range(1, nf+1):
        #     label_dict[x] = label[x-1]
        #
        # during = 48
        # if nf>15000:
        #     during =48
        # if nf>30000:
        #     during = 96
        # for i in range(0, nf+1, during):
        #     if i+49>nf:
        #         continue
        #     else:
        #         start_frame = max(i, 1)
        #     if self.mode == 'rgb':
        #         imgs, labels = load_rgb_frames(self.root, vid, start_frame, 48, label_dict)
        #     else:
        #         imgs, labels = load_flow_frames(self.root, vid, start_frame, 48, label_dict)
        #     imgs = self.transforms(imgs)
        #     clips.append(imgs)
        #     gt.append(labels)
        #
        # return torch.Tensor(clips), torch.Tensor(gt), vid


    def __len__(self):
        return len(self.data)

