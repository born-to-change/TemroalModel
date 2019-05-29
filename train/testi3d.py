#encoding:utf8

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import PIL
import torchvision
from torchvision import datasets, transforms
import numpy as np
from data.dataset import Merl
from models.i3d import InceptionI3d
from data import videotransforms
import os

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-split', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_model', type=str)
parser.add_argument('-mapping_file', type=str)
parser.add_argument('-gth_dir', type=str)
parser.add_argument('-output_dir', type=str)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def run(init_lr=0.1, max_steps=1, mode='flow', root='', split='', batch_size=1, output_dir='' ,load_model='', save_model='', mapping_file='', gth_dir='', numclass=5):
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                          videotransforms.RandomHorizontalFlip(),])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])


    dataset = Merl( split, root, output_dir, mode, mapping_file, gth_dir, numclass, transforms=train_transforms, save_dir='', num=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    if mode == 'flow':
        i3d = InceptionI3d(6, in_channels=2)

        i3d.load_state_dict(torch.load('/disk2/lzq/data/MERL/checkpoints/002000.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)

        i3d.load_state_dict(torch.load('../models/rgb_imagenet.pt'))
    i3d.replace_logits(6)
    i3d.cuda()

    i3d.train(False)

    total = 0
    correct = 0

    for data in dataloader:

        # get the inputs
        input, labels, vid = data  # inputs: (bz, channels,frames,224,224)  labels: (bz, class, frames)

        # wrap them in Variable
        inputslist = input.split(128, 2)
        labelslist = labels.split(128, 2)

        for i, inputs in enumerate(inputslist):
            inputs = Variable(inputs.cuda())
            t = inputs.size(2)  # frames
            labels = labelslist[i]
            labels = Variable(labels.cuda())


            per_frame_logits = i3d(inputs)  # shape:(bz, class, 7)  128 ->15
            # upsample to input size
            predictions = F.upsample(per_frame_logits, t, mode='linear')  # shape[2]: 7->64  shape:(bz, class, L)

            _, predicted = torch.max(predictions, 1)  # output the last stage's result, 1 means dim=1 ==> (1, frames)
            predicted = predicted.squeeze()
            predicted = predicted.cpu().data.numpy()

            labels = labels.cpu().data.numpy()
            labels = list(labels.argmax(axis=1).squeeze())

            for i in range(len(labels)):
                total += 1
                if labels[i] == predicted[i]:
                    correct += 1
    print("Acc: %.4f" % (100 * float(correct) / total))



if __name__ == '__main__':
    run(mode=args.mode, root=args.root, split=args.split, output_dir=args.output_dir,
        load_model=args.load_model, save_model=args.save_model, mapping_file=args.mapping_file, gth_dir=args.gth_dir)
    # run('/Users/user/Desktop/merl/dataset', '/Users/user/Desktop/merl/splits/train.txt', save_dir=)

    # -load_model -root /disk2/lzq/Videos_MERL_Shopping_Dataset -split /disk2/lzq/data/MERL/splits/train.txt -save_dir /disk2/lzq/data/MERL/i3d_feature -mapping_file /disk2/lzq/data/MERL/mapping.txt -gth_dir /disk2/lzq/data/MERL/groundTruth -output_dir /disk2/lzq/data/MERL/frames