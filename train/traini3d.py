import os

import sys
import argparse

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
from utils import videotransforms

import numpy as np

from models.i3d import InceptionI3d

from data.dataset import MerlClassify


def run(init_lr=0.1, max_steps=1000, mode='flow', root='', split='', batch_size=4, output_dir='' ,load_model='',
        save_model='', mapping_file='', gth_dir='', numclass=6):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    #test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = MerlClassify(split, root, output_dir, mode, mapping_file, gth_dir, numclass, transforms=train_transforms,
                   save_dir='', num=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # val_dataset = Merl(train_split, 'testing', root, mode, test_transforms)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36,
    #                                              pin_memory=True)

    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('../models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('../models/checkpoints/rgb_imagenet.pt'))
    i3d.replace_logits(6)
    # i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.module.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 4  # accum gradient
    steps = 0
    # train it
    while steps < max_steps:  # for epoch in range(num_epochs):
        print 'Step {}/{}'.format(steps, max_steps)
        print '-' * 10

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            for data in dataloader:
                num_iter += 1
                # get the inputs
                inputs, labels, vid = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())  # (bz,channels, L,H,W)
                t = inputs.size(2)
                labels = Variable(labels.cuda())  # (bz, cls, L)

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data[0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data[0]

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data[0]
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print '{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss / (
                                    10 * num_steps_per_update), tot_cls_loss / (10 * num_steps_per_update),
                                                                                             tot_loss / 10)
                        # save model
                        torch.save(i3d.module.state_dict(), save_model + str(steps).zfill(6) + '.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                print '{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss / num_iter,
                                                                                     tot_cls_loss / num_iter, (
                                                                                                 tot_loss * num_steps_per_update) / num_iter)


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, split=args.split, output_dir=args.output_dir,
        load_model=args.load_model, save_model=args.save_model, mapping_file=args.mapping_file, gth_dir=args.gth_dir)
