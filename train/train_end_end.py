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
from models.EndToEnd import casualTCN
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

def run(init_lr=0.1, max_steps=1000, mode='flow', root='', split='', batch_size=1, output_dir='' ,load_model='', save_model='', mapping_file='', gth_dir='', numclass=5):
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                          videotransforms.RandomHorizontalFlip(),])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])


    dataset = Merl( split, root, output_dir, mode, mapping_file, gth_dir, numclass, transforms=train_transforms, save_dir='', num=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    # if mode == 'flow':
    #     i3d = InceptionI3d(400, in_channels=2)
    #
    #     i3d.load_state_dict(torch.load('../models/flow_imagenet.pt'))
    # else:
    #     i3d = InceptionI3d(400, in_channels=3)
    #
    #     i3d.load_state_dict(torch.load('../models/rgb_imagenet.pt'))
    # i3d.replace_logits(6)
    # i3d.cuda()
    # i3d = nn.DataParallel(i3d)
    model = casualTCN.TCN(input_size=512, n_classes=6, num_channels=[128]*8, kernel_size=3, dropout=0.2) # inputchannels,nclass,channelsize,kernel,dropout
    model.cuda()


    lr = init_lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 4  # accum gradient
    steps = 0
    # train it
    while steps < max_steps:  # for epoch in range(num_epochs):
        print 'Step {}/{}'.format(steps, max_steps)
        print '-' * 10

        # Each epoch has a training and validation phase
        # for phase in ['train']:
        #     if phase == 'train':
        #         i3d.train(True)
        #     else:
        #         i3d.train(False)  # Set model to evaluate mode
        model.train(True)
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        num_iter = 0
        optimizer.zero_grad()

        # Iterate over data.
        for data in dataloader:
            num_iter += 1
            # get the inputs
            inputs, labels = data  # inputs: (bz, channels,frames,224,224)  labels: (bz, class, frames)

            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            t = inputs.size(2)  # frames
            labels = Variable(labels.cuda())

            per_frame_logits = model(inputs)  # shape:(bz, class, 7)
            # upsample to input size
            #per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')  # shape[2]: 7->64

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)  # sigmoid layer+ BCEloss
            tot_loc_loss += loc_loss.data[0]

            # compute classification loss (with max-pooling along time B x C x T)
            # cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
            #                                               torch.max(labels, dim=2)[0])  # (bz, 6)
            # tot_cls_loss += cls_loss.data[0]

            #loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
            loss = loc_loss / num_steps_per_update
            tot_loss += loss.data[0]
            loss.backward()

            if num_iter == num_steps_per_update:  # and phase == 'train':
                steps += 1
                num_iter = 0
                optimizer.step()
                optimizer.zero_grad()

                lr_sched.step()
                if steps % 10 == 0:
                    print '{} Loc Loss: {:.4f}  Tot Loss: {:.4f}'.format('train', tot_loc_loss / (
                            10 * num_steps_per_update), tot_loss / 10)
                    # save model
                    torch.save(model.state_dict(), save_model + str(steps).zfill(6) + '.pt')
                    tot_loss = tot_loc_loss = tot_cls_loss = 0.


                # if phase == 'val':
                #     print '{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss / num_iter,
                #                                                                          tot_cls_loss / num_iter, (
                #                                                                                  tot_loss * num_steps_per_update) / num_iter)



if __name__ == '__main__':
    run(mode=args.mode, root=args.root, split=args.split, output_dir=args.output_dir,
        load_model=args.load_model, save_model=args.save_model, mapping_file=args.mapping_file, gth_dir=args.gth_dir)
    # run('/Users/user/Desktop/merl/dataset', '/Users/user/Desktop/merl/splits/train.txt', save_dir=)

    # -load_model -root /disk2/lzq/Videos_MERL_Shopping_Dataset -split /disk2/lzq/data/MERL/splits/train.txt -save_dir /disk2/lzq/data/MERL/i3d_feature -mapping_file /disk2/lzq/data/MERL/mapping.txt -gth_dir /disk2/lzq/data/MERL/groundTruth -output_dir /disk2/lzq/data/MERL/frames