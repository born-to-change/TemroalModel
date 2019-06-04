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
from models.classifyModel.C3D import C3D
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

def run(init_lr=0.1, max_steps=2000, mode='flow', root='', split='', batch_size=16, output_dir='' ,load_model='', save_model='', mapping_file='', gth_dir='', numclass=6):
    train_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
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
    # i3d = nn.DataParallel(i3d) num_stages, num_layers, num_f_maps, features_dim, num_classes
    #model = msTCN.MultiStageModel(num_stages=6, num_layers=10, num_f_maps=64, dim=512, num_classes=6)
    model = C3D()

    model.cuda()
    #model = nn.DataParallel(model)


    lr = init_lr
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00001)
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 4  # accum gradient
    epoch = 0
    steps = 0
    # train it
    while epoch < max_steps:  # for epoch in range(num_epochs):
        print 'epoch {}/{}'.format(steps, max_steps)
        print '-' * 10

        model.train(True)
        tot_loss = 0.0

        epoch_loss = 0
        correct = 0
        total = 0

        tot_loc_loss = 0
        for data in dataloader:
            inputs, labels, vid = data     # inputs: (bz, channels,frames,224,224)  labels: (bz, class, frames)
            inputs = Variable(inputs.cuda())
            t = inputs.size(2)  # frames

            labels = Variable(labels.cuda())

            per_frame_logits = model(inputs)  # shape:(bz, class) # (1, 6)
            # upsample to input size
            # per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')  # shape[2]: 7->64

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels.squeeze(1))  # sigmoid layer+ BCEloss
            tot_loc_loss += loc_loss.data[0]

            # compute classification loss (with max-pooling along time B x C x T)
            # cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
            #                                               torch.max(labels, dim=2)[0])  # (bz, 6)
            # tot_cls_loss += cls_loss.data[0]

            # loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
            loss = loc_loss
            tot_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #lr_sched.step()
        steps += 1
        print '{} Loc Loss: {:.4f}  Tot Loss: {:.4f}'.format('train', tot_loc_loss / (
            10), tot_loss / 10)
        tot_loss = tot_loc_loss = 0.
        if steps % 100 == 0:
            # print '{} Loc Loss: {:.4f}  Tot Loss: {:.4f}'.format('train', tot_loc_loss / (
            #     10), tot_loss / 10)
            # save model
            torch.save(model.state_dict(), save_model + str(steps).zfill(6) + '.pt')






if __name__ == '__main__':
    run(mode=args.mode, root=args.root, split=args.split, output_dir=args.output_dir,
        load_model=args.load_model, save_model=args.save_model, mapping_file=args.mapping_file, gth_dir=args.gth_dir)
    # run('/Users/user/Desktop/merl/dataset', '/Users/user/Desktop/merl/splits/train.txt', save_dir=)

    # -load_model -root /disk2/lzq/Videos_MERL_Shopping_Dataset -split /disk2/lzq/data/MERL/splits/train.txt -save_dir /disk2/lzq/data/MERL/i3d_feature -mapping_file /disk2/lzq/data/MERL/mapping.txt -gth_dir /disk2/lzq/data/MERL/groundTruth -output_dir /disk2/lzq/data/MERL/frames