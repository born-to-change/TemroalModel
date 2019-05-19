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
from data.feature_dataset import Merl
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
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

def run(init_lr=0.1, max_steps=2e3, mode='flow', root='', split='', batch_size=1, output_dir='' ,load_model='', save_model='', mapping_file='', gth_dir='', numclass=5):
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                          videotransforms.RandomHorizontalFlip(),])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])


    dataset = Merl( split, root, output_dir, mode, mapping_file, gth_dir, numclass, transforms=train_transforms, save_dir='', num=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)

        i3d.load_state_dict(torch.load('../models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)

        i3d.load_state_dict(torch.load('../models/rgb_imagenet.pt'))
    i3d.replace_logits(6)
    i3d.cuda()
    #i3d = nn.DataParallel(i3d)

    for data in dataloader:
        inputs, labels, name = data  # inputs: (bz,channels, frames, H,W)
        # get the inputs
        features = []
        if os.path.exists(os.path.join(save_model, name[0] + '.npy')):
            continue

        for i in range(inputs.shape[2]//256):
            input = Variable(inputs[:, :, i * 256:(i+1) * 256 , :, :].cuda(), volatile=True)

            feature = i3d.extract_features(input)

            features.append(feature.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
        input = Variable(inputs[:, :, i * 256: -1, :, :].cuda(), volatile=True)

        feature = i3d.extract_features(input)
        features.append(feature.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
        np.save(os.path.join(save_model, name[0]), features)


if __name__ == '__main__':
    run(mode=args.mode, root=args.root, split=args.split, output_dir=args.output_dir,
        load_model=args.load_model, save_model=args.save_model, mapping_file=args.mapping_file, gth_dir=args.gth_dir)