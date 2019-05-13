import argparse
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import PIL
import torchvision
from torchvision import datasets, transforms
import numpy as np
from data.dataset import Merl
from models.i3d import InceptionI3d


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-split', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)
parser.add_argument('-mapping_file', type=str)
parser.add_argument('-gth_dir', type=str)
parser.add_argument('-output_dir', type=str)


args = parser.parse_args()


def run(mode='rgb', root='', split='', batch_size=1, output_dir='' ,load_model='', save_dir='', mapping_file='', gth_dir='', numclass=5):
    dataset = Merl( split, root, output_dir, mode, mapping_file, gth_dir, numclass, transforms=None, feature_dir='', num=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                                             pin_memory=True)
    # if mode == 'flow':
    #     i3d = InceptionI3d(400, in_channels=2)
    # else:
    #     i3d = InceptionI3d(400, in_channels=3)
    # i3d.replace_logits(400)
    # i3d.load_state_dict(torch.load(load_model))
    for data in dataloader:
        # get the inputs
        inputs, labels, name = data


if __name__ == '__main__':
    run(mode=args.mode, root=args.root, split=args.split, output_dir=args.output_dir,
        load_model=args.load_model, save_dir=args.save_dir, mapping_file=args.mapping_file, gth_dir=args.gth_dir)
    # run('/Users/user/Desktop/merl/dataset', '/Users/user/Desktop/merl/splits/train.txt', save_dir=)

    # -load_model -root /Users/user/Desktop/merl/dataset -split /Users/user/Desktop/merl/splits/train.txt -save_dir  /Users/user/Desktop/merl/feature -mapping_file /Users/user/Desktop/merl/mapping.txt -gth_dir /Users/user/Desktop/merl/groundTruth -output_dir /Users/user/Desktop/merl/frames