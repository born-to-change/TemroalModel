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
from models.classifyModel.C3D import C3D

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
    train_transforms = transforms.Compose([videotransforms.RandomCrop(112),
                                          videotransforms.RandomHorizontalFlip(),])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])


    dataset = Merl( split, root, output_dir, mode, mapping_file, gth_dir, numclass, transforms=train_transforms, save_dir='', num=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    model = C3D()
    #model = casualTCN.TCN(input_size=512, n_classes=6, num_channels=[64] * 8, kernel_size=3, dropout=0.2)
    #model = casualTCN.TCN(input_size=512, n_classes=6, num_channels=[128]*8, kernel_size=3, dropout=0.2) # inputchannels,nclass,channelsize,kernel,dropout

    #model = nn.DataParallel(model)

    model.load_state_dict(torch.load('/disk2/lzq/data/MERL/Results/c3d16x2/000400.pt'))
    model.cuda()
    model.train(False)
    total = 0
    correct = 0
    epoch =0
    while epoch < 30:
        epoch +=1
        print('{}epoch start'.format(epoch))
        for data in dataloader:
            inputs, labels, vid = data  # inputs: (bz, channels,frames,224,224)  labels: (bz, class, frames)

            # wrap them in Variable
            # inputslist = input.split(128, 2)
            # labelslist = labels.split(128, 2)
            # for i,inputs in enumerate(inputslist):
            inputs = Variable(inputs.cuda())
            t = inputs.size(2)  # frames
            # labels = labelslist[i]
            labels = Variable(labels.cuda())

            predictions = model(inputs)  # shape:(bz, class, L) classify: (bz, class)

            _, predicted = torch.max(predictions.cpu().data,
                                     1)  # output the last stage's result, 1 means dim=1 ==> (1, frames)
            labels = labels.squeeze()
            _, gth = torch.max(labels.cpu().data, 0)
            total += 1
            if gth.numpy()[0] == predicted.numpy()[0]:
                correct += 1


        # predicted = predicted.squeeze()
        # predicted = predicted.cpu().data.numpy()
        #
        # labels = labels.cpu().data.numpy()
        # labels = list(labels.argmax(axis=1).squeeze())

        # for i in range(len(labels)):
        #     total += 1
        #     if labels == predicted[i]:
        #         correct += 1
    print("Acc: %.4f" % (100 * float(correct) / total))
        # get the inputs





        # file_ptr = open('/disk2/lzq/data/MERL/mapping.txt', 'r')
        # actions = file_ptr.read().split('\n')[:-1]
        # file_ptr.close()
        # actions_dict = dict()
        #
        #
        #
        #
        # for a in actions:
        #     actions_dict[a.split()[1]] = int(a.split()[0])
        #
        # for i in range(len(predicted)):
        #     recognition = np.concatenate((recognition, [
        #         actions_dict.keys()[actions_dict.values().index(predicted[i])]]))
        #
        # f_name = vid[0]
        #
        # f_ptr = open('/disk2/lzq/data/MERL/Results/eteallframex8/' + f_name, "w")
        # f_ptr.write("### Frame level recognition: ###\n")
        # f_ptr.write(' '.join(recognition))
        # f_ptr.close()
        # upsample to input size
        # per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')  # shape[2]: 7->64







if __name__ == '__main__':
    run(mode=args.mode, root=args.root, split=args.split, output_dir=args.output_dir,
        load_model=args.load_model, mapping_file=args.mapping_file, gth_dir=args.gth_dir)
    # run('/Users/user/Desktop/merl/dataset', '/Users/user/Desktop/merl/splits/train.txt', save_dir=)
# -mode flow -gpu 2 -root /disk2/lzq/Videos_MERL_Shopping_Dataset -split /disk2/lzq/data/MERL/splits/text.txt -mapping_file /disk2/lzq/data/MERL/mapping.txt -gth_dir /disk2/lzq/data/MERL/groundTruth -output_dir /disk2/lzq/data/MERL/opticalflow
    # -load_model -root /disk2/lzq/Videos_MERL_Shopping_Dataset -split /disk2/lzq/data/MERL/splits/train.txt -save_dir /disk2/lzq/data/MERL/i3d_feature -mapping_file /disk2/lzq/data/MERL/mapping.txt -gth_dir /disk2/lzq/data/MERL/groundTruth -output_dir /disk2/lzq/data/MERL/frames