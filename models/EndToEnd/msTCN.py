#encoding:utf8
#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, inplanes, depth, num_classes, bottleneck=False):
        super(ResNet, self).__init__()
        self.inplanes = inplanes

        blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3],
                  200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        #self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x) # x:(64,2,224,224)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)    # (64,512,7,7)

        x = self.avgpool(x)   # kernel=7  ==> (64, 512,1,1)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

class MultiStageModel(nn.Module):
    # num_stages = 4  num_layers = 10  num_f_maps = 64  4个stage，一个模块10层，没层都是64个filter
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.cnn = ResNet(inplanes=64, depth=18, num_classes=6)
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])
        #self.bn2 = nn.BatchNorm1d(num_f_maps)
        #   copy.deepcopy 深拷贝 拷贝对象及其子对象

    def forward(self, x):
        # Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        inputs = x.squeeze().permute(1, 0, 2, 3)
        out = self.cnn(inputs)  # (1,2,64,224,224)    (256,512,1,1)
        # out: (1, 152600)
        out = out.squeeze()
        out = out.unsqueeze(0).permute(0, 2, 1)

        out = self.stage1(out)
        outputs = out.unsqueeze(0)   # batch_input_tensor:（bz, 2048, max)----> (1,bz,C_out,L_out)
        for s in self.stages:

            out = s(F.softmax(out, dim=1))  # （ bz,C_out,L_out） dim是要计算的维度
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        #outputs = self.bn2(outputs)
            #  将每个stage的output做并列拼接
            #  最后维度（4,bz,num_classes,max）

        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)  # kernel_size = 1 用1x1conv 降维，从2048降到64
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        #self.bn = nn.BatchNorm1d(num_f_maps)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        #out = self.bn(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        # 一维卷积层，输入的尺度是(N, C_in,L)，输出尺度（ N,C_out,L_out）  加padding后 L_out= L
        #  kernel_size=3 dilation=2 ** i 随层数指数递增 1~512, 感受野计算：lk = l(k-1) + 2*dilation, 最后一层每个filter的感受野2047
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)    # 尝试 x+out后再加bn和relu
        #return F.relu(x + out)
        return out


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        #self.model.load_state_dict(torch.load('/disk2/lzq/models/thumos/split_3/epoch-200.model'))
        self.model.train()

        self.model.to(device)
        #optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                #  batch_input_tensor:（bz, 2048, max), batch_target_tensor: (bz, max), mask: (bz, num_class, max)
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)   # predictions最后一层的输出维度(4,bz,num_classes,max)

                loss = 0
                for p in predictions:
                    #  target:将样本和标签转换数据维度，分别转成二维和一维，最后一维都是类别数
                    #  p.transpose(2, 1)交换维度 （bz,max,num_classes）
                    #  contiguous:返回一个内存连续的有相同数据的tensor，如果原tensor内存连续则返回原tensor
                    #  view():返回一个有相同数据但大小不同的tensor view(-1, self.num_classes)：将转成（bz*max, num_classes）
                    #  batch_target.view(-1):转成(bz*max)
                    #  nn.CrossEntropyLoss(ignore_index=-100):   Target: (N) N是mini-batch的大小，0 <= targets[i] <= C-1
                    #  loss(x,class)=−logexp(x[class])∑jexp(x[j])) =−x[class]+log(∑jexp(x[j]))  Input: (N,C) C 是类别的数量




                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    #  torch.clamp(input, min, max, out=None) → Tensor将输入input张量每个元素的夹紧到区间 [min,max]
                    #  nn.MSELoss(reduction='none')  F.log_softmax(p[:, :, 1:], dim=1):对所有类别求log(softmax)
                    #  dim类别维度 (int): A dimension along which log_softmax will be computed.
                    #  detach():返回一个新的 从当前图中分离的 Variable,被detach 的Variable 指向同一个tensor
                    #  对p中的向量，分别从max维度的：1~max帧和从0~max-1帧划分，错位做均方误差  x，y维度:(bz,max-1)
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                                                 F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                features = np.load(features_path + vid.split('.')[0] + '.npy').transpose()
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)     #  (4,1,class_nums, frames)
                # output = predictions[-1]
                # output = torch.nn.Softmax(dim=1)(output)   #  (1, class_nums, frames)

                # pred_video = output.mean(dim=0)
                # print(pred_video.data.cpu().numpy())

                # output = model(input_var)
                # output = torch.nn.Softmax(dim=1)(output)
                #
                # # store predictions
                # output_video = output.mean(dim=0)
                # outputs.append(output_video.data.cpu().numpy())
                # gts.append(target[0, :])
                # ids.append(meta['id'][0])
                # batch_time.update(time.time() - end)
                # end = time.time()

                _, predicted = torch.max(predictions[-1].data, 1)    # output the last stage's result, 1 means dim=1 ==> (1, frames)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [actions_dict.keys()[actions_dict.values().index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
