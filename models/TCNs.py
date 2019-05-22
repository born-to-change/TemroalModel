import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np

# class ED_TCN(nn.Module):
#     def __init__(self, n_nodes, conv_len, n_classes, dim, max_len, casual=False):
#         super(ED_TCN, self).__init__()
#         n_layers = len(n_nodes)
#         for i in range(n_layers):
#             if casual:
#                 self.padding = nn.ConstantPad1d((conv_len // 2, 0), 0)
#             self.conv1 = nn.Conv1d(dim, n_nodes[i], conv_len, padding=3)
#             # pad='same': pad=(f+1)/2    (in_channels, out_channels, kernel_size)
#
#             if casual:
#                 self.crop =









