import mmcv
import numpy as np
import os


# f = open('/Users/user/Desktop/merl/groundTruth/1_1.txt', 'r')
# print(f.read().split('\n')[:-1])
#
# file_ptr = open('/Users/user/Desktop/merl/mapping.txt', 'r')
# actions = file_ptr.read().split('\n')[:-1]
# file_ptr.close()
# actions_dict = dict()
# for a in actions:
#     actions_dict[a.split()[1]] = int(a.split()[0])
#
# gth_list = open('/Users/user/Desktop/merl/groundTruth/1_1.txt')
# gth_list = gth_list.read().split('\n')[:-1]
# gth_list = [actions_dict[x] for x in gth_list]


label = np.zeros(1, np.int64)
print(label)