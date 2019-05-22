import numpy as np
import torch
import torch.nn.functional as F

a = np.arange(12).reshape(3,4)
input = torch.linspace(1, 12, 12).reshape(1,1,3,4)

print(input.shape)
print(F.avg_pool2d(input, kernel_size=1, stride=1))