#encoding:utf8

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

time_step = 10
input_size = 1
lr = 0.02

steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )