import torch
import numpy as np
from torch.autograd import Variable
np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()



data = [[1,2],[3,4]]
list = [1,2,3,4]
tensor = torch.FloatTensor(data)

variable = Variable(tensor, requires_grad=True)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

v_out.backward()
print(tensor*tensor)
print(v_out)

# v_out = 1/4 * sum(variable*variable)
# the gradients w.r.t the variable, d(v_out)/d(variable) = 1/4*2*variable = variable/2
print(variable.grad)

print(variable.data)


