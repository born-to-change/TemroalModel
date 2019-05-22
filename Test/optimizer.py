# encoding:utf8

import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

lr = 0.01
batch_size = 32
epoch = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

plt.scatter(x.numpy(), y.numpy())
plt.show()

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True, num_workers=2,)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__ == '__main__':
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=lr)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=lr, momentum=0.9)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=lr, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=lr, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []]

    for e in range(epoch):
        print('Epoch:', e)
        for step, (b_x, b_y) in enumerate(loader):
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)
                loss = loss_func(output, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']

# 可见SGD很慢

    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])   # 总共384个loss值，缺省x默认为[0,1,2,3,4,...,383]
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()

    print(len(l_his))
