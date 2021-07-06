import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

# MLP模型 5层
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 10)
        # self.l3 = nn.Linear(256, 128)
        # self.l4 = nn.Linear(128, 10)

    def forward(self, x):
        # 拍平张量
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        return self.l2(x)


if __name__ == '__main__':
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=False)
    test = datasets.MNIST(root='./data/mnist', train=False, transform=transform, download=False)

    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_size)
    model = MLP()
    # 交叉熵
    criterion = nn.CrossEntropyLoss()
    # SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    epoch = 50

    # lossTime = []
    # lossList = []
    # lossTest = []
    count = 0
    writer = SummaryWriter('run/exp')
    for i in range(epoch):
        running_loss = 0.0
        for idx, (pic, tag) in enumerate(train_loader, 0):
    #         清除梯度
            optimizer.zero_grad()
            # 前向传播计算loss
            pre = model(pic)
            loss = criterion(pre, tag)
            # 反向传播
            loss.backward()
            # 更新梯度
            optimizer.step()
            # 数据处理
            running_loss += loss.item()
            if idx % 300 == 299:
                print('[%d, %5d] loss: %.3f' % (i + 1, idx+1 , running_loss / 300))
                writer.add_scalar('loss', running_loss/300, global_step=count)
                running_loss = 0.0
                count += 1
                # lossTime.append(count)
                # lossList.append(loss.item())
                # lt = 0
                # num = 0
                # with torch.no_grad():
                #     for (pic, lab) in test_loader:
                #         lt += criterion(model(pic),lab).item()
                #         num += 1
                # lossTest.append(lt/num)

    # plt.title('learning process')
    # plt.xlabel('times')
    # plt.ylabel('loss')
    # plt.plot(lossTime, lossList, label='train set')
    # plt.plot(lossTime, lossTest, label='test set')
    # plt.legend()
    # plt.show()
    # torch.save(model.state_dict(), 'parameter.pkl')
    # model.load_state_dict(torch.load('parameter.pkl'))
    writer.close()
    correct = 0
    total = 0
    # 测试
    with torch.no_grad():
        for (pic, lab) in test_loader:
            output = model(pic)
            _, pre = torch.max(output.data, dim=1)
            total += lab.size(0)
            correct += (pre == lab).sum().item()
    print('Accuracy on test set: %.4f %%' % (100 * correct / total))
