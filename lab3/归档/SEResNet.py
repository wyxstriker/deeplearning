# -*- coding: utf-8 -*-
# @Time   : 2021/5/27 10:31
# @Author : hollow
# @File   : ResNet.py

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-g", "--gpu", help="if use gpu")
# args = parser.parse_args()

class SEResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, reduce=16):
        super(SEResBlock, self).__init__()
        self.resBlock = ResBlock(in_channel=in_channel, out_channel=out_channel, stride=stride)
        self.se = nn.Sequential(nn.Linear(out_channel, out_channel//reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(out_channel//reduce, out_channel),
                                nn.Sigmoid())
        # self.shortcut = nn.Sequential(nn.Conv2d())

    def forward(self, x):
        x = self.resBlock(x)
        batch_size, channel, _, _=x.size()
        y = nn.AvgPool2d(x.size(2))(x)
        y = nn.Flatten(start_dim=1)(y)
        y = self.se(y).view(batch_size, channel, 1, 1)
        y = x * y.expand_as(x)
        out = x + y
        return out



class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.main_path(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class SEResNet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # [64,((img_size-1)/2+1)/2,((img_size-1)/2+1)/2]

        self.ResBlock1 = SEResBlock(64, 64)
        self.ResBlock2 = SEResBlock(64, 128, stride=2)  # [128,(((img_size-1)/2+1)/2-1)/2+1,(((img_size-1)/2+1)/2-1)/2+1]
        self.ResBlock3 = SEResBlock(128, 128)
        self.ResBlock4 = SEResBlock(128, 256, stride=2)
        self.ResBlock5 = SEResBlock(256, 256)
        self.ResBlock6 = SEResBlock(256, 512, stride=2)
        self.ResBlock7 = SEResBlock(512, 512)  # [512,(((img_size-1)/2+1)/2-1)/8+1,(((img_size-1)/2+1)/2-1)/8+1]

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.bn1(self.conv1(x))))
        x = self.ResBlock1(x)
        x = self.ResBlock1(x)

        x = self.ResBlock2(x)
        x = self.ResBlock3(x)

        x = self.ResBlock4(x)
        x = self.ResBlock5(x)

        x = self.ResBlock6(x)
        x = self.ResBlock7(x)

        x = self.avg_pooling(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    model = SEResBlock(64, 256)
    x = torch.rand([5, 64,224,224])
    model(x)