import os

import torch
from data_load import matSet
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data.dataloader
from tensorboardX import SummaryWriter
import time
from PIL import Image
import numpy as np


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.liner1 = torch.nn.Linear(g_size, hiddensize)
        self.liner2 = torch.nn.Linear(hiddensize, hiddensize)
        self.liner3 = torch.nn.Linear(hiddensize, 2)
        self.sigmod = torch.nn.ReLU()

    def forward(self, x):
        x = self.sigmod(self.liner1(x))
        x = self.sigmod(self.liner2(x))
        x = self.liner3(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.liner1 = torch.nn.Linear(2, hiddensize)
        self.liner2 = torch.nn.Linear(hiddensize, hiddensize)
        self.liner3 = torch.nn.Linear(hiddensize, 1)
        self.sigmod = torch.nn.ReLU()

    def forward(self, x):
        x = self.sigmod(self.liner1(x))
        x = self.sigmod(self.liner2(x))
        x = self.liner3(x)
        return x


def train(G, D, epoch, device):
    g_optimizer = optim.Adam(G.parameters(), lr=lr)
    d_optimizer = optim.Adam(D.parameters(), lr=lr)
    #     name = 'weightGANsigmod'
    name = 'weight'
    pathG = './'+name+'G.tar'
    pathD = './'+name+'D.tar'
    initepoch = 0
    if (os.path.exists(pathG) and os.path.exists(pathD)) is not True:
        loss = torch.nn.BCEWithLogitsLoss()
    else:
        checkpoint = torch.load(pathG, map_location=device)
        checkpoint2 = torch.load(pathD, map_location=device)

        G.load_state_dict(checkpoint['model_state_dict'])
        D.load_state_dict(checkpoint2['model_state_dict'])

        g_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint2['optimizer_state_dict'])

        initepoch = checkpoint['epoch']
        loss = checkpoint['loss']

    for e in range(initepoch, epoch):
        starttime = time.time()

        # D.zero_grad()
        l = 0
        total = 0
        for k, data in enumerate(mLoader):
            D.zero_grad()

            data = data.to(device)
            decision = D(data).mean()
            bce = loss(decision, torch.ones(decision.size()).to(device))
            bce.backward()

            fake = torch.rand([data.size(0), g_size]).to(device)
            fake = G(fake)
            decision = D(fake).mean()
            bce = loss(decision, torch.zeros(decision.size()).to(device))
            bce.backward()

            d_optimizer.step()

            if k%5 == 0:
                G.zero_grad()

                data = torch.rand([data.size(0), g_size]).to(device)
                data = G(data)
                decision = D(data).mean()
                bce = loss(decision, torch.ones(decision.size()).to(device))
                l += bce.item()
                total += data.size(0)
                bce.backward()

                g_optimizer.step()

        if e % 10 == 9 or e == 0:
            test(e)
        print("epoch %d: loss = %.6f" %(e, l/total))
        print("cost %.3f s" %(time.time()-starttime))

    torch.save({'epoch': e,
                'model_state_dict': G.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
                'loss': loss
                }, './'+name+'G.tar')
    torch.save({'epoch': e,
                'model_state_dict': D.state_dict(),
                'optimizer_state_dict': d_optimizer.state_dict(),
                'loss': loss
                }, './'+name+'D.tar')

def draw_background(D, x_min, x_max, y_min, y_max):
    xline = np.linspace(x_min, x_max, 100)
    yline = np.linspace(y_min, y_max, 100)
    bg = np.array([(x, y) for x in xline for y in yline])
    color = D(torch.Tensor(bg).to(device))
    color = (color - color.min())/(color.max() - color.min())
    cm = plt.cm.get_cmap('gray')
    sc = plt.scatter(bg[:, 0], bg[:, 1], c= np.squeeze(color.cpu().data), cmap=cm)
    # 显示颜色等级
    cb = plt.colorbar(sc)
    return cb


def test(epoch=0):
    with torch.no_grad():
        plt.clf()
        t = torch.rand([len(mSet), g_size])
        t = G(t)
        t = t.transpose(0, 1)
        t = torch.Tensor.cpu(t)

        draw_background(D, -0.5, 1.5, 0, 1)
        plt.xlim(-0.5, 1.5)
        plt.ylim(0, 1)

        mSet.show_plt()
        plt.scatter(t[0], t[1], label='g', c='b', alpha=0.2)

        plt.legend()
        # plt.savefig('./pic2/epoch'+str(epoch)+'.jpg')
        plt.show()



if __name__ == '__main__':
    g_size = 2
    hiddensize = 256
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G = Generator().to(device)
    D = Discriminator().to(device)
    # writer = SummaryWriter('/output/GAN/r1')
    lr = 1e-5
    epoch = 4000
    batchSize = 64
    mSet = matSet('./points.mat', 'xx')
    mLoader = torch.utils.data.dataloader.DataLoader(mSet, batch_size=batchSize, shuffle=True)
    # train(G, D, epoch, device)
    name = 'weight'
    pathG = './'+name+'G.tar'
    pathD = './'+name+'D.tar'
    checkpoint = torch.load(pathG, map_location=device)
    checkpoint2 = torch.load(pathD, map_location=device)
    G.load_state_dict(checkpoint['model_state_dict'])
    D.load_state_dict(checkpoint2['model_state_dict'])
    test()

