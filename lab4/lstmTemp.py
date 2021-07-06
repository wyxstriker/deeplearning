import os

import numpy as np
import torch
import torch.utils.data.dataloader as dataloader
from tensorboardX import SummaryWriter
from torch.nn import Parameter
import time
from data_load import TextSet
import torch.optim as optim
from temp_load import TempSet


class MyLSTM(torch.nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        # self.rnncell = MyRNNcell(256, 10)
        self.rnncell = torch.nn.LSTMCell(1, 10)

    def forward(self, x):
        x = x.transpose(1, 0)
        h = self.init_hidden()
        output = None
        for data in x:
            h = self.rnncell(data, h)
            if output is None:
                output = h[0].unsqueeze(0)
            else:
                output = torch.cat([output,h[0].unsqueeze(0)], dim=0)
        output = output.transpose(1, 0)
        return output, h[0]

    def init_hidden(self):
        return (torch.zeros(batchsize, 10).to(device), torch.zeros(batchsize, 10).to(device))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = MyLSTM()
        self.liner1 = torch.nn.Linear(7200,2048)
        self.relu = torch.nn.ReLU()
        self.liner2 = torch.nn.Linear(2048,256)
        self.liner3 = torch.nn.Linear(256,1)


    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        output, h = self.rnn(x)
        output = output.contiguous().view(-1, 7200)
        x = self.liner1(output)
        x = self.relu(x)
        x = self.liner2(x)
        x = self.relu(x)
        x = self.liner3(x)
        return x


def train(epoch, model, device):
    path = './weights299.tar'
    optimizer = optim.Adam(model.parameters())
    initepoch = 0
    if os.path.exists(path) is not True:
        loss = torch.nn.MSELoss()
    else:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initepoch = checkpoint['epoch']
        loss = checkpoint['loss']
    for i in range(initepoch, epoch):
        running_loss = 0.0
        sum_loss = 0.0
        sum_total = 0
        timestart = time.time()
        for k, data in enumerate(trainloader):
            optimizer.zero_grad()
            temps, tag = data
            if temps.size(0) != batchsize:
                continue
            temps, tag = temps.to(device), tag.to(device)
            tag = tag.view(-1,1)
            pre = model.forward(temps)
            l = loss(pre, tag).float()
            running_loss += l.item()
            sum_loss += l.item()
            sum_total += temps.size(0)
            l.backward()
            optimizer.step()
            nbatch = 25
            if k % nbatch == nbatch - 1:
                print('[%d, %5d] loss: %.6f' % (i, k, running_loss / (nbatch * batchsize)))
                running_loss = 0.0
                print('[%d, %d]' % (pre[0], tag[0]))
                # writer.add_scalar('pre_graph', pre[0])
                # writer2.add_scalar('pre_graph', tag[0])
        print('epoch %d cost %3f sec' % (i, time.time() - timestart))
        # writer.add_scalar('train_loss', sum_loss / sum_total, global_step=i)
        if epoch % 2 == 0:
            model.eval()
            l = 0
            total = 0
            meand = 0
            with torch.no_grad():
                for data in verifyloader:
                    temps, tag = data
                    if temps.size(0) != batchsize:
                        continue
                    temps, tag = temps.to(device), tag.to(device)
                    tag = tag.view(-1,1)
                    outputs = model.forward(temps)
                    l += loss(outputs, tag)
                    meand += torch.sum(outputs - tag)
                    total += temps.size(0)
            print('average miss of the network on the verify images: %.6f' % (meand / total))
            print('Loss of the network on the verify images: %.6f' % (l / total))
            # writer.add_scalar('verity_loss', l / total, global_step=i)
            model.train()
        # writer.flush()
        if i % 10 == 9:
            torch.save({'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, 'weights2'+str(i)+'.tar')


def test(model, device):
    total = 0
    l = 0
    loss = torch.nn.MSELoss()
    meand = []
    with torch.no_grad():
        for k, data in enumerate(testloader):
            temps, tag = data
            temps, tag = temps.to(device), tag.to(device)
            if temps.size(0) != batchsize:
                continue
            # displaytemps = temps
            # displaytag = tag
            outputs = torch.zeros(tag.size()).to(device)
            for h in range(tag.size(1)):
                input = torch.cat([temps[:, h:],tag[:,:h]], dim=1)
                outputs[:,h] = model.forward(input).view(-1)
                l += loss(outputs, tag)
            meand.extend((outputs-tag).tolist())
            #             meand += torch.sum(outputs - tag)
            total += temps.size(0)*tag.size(1)

        # for i in range(displaytemps[0].size(0)):
        #     writer.add_scalar('predict-true',displaytemps[1][i], global_step=i)
        #     writer.add_scalar('predict-true2',displaytemps[5][i], global_step=i)
        # for j in range(displaytag[0].size(0)):
        #     writer.add_scalar('predict-true',displaytag[1][j],global_step=temps[0].size(0)+j)
        #     writer2.add_scalar('predict-true',outputs[1][j],global_step=temps[0].size(0)+j)
        #     writer.add_scalar('predict-true2',displaytag[5][j],global_step=temps[0].size(0)+j)
        #     writer2.add_scalar('predict-true2',outputs[5][j],global_step=temps[0].size(0)+j)

    print('average miss of the network on the verify images: %.6f' % (np.mean(meand)))
    print('mid miss of the network on the verify images: %.6f' % (np.median(meand)))
    print('loss of the network on the all test images: %.3f ' % (l / total))


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batchsize = 32
    epoch = 80
    trainSet = TempSet('./data/jena_climate_2009_2016.csv', 'train')
    trainloader = dataloader.DataLoader(trainSet, batch_size=batchsize, shuffle=True)
    verifySet = TempSet('./data/jena_climate_2009_2016.csv', 'verify')
    verifyloader = dataloader.DataLoader(verifySet, batch_size=batchsize, shuffle=False)
    testSet = TempSet('./data/jena_climate_2009_2016.csv', 'test')
    testloader = dataloader.DataLoader(testSet, batch_size=batchsize, shuffle=False)
    #     writer = SummaryWriter('/output/temp1/picTrue3')
    # writer = SummaryWriter('/output/temp2/r1')
    # writer2 = SummaryWriter('/output/temp2/r2')
    #     writer2 = SummaryWriter('/output/temp1/picPredic3')
    model = Model()
    model.to(device)
    # train(epoch, model, device)
    checkpoint = torch.load('./weightsTemp.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(model, device)
    # writer.close()
