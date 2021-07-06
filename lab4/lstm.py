import os

import numpy as np
import torch
import torch.utils.data.dataloader as dataloader
from tensorboardX import SummaryWriter
from torch.nn import Parameter
import time
from data_load import TextSet
import torch.optim as optim


# class MyRNNcell(torch.nn.Module):
#     def __init__(self, inputsize, hidesize):
#         super(MyRNNcell, self).__init__()
#         self.w = Parameter(torch.zeros([inputsize + hidesize, 10]))
#         self.b = Parameter(torch.zeros([10]))
#
#     def forward(self, x, h):
#         X = torch.cat([x, h], dim=1)
#         h = torch.mm(X, self.w) + self.b
#         return h


class MyLSTM(torch.nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        # self.rnncell = MyRNNcell(256, 10)
        self.rnncell = torch.nn.LSTMCell(64, 10)

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
        self.emb = torch.nn.Embedding(word_num, 64)
        self.rnn = MyLSTM()
        self.liner1 = torch.nn.Linear(2000,1024)
        self.liner2 = torch.nn.Linear(1024,10)


    def forward(self, x):
        x = self.emb(x)
        output, h = self.rnn(x)
        output = output.contiguous().view(-1, 2000)
        x = self.liner1(output)
        x = self.liner2(x)
        return x


def train(epoch, model, device):
    path = './weightslstm.tar'
    optimizer = optim.Adam(model.parameters())
    initepoch = 0
    if os.path.exists(path) is not True:
        loss = torch.nn.CrossEntropyLoss()
    else:
        checkpoint = torch.load(path)
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
            text, tag = data
            if text.size(0) != batchsize:
                continue
            text, tag = text.to(device), tag.to(device)
            pre = model.forward(text)
            l = loss(pre, tag)
            running_loss += l.item()
            sum_loss += l.item()
            sum_total += text.size(0)
            l.backward()
            optimizer.step()
            nbatch = 25
            if k % nbatch == nbatch - 1:
                print('[%d, %5d] loss: %.6f' % (i, k, running_loss / (nbatch * batchsize)))
                running_loss = 0.0
                _, predicted = torch.max(pre.data, 1)
                total = text.size(0)
                correct = (predicted == tag).sum().item()
                print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * correct / total))
        print('epoch %d cost %3f sec' % (i, time.time() - timestart))
        # writer.add_scalar('train_loss', sum_loss / sum_total, global_step=i)
        if epoch % 2 == 0:
            model.eval()
            correct = 0
            total = 0
            l = 0
            with torch.no_grad():
                for data in verifyloader:
                    text, tag = data
                    if text.size(0) != batchsize:
                        continue
                    text, tag = text.to(device), tag.to(device)
                    outputs = model.forward(text)
                    l += loss(outputs, tag)
                    _, predicted = torch.max(outputs.data, 1)
                    total += text.size(0)
                    correct += (predicted == tag).sum().item()

            print('Accuracy of the network on the verify images: %.3f %%' % (100.0 * correct / total))
            print('Loss of the network on the verify images: %.6f' % (l / total))
            # writer.add_scalar('verity_loss', l / total, global_step=i)
            # writer.add_scalar('verity_accuracy', 100.0 * correct / total, global_step=i)
            model.train()
        # writer.flush()
        if i % 10 == 9:
            torch.save({'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, 'weightslstm'+str(i)+'.tar')


def test(model, device):
    mm = np.zeros([10,10])
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for k, data in enumerate(testloader):
            text, tag = data
            text, tag = text.to(device), tag.to(device)
            if text.size(0) != batchsize:
                continue
            outputs = model.forward(text)
            _, predicted = torch.max(outputs.data, 1)
            total += tag.size(0)
            correct += (predicted == tag).sum().item()
            tag = tag.cpu()
            predicted = predicted.cpu()
            for n in range(text.size(0)):
                mm[tag[n]][predicted[n]] += 1
    recall = []
    precision = []
    for c in range(10):
        precision.append(mm[c][c]/np.sum(mm[:,c]))
        recall.append(mm[c][c]/np.sum(mm[c,:]))
    recall = np.mean(recall)
    precision = np.mean(precision)
    f = 2/(1/recall+1/precision)
    print('Accuracy of the network on the all test images: %.3f %%' % (
            100.0 * correct / total))
    print('Precision of the network on the all test images: %.3f %%' % (
            100.0 * precision))
    print('Recall of the network on the all test images: %.3f %%' % (
            100.0 * recall))
    print('F of the network on the all test images: %.3f' % (
        f))


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batchsize = 128
    epoch = 50
    trainSet = TextSet('./data/online_shopping_10_cats.csv', 'train')
    word_num = trainSet.word_num()
    trainloader = dataloader.DataLoader(trainSet, batch_size=batchsize, shuffle=True)
    testSet = TextSet('./data/online_shopping_10_cats.csv', 'test')
    testloader = dataloader.DataLoader(testSet, batch_size=batchsize, shuffle=False)
    verifySet = TextSet('./data/online_shopping_10_cats.csv', 'verify')
    verifyloader = dataloader.DataLoader(verifySet, batch_size=batchsize, shuffle=False)
    # writer = SummaryWriter('/output/lstm1')
    model = Model()
    model.to(device)
    #     train(epoch, model, device)
    checkpoint = torch.load('./weightsLSTM.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(model, device)
    # writer.close()
