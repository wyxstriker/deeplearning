import os
import time

import torch
import torch.nn as nn
import torch.utils.data.dataloader as dl
import torch.optim as optim
import data_load
from tensorboardX import SummaryWriter

class my_AlexNet(nn.Module):
    def __init__(self):
        super(my_AlexNet, self).__init__()
        self.l1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=2),
                                nn.BatchNorm2d(96),
                                nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=3, stride=2)
                                )
        self.l2 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=2),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2)
                                )
        self.l3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=2),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=2),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2),
                                nn.Dropout(0.5),
                                nn.Flatten(start_dim=1)
                                )
        self.l4 = nn.Sequential(nn.Linear(4096, 4096),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(4096, 4096),
                                nn.ReLU(),
                                nn.Linear(4096, 101)
                                )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x

    def my_train(self, device, epoch):
        optimizer = optim.Adam(self.parameters())
        # optimizer = optim.SGD(self.parameters(), lr=0.01)

        path = 'weights6.tar'
        initepoch = 0

        if os.path.exists(path) is not True:
            loss = nn.CrossEntropyLoss()

        else:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initepoch = checkpoint['epoch']
            loss = checkpoint['loss']

        for epoch in range(initepoch, epoch):  # loop over the dataset multiple times
            timestart = time.time()

            running_loss = 0.0
            total = 0
            correct = 0
            sum_loss = 0.0
            sum_total = 0
            for i, data in enumerate(trainSet, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device),labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                # print statistics
                running_loss += l.item()
                sum_loss += l.item()
                sum_total+= 1
                # print("i ",i)
                if i % 5 == 0:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.4f' %(epoch, i, running_loss / 500))
                    running_loss = 0.0
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * correct / total))
                    total = 0
                    correct = 0
                    torch.save({'epoch':epoch,
                            'model_state_dict':self.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'loss':loss
                            },path)
            writer.add_scalar('train_loss', sum_loss/sum_total, global_step=epoch)
            print('epoch %d cost %3f sec' %(epoch,time.time()-timestart))
            if epoch%5 == 4:
                self.eval()
                correct = 0
                total = 0
                l = 0
                with torch.no_grad():
                    for data in veritySet:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = self(images)
                        l += loss(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the verify images: %.3f %%' % (100.0 * correct / total))
                print('Loss of the network on the verify images: %.3f' % (l/total))
                writer.add_scalar('verity_loss', l/total, global_step=epoch)
                self.train()

        print('Finished Training')

    def my_test(self, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testSet:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the all test images: %.3f %%' % (
                100.0 * correct / total))


if __name__ == '__main__':
    model = my_AlexNet()
    path = '../data/caltech'
    batchSize = 64
    trainSet = dl.DataLoader(data_load.CaltechSet(path, 'train'), batch_size=batchSize, shuffle=True)
    veritySet = dl.DataLoader(data_load.CaltechSet(path, 'test'), batch_size=batchSize, shuffle=False)
    testSet = dl.DataLoader(data_load.CaltechSet(path, 'verity'), batch_size=batchSize, shuffle=False)
    # use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = 50
    writer = SummaryWriter('./run6/')
    model = model.to(device)
    model.train()
    model.my_train(device, epoch)
    writer.close()
    model.eval()
    model.my_test(device)
    # input = torch.rand(1, 3, 64, 64)
    # writer.add_graph(model, input)
