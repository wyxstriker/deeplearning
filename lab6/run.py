import time

import torch
from data_load import PicSet
import torch.utils.data.dataloader as dl
from SRNet import SRNet
import os
from tensorboardX import SummaryWriter


def train(epochall, device, model):
    path = './SRNetSUNI04.tar'
    initepoch = 0
    #     optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-4)
    if os.path.exists(path)  is not True:
        loss = torch.nn.CrossEntropyLoss()
    else:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initepoch = checkpoint['epoch']
        loss = checkpoint['loss']

    for epoch in range(initepoch, epochall):
        time_s = time.time()

        running_loss = 0.0
        sum_loss = 0.0
        sum_total = 0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            inputs = inputs.view(inputs.size(0)*2, 1, 256, 256)
            labels = labels.view(labels.size(0)*2)
            inputs, labels = inputs.to(device),labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            l = loss(outputs, labels)

            regularization_loss = 0
            for m in model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    regularization_loss += torch.sum(torch.abs(m.weight))
            l += 2*1e-6 * regularization_loss
            l.backward()
            optimizer.step()

            # print statistics
            running_loss += l.item()
            sum_loss += l.item()
            sum_total+= inputs.size(0)

            nbatch = 100
            if i % nbatch == nbatch-1:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.4f' %(epoch, i, running_loss / (nbatch*bs)))
                running_loss = 0.0
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * correct / total))

        writer.add_scalar('train_loss', sum_loss/sum_total, global_step=epoch)
        print('epoch %d cost %3f sec' %(epoch,time.time()-time_s))
        if epoch%2 == 0:
            model.eval()
            correct = 0
            total = 0
            l = 0
            with torch.no_grad():
                for data in verifyLoader:
                    images, labels = data
                    images = images.view(images.size(0)*2, 1, 256, 256)
                    labels = labels.view(labels.size(0)*2)
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    l += loss(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the verify images: %.3f %%' % (100.0 * correct / total))
            print('Loss of the network on the verify images: %.4f' % (l/total))
            writer.add_scalar('verity_loss', l/total, global_step=epoch)
            writer.add_scalar('verity_accuracy', 100.0 * correct / total, global_step=epoch)
            test(device, model)
            model.train()

        if epoch%10 == 9:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, './SRNetSUNI04'+str(epoch)+'.tar')

    print('Finished Training')

def test(device, model):
    total = 0
    correct = 0
    model.eval()
    C = 0
    S = 0
    P = 0
    N = 0
    with torch.no_grad():
        for k, data in enumerate(testLoader):
            text, tag = data
            text = text.view(text.size(0)*2, 1, 256, 256)
            tag = tag.view(tag.size(0)*2)
            text, tag = text.to(device), tag.to(device)
            outputs = model(text)
            _, predicted = torch.max(outputs.data, 1)
            total += tag.size(0)
            correct += (predicted == tag).sum().item()
            tag = tag.cpu()
            predicted = predicted.cpu()
            for i in range(tag.size(0)):
                if tag[i] == 0:
                    C += 1
                    if tag[i] == predicted[i]:
                        N += 1
                else:
                    S += 1
                    if tag[i] == predicted[i]:
                        P += 1

    print('Accuracy of the network on the all test images: %.3f %%' % (
            100.0 * (P+N) / (C+S)))
    print('FA of the network on the all test images: %.3f %%' % (
            100.0 * (C-N)/C))
    print('MD of the network on the all test images: %.3f %%' % (
            100.0 * (S-P)/S))
    model.train()


if __name__ == '__main__':
    bs = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datapath = './data_hugo10/6.HUGO_1'
    #     datapath = './data'
    trainSet = PicSet(datapath, 'train')
    trainLoader = dl.DataLoader(trainSet, shuffle=True, batch_size=bs)
    #     verifySet = PicSet(datapath, 'verify')
    verifySet = PicSet(datapath, 'validation')
    verifyLoader = dl.DataLoader(verifySet, shuffle=False, batch_size=bs)
    testSet = PicSet(datapath, 'mytest')
    #     testSet = PicSet(datapath, 'train')
    testLoader = dl.DataLoader(testSet, shuffle=False, batch_size=bs)
    model = SRNet()
    model = model.to(device)
    #     for m in model.modules():
    #         if isinstance(m, torch.nn.Conv2d):
    #             torch.nn.init.kaiming_normal_(m.weight.data)
    #             torch.nn.init.constant_(m.bias.data, 0.2)
    #         elif isinstance(m, torch.nn.Linear):
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()
    writer = SummaryWriter('/output/runMAX7')
    epoch = 200
    #     train(epoch, device, model)
    checkpoint = torch.load('./SRNet399.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(device, model)