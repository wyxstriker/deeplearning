import argparse

from tensorboardX import SummaryWriter
from torchvision.transforms.functional import to_pil_image

import ResNet
import SEResNet
import inResv1
import VGG11
from data_set import dogset
import torch
import os
import torch.utils.data.dataloader as dl
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time


def train(model, device, epoch):
    optimizer = optim.Adam(model.parameters())
    initepoch = 0
    path = './weights.tar'
    if os.path.exists(path) is not True:
        loss = nn.CrossEntropyLoss()
    else:
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initepoch = checkpoint['epoch']
        loss = checkpoint['loss']

    # for epoch in range(initepoch, epoch):  # loop over the dataset multiple times
    #     timestart = time.time()
    #
    #     running_loss = 0.0
    #     sum_loss = 0.0
    #     sum_total = 0
    #     for i, data in enumerate(trainload, 0):
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device),labels.to(device)
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         outputs = model(inputs)
    #         l = loss(outputs, labels)
    #         l.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += l.item()
    #         sum_loss += l.item()
    #         sum_total+= inputs.size(0)
    #
    #         nbatch = 25
    #         if i % nbatch == nbatch-1:  # print every 10 mini-batches
    #             print('[%d, %5d] loss: %.4f' %(epoch, i, running_loss / (nbatch*batchsize)))
    #             running_loss = 0.0
    #             _, predicted = torch.max(outputs.data, 1)
    #             total = labels.size(0)
    #             correct = (predicted == labels).sum().item()
    #             print('Accuracy of the network on the %d tran images: %.3f %%' % (total, 100.0 * correct / total))
    #
    #     writer.add_scalar('train_loss', sum_loss/sum_total, global_step=epoch)
    #     print('epoch %d cost %3f sec' %(epoch,time.time()-timestart))
    #     if epoch%2 == 0:
    #         model.eval()
    #         correct = 0
    #         total = 0
    #         l = 0
    #         with torch.no_grad():
    #             for data in verifyload:
    #                 images, labels = data
    #                 images, labels = images.to(device), labels.to(device)
    #                 outputs = model(images)
    #                 l += loss(outputs, labels)
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 total += labels.size(0)
    #                 correct += (predicted == labels).sum().item()
    #
    #         print('Accuracy of the network on the verify images: %.3f %%' % (100.0 * correct / total))
    #         print('Loss of the network on the verify images: %.3f' % (l/total))
    #         writer.add_scalar('verity_loss', l/total, global_step=epoch)
    #         writer.add_scalar('verity_accuracy', 100.0 * correct / total, global_step=epoch)
    #         model.train()
    #     #             torch.save({'epoch':epoch,
    #     #                         'model_state_dict':model.state_dict(),
    #     #                         'optimizer_state_dict':optimizer.state_dict(),
    #     #                         'loss':loss
    #     #                         },path)
    #     if epoch%10 == 9:
    #         torch.save({'epoch':epoch,
    #                     'model_state_dict':model.state_dict(),
    #                     'optimizer_state_dict':optimizer.state_dict(),
    #                     'loss':loss
    #                     },'weights'+str(epoch)+'.tar')
    #
    # print('Finished Training')


def test(model, device):
    model.eval()
    f = open('res.csv', 'w')
    with open('./data/sample_submission.csv') as g:
        f.write(g.readline())
    data_dir = os.listdir('./data/test')
    data_dir.sort()
    index = 0
    total_time = 0
    with torch.no_grad():
        for data in testload:
            images, _ = data
            images = images.to(device)
            timestart = time.time()
            outputs = model(images)
            total_time += time.time() - timestart
            outputs = torch.softmax(outputs, dim=1)
            for row in outputs:
                f.write(str(data_dir[index].split('.')[0]))
                index+=1
                for data in row:
                    f.write(','+str(data.item()))
                f.write('\n')
    f.close()
    print("time cost %.3f msec" %(1000*total_time/index))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c','--cpu', help='if use cpu', action='store_true')
    # args = parser.parse_args()
    # if not args.cpu:
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     print('gpu')
    # else:
    #     device = torch.device("cpu")
    #     print('cpu')
    #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trans = transforms.Compose([transforms.Resize((299, 299)),
                                transforms.ToTensor()])
    trans_train = transforms.Compose([transforms.RandomResizedCrop(299),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation((-30, 30)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])])
    trans_test = transforms.Compose([transforms.Resize((350,350)),
                                     transforms.CenterCrop(299),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])])
    batchsize = 32
    # trainload = dl.DataLoader(dogset('./data','train',trans_train), batch_size=batchsize, shuffle=True)
    # verifyload = dl.DataLoader(dogset('./data','verify',trans_test), batch_size=batchsize, shuffle=False)
    testload = dl.DataLoader(dogset('./data','test',trans_test), batch_size=batchsize, shuffle=False)

    writer = SummaryWriter('./run/inv1')

    # model = ResNet.ResNet18(num_classes=120).to(device)
    #     model = VGG11.VGGNet().to(device)
    #     model = SEResNet.SEResNet(num_classes=120).to(device)
    model = inResv1.InceptionResNetv1().to(device)

    epoch = 0

    train(model, device, epoch)
    test(model, device)