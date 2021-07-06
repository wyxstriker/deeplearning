import torch
import torch.utils.data.dataset as dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dl


class PicSet(dataset.Dataset):
    def __init__(self, path, mode):
        super(PicSet, self).__init__()
        self.path = path + '/' + mode
        self.mode = mode
        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
        ])
        if mode == 'train':
            self.start = 1
            self.end = 14000
        #             self.end = 20000
        #         elif mode == 'verify':
        elif mode == 'validation':
            self.start = 14001
            self.end = 15000
        elif mode == 'mytest':
            self.start = 15001
            self.end = 20000

    def __getitem__(self, item):
        index = item // 2
        pic1 = self.trans(Image.open('./data/'+str(self.mode)+'/cover/'+str(index+self.start)+'.pgm'))
        pic2 = self.trans(Image.open(self.path+'/stego/'+str(index+self.start)+'.pgm'))
        flag = torch.tensor([[0], [1]])
        pic = torch.stack([pic1, pic2], dim=0)
        return pic, flag


    def __len__(self):
        return (self.end - self.start + 1)


if __name__ == '__main__':
    s = PicSet('./data', 'test')
    d = dl.DataLoader(s, shuffle=True, batch_size = 16)
    for data in d:
        x, y = data
        x = x.view(x.size(0)*2, 256, 256)
        y = y.view(y.size(0)*2, 1)
        print(x.size())
        print(y.size())
        print(y)
        exit()