import math

import torch.utils.data.dataset as dataset
import os
import torchvision.transforms as transforms
import PIL.Image as Image


class dogset(dataset.Dataset):
    def __init__(self, path, mode, trans = transforms.Compose([transforms.Resize((64, 64)),
                                                               transforms.ToTensor()])):
        self.tag_dir = {}
        if mode == 'train':
            self.data_path = path + '/train'
            self.data_dir = os.listdir(self.data_path)
            self.data_dir.sort()
            temp = math.ceil(0.9*len(self.data_dir))
            self.data_dir = self.data_dir[:temp]
            with open(path+'/sample_submission.csv') as f:
                self.class_list  = f.readline()[:-1].split(',')[1:]
            with open(path+'/labels.csv', 'r') as f:
                f.readline()
                content = f.readline().strip('\n')
                while content:
                    temp = content.split(',')
                    self.tag_dir[temp[0]] = temp[1]
                    content = f.readline().strip('\n')

        elif mode == 'verify':
            self.data_path = path + '/train'
            self.data_dir = os.listdir(self.data_path)
            self.data_dir.sort()
            temp = math.ceil(0.9*len(self.data_dir))
            self.data_dir = self.data_dir[temp:]
            with open(path+'/sample_submission.csv') as f:
                self.class_list  = f.readline()[:-1].split(',')[1:]
            with open(path+'/labels.csv', 'r') as f:
                f.readline()
                content = f.readline().strip('\n')
                while content:
                    temp = content.split(',')
                    self.tag_dir[temp[0]] = temp[1]
                    content = f.readline().strip('\n')
        else:
            self.data_path = path + '/test'
            self.data_dir = os.listdir(self.data_path)
            self.data_dir.sort()

        self.trans = trans


    def __getitem__(self, item):
        pathname = self.data_dir[item]
        pic = Image.open(self.data_path+'/'+pathname)
        if not bool(self.tag_dir):
            return self.trans(pic), 0
        tag = self.tag_dir[pathname.split('.')[0]]
        return self.trans(pic), self.class_list.index(tag)

    def __len__(self):
        return self.data_dir.__len__()


if __name__ == '__main__':
    train = dogset('./data', 'train')
    print(len(train))
    test = dogset('./data', 'test')
    print(len(test))
    verify = dogset('./data', 'verify')
    print(len(verify))