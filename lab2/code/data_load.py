import math
import os

import torch
import torch.utils.data.dataset as ds
import torch.utils.data.dataloader as dl
import PIL.Image as Image
from torchvision import transforms


class CaltechSet(ds.Dataset):
    def __init__(self, path, mode):
        self.path = path
        dirlist = os.listdir(path)
        try:
            dirlist.remove('BACKGROUND_Google')
        except:
            pass
        dirlist.sort()
        self.num = [len(os.listdir(self.path+'/'+x)) for x in dirlist if os.path.isdir(self.path+'/'+x)]
        self.classlist = [x for x in dirlist if os.path.isdir(self.path+'/'+x)]
        self.mode = mode
        self.trans = transforms.Compose([transforms.Resize((64, 64)),
                                         transforms.ToTensor()])
        s = 0
        e = 1
        if mode == 'train':
            s = 0
            e = 0.8
        elif mode == 'verity':
            s = 0.8
            e = 0.9
        elif mode == 'test':
            s = 0.9
            e = 1

        self.startindex = [math.floor(x*s) for x in self.num]
        self.endindex = [math.floor(x*e) for x in self.num]
        self.num = [self.endindex[i]-self.startindex[i] for i in range(len(self.num))]
        self.pdf = self.num.copy()
        temp = self.num[1:]
        for i in range(1, len(self.pdf)):
            self.pdf[i] = self.pdf[i-1] + temp[i-1]

    def __getitem__(self, item):
        tag = -1
        real_tag = item
        out_flag = True
        for tag in self.pdf:
            if item < tag:
                out_flag = False
                break
            else:
                real_tag = item - tag
        if out_flag:
            return None, None
        class_num = self.pdf.index(tag)
        path = self.path+'/'+self.classlist[class_num]
        res = os.listdir(path)
        res.sort()
        path += '/'+res[real_tag+self.startindex[class_num]]
        pic = Image.open(path)
        if pic.mode == 'L':
            pic = Image.merge("RGB",[pic, pic, pic])
        pic = self.trans(pic)
        return pic, class_num

    def __len__(self):
        return sum(self.num)


if __name__ == '__main__':
    path = '../data/caltech'
    # 349
    c = CaltechSet(path, 'a')
