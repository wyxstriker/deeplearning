import numpy as np
import torch
import torch.utils.data.dataset as dataset
import pandas as pd
import re

class TempSet(dataset.Dataset):
    def __init__(self, path, mode):
        super(TempSet, self).__init__()
        pd_all = pd.read_csv(path)[['Date Time','T (degC)']]
        pd_all.dropna()
        for k, datetime in enumerate(pd_all['Date Time']):
            if re.search('2015', datetime):
                break
        self.mode = mode
        if mode == 'train':
            self.pd_all = pd_all.loc[:k-1]
            self.window_size = 6*24*5
        elif mode == 'verify':
            self.pd_all = pd_all.loc[k:]
            self.window_size = 6*24*5
        elif mode == 'test':
            self.pd_all = pd_all.loc[k:]
            self.window_size = 6*24*7
        self.pd_all = self.pd_all.reset_index()



    def __getitem__(self, item):
        if self.mode == 'test':
            base = item * self.window_size
            res = torch.tensor(self.pd_all['T (degC)'].loc[base:base+6*24*5-1].tolist())
            tem = torch.tensor(self.pd_all['T (degC)'].loc[base+6*24*5:base+self.window_size-1].tolist())
        else:
            base = item
            res = torch.tensor(self.pd_all['T (degC)'].loc[base:base+self.window_size-1].tolist())
            tem = torch.tensor(self.pd_all['T (degC)'].loc[base+self.window_size]).float()
        return res, tem


    def __len__(self):
        if self.mode == 'test':
            return self.pd_all.shape[0] // self.window_size
        else:
            return self.pd_all.shape[0] - self.window_size

if __name__ == '__main__':
    tSet = TempSet('./data/jena_climate_2009_2016.csv', 'test')
    x, y = tSet[0]
    print(x.size())
    print(y.size())