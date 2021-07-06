import numpy as np
import pandas as pd
import torch
import torch.utils.data.dataset as dataset
import jieba
import re
import matplotlib.pyplot as plt


class TextSet(dataset.Dataset):
    def __init__(self, path, mode):
        super(TextSet, self).__init__()
        self.all_cats = ['书籍', '平板', '手机', '水果', '洗发水', '热水器', '蒙牛', '衣服', '计算机', '酒店']
        pd_all = pd.read_csv(path)
        pd_all = pd_all.dropna()
        #         self.dic = self.get_dic(pd_all)
        #         np.save('./word_dic.npy', self.dic)
        self.dic = np.load('./word_dic.npy', allow_pickle=True).item()
        pd_all['index'] = range(1, pd_all.shape[0] + 1)
        if mode == 'train':
            pd_all = pd_all[pd_all.index % 5 != 0]
            self.pd_all = pd_all[pd_all.index % 5 != 4]
        elif mode == 'verify':
            self.pd_all = pd_all[pd_all.index % 5 == 4]
        elif mode == 'test':
            self.pd_all = pd_all[pd_all.index % 5 == 0]
        else:
            self.pd_all = pd_all
        self.mode = mode
        self.pd_all = self.pd_all.reset_index()

    def get_dic(self, pd_all):
        word_all = set()
        self.numlist = []
        for text in pd_all.review:
            text = re.sub("[\r|\n|\\s!\"#$%&'()*+,-./:;《》<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·]+", "", text)
            cut = jieba.lcut(text)
            self.numlist.append(len(cut))
            for word in cut:
                word_all.add(word)
        word2id = {}
        for i, word in enumerate(word_all,start=1):
            word2id[word] = i
        word2id[''] = 0
        return word2id

    def __getitem__(self, item):
        text = self.pd_all.review[item]
        tag = self.all_cats.index(self.pd_all.cat[item])
        text = re.sub("[\r|\n|\\s!\"#$%&'()*+,-./:;《》<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·]+", "", text)
        cut = jieba.lcut(text)
        word_all = []
        for word in cut:
            word_all.append(self.dic[word])
        if len(word_all) <200:
            while len(word_all)!=200:
                word_all.append(0)
        else:
            word_all = word_all[:200]
        return torch.tensor(word_all), tag

    def __len__(self):
        return self.pd_all.shape[0]

    def word_num(self):
        return len(self.dic)


if __name__ == '__main__':
    set = TextSet('./data/online_shopping_10_cats.csv', 'train')
    print(len(set))
    set = TextSet('./data/online_shopping_10_cats.csv', 'verify')
    print(len(set))
    set = TextSet('./data/online_shopping_10_cats.csv', 'test')
    print(len(set))
    # print(set[23614])
    print(len(set.dic))
    # index = num_tokens.index(max(num_tokens))
    # print(set[index][0].__len__())
    # print(set[index][0])
    # print(set[213][0].__len__())
    # print(set[213])
    # plt.hist(num_tokens, bins=25)
    # plt.ylabel('number of tokens')
    # plt.xlabel('length of tokens')
    # plt.title('Distribution of tokens length')
    # plt.show()
