import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import torch.utils.data.dataset


class matSet(torch.utils.data.dataset.Dataset):
    def __init__(self, path, mode):
        super(matSet, self).__init__()
        self.mat = scio.loadmat(path)[mode]
        self.mode = mode


    def __getitem__(self, item):
        return torch.tensor(self.mat[item], dtype=torch.float)

    def __len__(self):
        return self.mat.shape[0]

    def show_plt(self):
        plt.scatter(self.mat.T[0], self.mat.T[1], label=self.mode, alpha=0.2, c='r')


if __name__ == '__main__':
    pointsA = matSet('./points.mat', 'a')
    pointsB = matSet('./points.mat', 'b')
    pointsC = matSet('./points.mat', 'c')
    pointsD = matSet('./points.mat', 'd')
    pointsXX = matSet('./points.mat', 'xx')
    # xx 8192
    #
    # print(points.mat.shape)
    # points.mat = points.mat.T
    # print(points.mat.shape)
    # plt.plot(points.mat[0], points.mat[1], '.')
    # for x in range(len(points)):
    #     plt.scatter(points[x][0], points[x][1])
    # pointsA.show_plt()
    # pointsB.show_plt()
    # pointsC.show_plt()
    # pointsD.show_plt()
    pointsXX.show_plt()
    plt.legend()
    plt.show()

