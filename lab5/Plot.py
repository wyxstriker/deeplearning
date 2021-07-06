import matplotlib.pyplot as plt
import numpy as np
import torch

# 使用 cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_scatter(data, color, x_min, x_max, y_min, y_max):
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.scatter(data[:, 0], data[:, 1], c=color, s=10)


# 画背景
def draw_background(D, x_min, x_max, y_min, y_max):
    i = x_min
    bg = []
    while i <= x_max - 0.01:
        j = y_min
        while j <= y_max - 0.01:
            bg.append([i, j])
            j += 0.01
        bg.append([i, y_max])
        i += 0.01
    j = y_min
    while j <= y_max - 0.01:
        bg.append([i, j])
        j += 0.01
        bg.append([i, y_max])
    bg.append([x_max, y_max])
    color = D(torch.Tensor(bg).to(device))
    print(color)
    bg = np.array(bg)
    cm = plt.cm.get_cmap('gray')
    sc = plt.scatter(bg[:, 0], bg[:, 1], c= np.squeeze(color.cpu().data), cmap=cm)
    # 显示颜色等级
    cb = plt.colorbar(sc)
    return cb


