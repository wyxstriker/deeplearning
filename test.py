import torch.nn as nn
import torch


class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.w = torch.Tensor([1])
        self.w.requires_grad = True

    def forward(self, x):
        return torch.mean(x*self.w)


if __name__ == '__main__':
    x = torch.tensor([1,2,3,4])
    model = test()
    optim = torch.optim.SGD([model.w], lr=1)
    loss = model(x)
    print(loss)
    loss.backward()
    print(model.w.grad)
    optim.step()
    print(model.w)
