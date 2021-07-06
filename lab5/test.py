import torch
a = 2

def f():
    global a
    a = 1

if __name__ == '__main__':
    f()
    print(a)