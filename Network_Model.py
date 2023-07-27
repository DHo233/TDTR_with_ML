import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
import torch
from torchvision import transforms

#搭建神经网络
class HeHe(nn.Module):
    def __init__(self):
        super(HeHe, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, padding = 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding = 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding = 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 20)
            )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class XKA(nn.Module):
    def __init__(self):
        super(XKA, self).__init__()
        self.network = nn.Sequential(
            Conv2d(3, 96, 11, stride = 4, padding = 5), #1
            MaxPool2d(kernel_size = 3, stride = 2), #2
            Conv2d(96, 256, 3, padding = 2), #3
            MaxPool2d(kernel_size = 3, stride = 2), #4
            Conv2d(256, 384, 3, padding = 1), #5
            #Conv2d(384, 384, 3, padding = 1), #6
            Conv2d(384, 256, 3, padding = 1), #7
            MaxPool2d(kernel_size = 3, stride = 2), #8
            Flatten(),
            Linear(9216, 4096),
            #Linear(4096, 4096),
            Linear(4096, 20),
            nn.Softmax(dim = 1)
            )
        
    def forward(self, x):
        x = self.network(x)
        return x    
    
    
# if __name__ == '__main__':
#     hehe = HeHe()
#     input = torch.ones((64, 3, 32, 32))
#     output = hehe(input)
#     print(output.shape)