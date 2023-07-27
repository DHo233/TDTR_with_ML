import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from Network_Model import *
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets


train_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])
test_dir = "D:/Courses/SRTP/coil-20-proc/coil-20-proc/data/train_set"
train_dir = "D:/Courses/SRTP/coil-20-proc/coil-20-proc/data/test_set"

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(train_dir, transform=train_transforms)



train_data_size = len(train_data)
test_data_size = len(test_data)

#print("训练集：{}".format(train_data_size))
#print("测试集：{}".format(test_data_size))


train_dataloader = DataLoader(train_data, batch_size = 2)
test_dataloader = DataLoader(test_data, batch_size = 64)
print(train_dataloader)

#创建网络模型
hehe = HeHe()

#损失函数
loss_fn = nn.CrossEntropyLoss()


#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(hehe.parameters(), lr = learning_rate)

#设置训练网络的参数
#记录训练的次数
total_train_step = 0
#测试的次数
total_test_step = 0
#训练的轮数
epoch = 200

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i + 1))
    for data in train_dataloader:
        imgs, targets = data
        #print(imgs.shape)
        #print(imgs, targets)
        outputs = hehe(imgs)
        loss = loss_fn(outputs, targets)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(loss)
        total_train_step = total_train_step + 1
        



    #测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = hehe(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
        print("Loss：{}".format(total_test_loss))
        print("正确率：{}".format(total_accuracy / test_data_size))
        
        torch.save(hehe, "hehe_{}.pth".format(i))
        print("模型已保存")


