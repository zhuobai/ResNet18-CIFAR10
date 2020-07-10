'''
训练一个图像分类器
1.使用torchvision加载并且归一化CIFAR10的训练和测试数据集
2.定义一个卷积神经网络
3.定义一个损失函数
4.在训练样本数据上训练网络
5.在测试样本数据上测试网络
'''

import torch.nn as nn
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import resnet18
'''
torchvision数据集的输出是范围在[0,1]之间的PILImage
我们将他们转换成归一化范围为[-1,1]之间的张量Tensors
'''

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 第一步加载数据集
# 下载训练数据集
trainset = torchvision.datasets.CIFAR10('./data',train=True,download=False,transform=transform_train)

# 装载训练数据集
trainload = torch.utils.data.DataLoader(trainset,batch_size=100,
                                        shuffle=True,num_workers=2)

# 下载测试数据集
testset = torchvision.datasets.CIFAR10('./data',train=False,transform=transform_test,download=False)

# 装载测试数据集
testload = torch.utils.data.DataLoader(testset,batch_size=100,
                                       shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog',
           'horse','ship','truck')

# 第二部：定义一个卷积神经网络
net = resnet18()

# 第三步：定义一个损失函数和优化器
# 本次使用交叉熵Cross_Entropy作损失函数，优化器使用SGD
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
if __name__ == '__main__':
    # 第四步：数据训练
    for epoch in range(10):  # 多次循环数据集
        net.train()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for i,data in enumerate(trainload,0):
            # 获取输入
            inputs, labels = data

            # 把参数梯度归零
            optimizer.zero_grad()

            # 前向传播+反向传播+优化器
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()    # 优化器

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * len(trainload)), running_loss / (i + 1), 100. * correct / total))

    print('Finished Training')

    # 第五步：在测试样本数据上测试网络
    correct = 0 # 预测正确的样本数
    total = 0   # 总共的样本数
    net.eval()
    for data in testload:
        images,labels = data
        outputs = net(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))