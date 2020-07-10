# 使用torchvision加载CIFaR10超级简单
import torch
import torchvision
import torchvision.transforms as transforms   #用于数据预处理模块

'''
torchvision 数据集加载完后得输出是范围在[0,1]之间的PILImage。我们将其标准化为
范围在[-1,1]之间的张量
torchvision.datasets   数据模块
torchvision.models     模型模块
torchvision.transforms 数据变换模块
torchvision.utils      工具
'''
#前面的（0.5，0.5，0.5） 是 R G B 三个通道上的均值， 后面(0.5, 0.5, 0.5)是三个通道的标准差
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# 下载训练集
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform = transform)
# 装载训练集
'''
数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器
dataset:就是数据的来源，比如训练集添入我们定义的trainset
batch_size:每批次进入多少数据，本例中填的是4
shuffle:如果为真，就打乱数据的顺序，本例为True
num_workers:用多少个子进程加载数据。0表示数据将在主进程中加载（默认：0）
本例中为2，你选的用来加载数据的子进程越多，那么显然数据读的就越快，这样的话消耗cpu的资源也就越多
一般呢，既不要让花在加载数据上的时间太多，也不要占用太多电脑资源
'''
train_loader = torch.utils.data.DataLoader(trainset,batch_size = 4,shuffle=True,num_workers=0)

# 下载测试集
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform = transform)
# 装载测试集
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=0)


classes = ('plane','car','bird','cat',
           'deer','dog','frog','horse','ship','truck')



# 定义卷积神经网络

# 首先是调用Variable、torch.nn、torch.nn.functional
from torch.autograd import Variable  # 这一步还没有显示用到Variable,但是现在写在这里也没有问题，后面会用到
import torch.nn as nn    #神经网路工具箱
import torch.nn.functional as F  #神经网络函数
'''
torch.nn中大多数layer在torch.nn.functional中都有一个与之对应的函数。
区别在于：torch.nn.Module中实现layer的都是一个特殊的类
nn.functional中的函数，更像是纯函数，由def function()定义
如果模型有可学习的参数，最好使用nn.Module对应的相关layer，否则二者都可以使用
比如此例中的Relu其实没有可学习的参数，只是进行一个运算而已，所以就使用的是functional中的Relu函数
而卷积层和全连接层都有可学习的参数，所以用的是nn.Module中的类

不具备可学习参数的层，将他们用函数代替，这样可以不用放在构造函数中进行初始化

定义网络模型，主要会用到的就是torch.nn和torch.nn.functional这两个模块
'''

#nn.Module是所有神经网络的基类，我们自己定义任何神经网络，都要继承nn.Module类
class Net(nn.Module):               #我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):
        super(Net,self).__init__()   #第二、三行都是python类继承的基本操作，此写法应该是python2.7的继承格式，但python3这样写也可以

        #Conv2d(3,6,5)的意思是说，输入是3通道的图像，输出是20通道，也就是20个卷积核
        #卷积核是5*5，其余参数都是用的默认值
        self.conv1 = nn.Conv2d(3,6,5)   #添加第一个卷积层，调用了nn里面的Conv2d()


        self.pool = nn.MaxPool2d(2,2)   #最大池化层
        self.conv2 = nn.Conv2d(6,16,5)  #同样是卷积层
        self.fc1 = nn.Linear(16*5*5,120)    # 接着三个全连接层
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):                #这里定义前向传播的方法，为什么反向传播的方法呢？这其实就涉及到torch.autograd模块了
                                        #但说实话，这部分网络定义的部分还没有用到autorgrad的知识，所以后面用到了再讲
         x = self.pool(F.relu(self.conv1(x)))      # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
         x = self.pool(F.relu(self.conv2(x)))
         x = x.view(-1,16*5*5)
         '''
         .view()是一个tensor的方法，使得tensor改变size（即维数）但是元素的总数是不变的
         第一个参数-1是说这个参数由另一个参数确定，比如矩阵在元素总数一定的情况下，确定列数就能确定行数
         那么为什么这里只关心列数不关心行数呢，因为马上就要进入全连接层了，而全连接层说白了就是矩阵乘法
         你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
         
         '''
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = self.fc3(x)
         return x
# 和python中一样，类定义完之后实例化就很简单了，这里实例化一个net
net = Net()

#定义损失函数和优化器
'''
1.优化器：pytorch将深度学习中常用的优化方法全部封装在torch.optim之中，所有的优化方法都是继承基类optim.optimizier

2.损失函数
损失函数是封装在神经网络工具箱nn中的，包含很多损失函数
此例中用到的是交叉熵损失 criterion = nn.CrossEntropyLoss()


'''

import torch.optim as optim         #导入torch.optim模块

criterion = nn.CrossEntropyLoss()   #同样是用到了神经网络工具箱 nn 中的交叉熵函数
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9) #optim模块中的SGD梯度优化方式---随机梯度下降


for epoch in range(2):              # loop over the dataset multiple times 指定训练一共要循环几个epoch

    running_loss = 0.0      #定义一个变量方便我们对loss进行输出
    for i,data in enumerate(train_loader,0): #这里我们遇到了第一步中出现的trainloader,代码传入数据
                                             # enumerate是python的内置函数，即获得索引也获得数据
       # get the inputs
        inputs,labels = data        # data是从enumerate返回的data,包含数据和标签信息，分别赋值给inputs和labels

        # wrap them in Variable
        inputs,labels = Variable(inputs),Variable(labels) #将数据转换成Variable,第二步里面我们已经引入这个模块

        # zero the parameter gradients
        optimizer.zero_grad()          # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度

        # forward + backward + optimize
        outputs = net(inputs)           #把数据输入网络net,这个net()在第二步的代码最后我们已经定义了
        loss = criterion(outputs,labels)  # 计算损失值，criterion我们在第三步里面定义了
        loss.backward()      # loss进行反向传播
        optimizer.step()     # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮

        # print statistics   #这几行代码不是必须的，为了打印出loss方便我们看而已，不影响训练过程
        running_loss += loss.item()    #从下面一行代码可以看出它是每循环0-1999共两千次才打印一次
        if i % 2000 == 1999:    #print every 2000 mini-batches 所以每个2000次之类先用running_loss进行累加
            print('[%d,%5d] loss: %.3f' %
                  (epoch + 1 , i+1 , running_loss / 2000))  # 然后再除以2000，就得到这个两千次的平均损失值

            running_loss = 0.0      # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用

       # print('Finished Training')


#测试

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))

dataiter = iter(testloader)  # 创建一个python迭代器，读入的是我们第一步里面就已经加载好的testloader
images,labels = dataiter.next()   # 返回一个batch_size的图片，根据第一步的设置，应该是4张

#print images
imshow(torchvision.utils.make_grid(images))  # 展示这四张图片
print('GroundTruth: ',' '.join('%5s' % classes[labels[j]] for j in range(4)))  # python字符串格式化 ''.join表示用空格来连接后面的字符串，参考python的join（）方法

outputs = net(Variable(images))   #注意这里的images是我们从上面获得的那四张图片，所以首先要转化成variable
_,predicted = torch.max(outputs.data,1)
'''
这个_,predicted 是python的一种常用的写法，表示后面的函数其实
会返回两个值
但是我们对第一个值不感兴趣，就写一个_在那里，把它赋值给_就好，我们只关心第二个值predicted
比如_,a = 1,2 这中赋值语句在python中是可以通过的，你只关心后面的等式中的第二个位置的值是多少

'''

print('Predicted:',' '.join('%5s' % classes[predicted[j]] for j in range(4))) #python的字符串格式化

correct = 0  # 定义预测正确的图片数，初始化为0
total = 0    # 总共参与测试的图片数，也初始化为0
for data in testloader: # 循环每一个batch
    images,labels = data
    outputs = net(Variable(images))   #输入网络进行测试
    _,predicted = torch.max(outputs.data,1)
    total += labels.size(0)     #更新测试图片的数量
    correct += (predicted == labels).sum()  #更新正确分类的图片的数量

print('Accuracy of the network on the 10000 test images: %d %%' % (
      100*correct / total))  #打印最后的结果

class_correct = list(0. for i in range(10)) #定义一个存储每类中测试正确的个数的列表，初始化为0
class_total = list(0. for  i in range(10))   # 定义一个存储每类中测试总数的个数的列表，初始化为0
for data in testloader:     #以一个batch为单位进行循环
    images,labels = data
    outputs = net(Variable(images))
    _,predicted = torch.max(outputs.data,1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label]  += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' %(
        classes[i],100 * class_correct[i] / class_total[i]))
