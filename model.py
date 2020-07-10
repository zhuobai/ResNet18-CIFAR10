import torch
import torch.nn as nn

# 18层的残差网络
class BasicBlock(nn.Module):
    expansion = 1   #用来当处于conv3_x,conv4_x,conv5_x,捷径的深度扩大
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                               kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                               kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        # 下采样结构
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差网络的不同层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    '''
    block是残差结构，
    blocks_num是每层有多少个循环，比如34层残差结构，blocks_num=[3,4,6,3]

    '''
    def __init__(self,block,blocks_num,num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 64
        # 刚输入数据时第一个卷积网络
        self.conv1 = nn.Conv2d(3,self.in_channel,kernel_size=3,stride=2,
                               padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()

        # 往下四层的残差网络
        self.layer1 = self._make_layer(block,64,blocks_num[0])
        self.layer2 = self._make_layer(block,128,blocks_num[1],stride=2)
        self.layer3 = self._make_layer(block,256,blocks_num[2],stride=2)
        self.layer4 = self._make_layer(block,512,blocks_num[3],stride=2)

        # 自适应的一个平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # 全连接层
        self.fc = nn.Linear(512,num_classes)



    def _make_layer(self,block,channel,block_num,stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel,out_channels=channel * block.expansion,
                          kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,channel,downsample=downsample,stride=stride))
        self.in_channel = channel * block.expansion

        for i in range(1,block_num):
            layers.append(block(self.in_channel,channel))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        # 扁平化处理，保留第0维，后面维度全部推平
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

def resnet18(num_classes=10):
    return ResNet(block=BasicBlock,blocks_num=[2,2,2,2])


