import torch
import torch.nn as nn
import torch.nn.functional as F
from model.simple import SimpleNet


class Mobilenet(SimpleNet):
    # mobilenet  基本单元
    def conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def __init__(self):
        # 核心算子的定义
        super(Mobilenet, self).__init__()

        self.conv1 = nn.Sequential(  # 序列容器, 用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输入通道为3, 输出通道为64
            nn.BatchNorm2d(32),  # 归一化处理, 输出channel数64
            nn.ReLU()
        )
        self.conv_dw2 = self.conv_dw(32, 32, 1)
        self.conv_dw3 = self.conv_dw(32, 64, 2)  # 进行一次下采样
        self.conv_dw4 = self.conv_dw(64, 64, 1)
        self.conv_dw5 = self.conv_dw(64, 128, 2)
        self.conv_dw6 = self.conv_dw(128, 128, 1)
        self.conv_dw7 = self.conv_dw(128, 256, 2)
        self.conv_dw8 = self.conv_dw(256, 256, 1)
        self.conv_dw9 = self.conv_dw(256, 512, 2)
        # 以上conv_dw(*, *, 2) 4个2, 即进行2^4=16倍的下采样
        self.linear = nn.Linear(512, 100)

    # 对以上算子进行串联, 完成前项运算
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)
        out = F.avg_pool2d(out, 2)  # 上面16倍的下采样, 此处采用2 * 2 的＆进行average pooling
        out = out.view(-1, 512)  # 转为2维图
        out = self.linear(out)  # 对5122维向10维分布转化

        return out

    def first_activations(self, x):
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.conv1(x)
        #x = self.conv_dw2(x)
        out = self.conv1(x)
        return out


class SupConMobilenet(SimpleNet):
    # mobilenet  基本单元
    def conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),

            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def __init__(self):
        # 核心算子的定义
        super(Mobilenet, self).__init__()

        self.conv1 = nn.Sequential(  # 序列容器, 用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输入通道为3, 输出通道为64
            nn.BatchNorm2d(32),  # 归一化处理, 输出channel数64
            nn.ReLU()
        )
        self.conv_dw2 = self.conv_dw(32, 32, 1)
        self.conv_dw3 = self.conv_dw(32, 64, 2)  # 进行一次下采样
        self.conv_dw4 = self.conv_dw(64, 64, 1)
        self.conv_dw5 = self.conv_dw(64, 128, 2)
        self.conv_dw6 = self.conv_dw(128, 128, 1)
        self.conv_dw7 = self.conv_dw(128, 256, 2)
        self.conv_dw8 = self.conv_dw(256, 256, 1)
        self.conv_dw9 = self.conv_dw(256, 512, 2)
        # 以上conv_dw(*, *, 2) 4个2, 即进行2^4=16倍的下采样


    # 对以上算子进行串联, 完成前项运算
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)
        out = F.avg_pool2d(out, 2)  # 上面16倍的下采样, 此处采用2 * 2 的＆进行average pooling
        out = out.view(-1, 512)  # 转为2维图
        return out





