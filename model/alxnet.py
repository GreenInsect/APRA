import torch
import torch.nn as nn
import torch.nn.functional as F


class Alxnet(nn.Module):
    def __init__(self, **kwargs):
        super(Alxnet, self).__init__(**kwargs)
        # 第一层是 4*4 的卷积, 输入的channels是3, 输出的channels是64,步长 2,没有 padding
        # Conv2d 的第一个参数为输入通道, 第二个参数为输出通道, 第三个参数为卷积核大小
        # ReLU 的参数为inplace, True表示直接对输入进行修改, False表示创建新创建一个对象进行修改
        # 3*32*32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2),  # 64*15*15
            nn.ReLU(True)
        )

        # 第二层为 2*2 的池化, 步长为1, 没有padding
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=1)  # 64*14*14

        # 第三层是3*3的卷积, 输入的channels是64, 输出的channels是256,步长为1, padding为1
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),  # 256*14*14
            nn.ReLU(True)
        )

        # 第四层是 2*2 的池化,  步长是2, 没有padding
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256*7*7

        # 使用三个连续的卷积层和较小的卷积窗口. 
        # 除了最后的卷积层, 输出通道的数量进一步增加. 
        # 在前两个卷积层之后, 汇聚层不用于减少输入的高度和宽度

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 384*7*7
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # 384*7*7
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.ReLU(True)
        )

        # 第八层是 2*2 的池化,  步长是 1, 没有padding
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=1)  # 256*6*6

        # 第九层张量展平
        self.flaten1 = nn.Flatten()

        # 第10层是全连接层, 输入是 9216 , 输出是4608
        self.fc1 = nn.Sequential(
            nn.Linear(9216, 4608),
            nn.ReLU(True),
            nn.Dropout(p=0.5)
        )

        # 第六层是全连接层, 输入是 4608,  输出是4608
        self.fc2 = nn.Sequential(
            nn.Linear(4608, 4608),
            nn.ReLU(True),
            nn.Dropout(p=0.5)
        )

        # 第七层是全连接层, 输入是4608,  输出是 10
        self.linear = nn.Linear(4608, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool3(x)
        x = self.flaten1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.linear(x)
        return x

    # 实例化model


# model = Model()
# # 调用gpu
# model = model.cuda()
# # 优化器选用SGD
# optimize = torch.optim.SGD(model.parameters(), lr=0.05)
# # 损失函数选用交叉熵损失函数
# loss_fn = nn.CrossEntropyLoss()
# # 输出模型
# print(model)