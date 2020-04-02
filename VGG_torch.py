import torch
import torch.nn as nn
from torch.autograd import Variable

cfg = {'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = self._make_layers(cfg['vgg16'])
        print(self.features)
        self.classifier = nn.Linear(512, 10)  # 主要是实现CIFAR10，不同的任务全连接的结构不同
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:  # 遍历list
            if x == 'M':
                layers += [nn.MaxPool2d(2, 2)]
            else:
                layers += [nn.Conv2d(in_channels, x, 3, 1, 1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]  ##inplace为True，将会改变输入的数据 ，
                # 否则不会改变原输入，只会产生新的输出
                in_channels = x
        # print(layers)
        return nn.Sequential(*layers)

def main():
    vgg16 = VGG16()

if __name__ == '__main__':
    main()


