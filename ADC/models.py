import torch
import torch.nn.functional as F
from torch import nn
from ADC.quantized_layers import LinearADC, LinearQuant, LinearADCAshift, Conv2dADC

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x) 


class MLPADC(nn.Module):
    def __init__(self, bx=8, bw=8, ba=8, k=4):
        super(MLPADC, self).__init__()
        self.layers = nn.Sequential(
            LinearADC(784, 256, bx, bw, ba, k),
            nn.ReLU(),
            LinearADC(256, 128, bx, bw, ba, k),
            nn.ReLU(),
            LinearADC(128, 10, bx, bw, ba, k)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.layers(x)
    

class MLPADCAshift(nn.Module):
    def __init__(self, bx=8, bw=8, ba=8, k=4, ashift_enabled=True):
        super(MLPADCAshift, self).__init__()
        self.layers = nn.Sequential(
            LinearADCAshift(784, 256, bx, bw, ba, k, ashift_enabled=ashift_enabled),
            nn.ReLU(),
            LinearADCAshift(256, 128, bx, bw, ba, k, ashift_enabled=ashift_enabled),
            nn.ReLU(),
            LinearADCAshift(128, 10, bx, bw, ba, k, ashift_enabled=ashift_enabled)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.layers(x)


class MLPQuant(nn.Module):
    def __init__(self, bx=8, bw=8):
        super(MLPQuant, self).__init__()
        self.layers = nn.Sequential(
            LinearQuant(784, 256, bx, bw),
            nn.ReLU(),
            LinearQuant(256, 128, bx, bw),
            nn.ReLU(),
            LinearQuant(128, 10, bx, bw)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class BasicBlockADC(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockADC, self).__init__()
        self.conv1 = nn.Conv2dADC(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2dADC(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_channels = 64

        # CIFAR: input 3x32x32 → 64x32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # No maxpool for CIFAR
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 3x32x32 → 64x32x32
        x = self.layer1(x)                      # 64x32x32
        x = self.layer2(x)                      # 128x16x16
        x = self.layer3(x)                      # 256x8x8
        x = self.layer4(x)                      # 512x4x4

        x = self.avgpool(x)                     # 512x1x1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNetCIFAR_ADC(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetCIFAR_ADC, self).__init__()
        self.in_channels = 64

        # CIFAR: input 3x32x32 → 64x32x32
        self.conv1 = nn.Conv2dADC(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # No maxpool for CIFAR
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.LinearADC(512 * block.expansion, num_classes)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2dADC):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2dADC(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 3x32x32 → 64x32x32
        x = self.layer1(x)                      # 64x32x32
        x = self.layer2(x)                      # 128x16x16
        x = self.layer3(x)                      # 256x8x8
        x = self.layer4(x)                      # 512x4x4

        x = self.avgpool(x)                     # 512x1x1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_cifar(num_classes=10):
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet18_cifar_adc(num_classes=10):
    return ResNetCIFAR_ADC(BasicBlockADC, [2, 2, 2, 2], num_classes=num_classes)
