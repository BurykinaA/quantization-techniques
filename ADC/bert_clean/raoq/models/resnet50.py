
import torch
import torch.nn as nn
from torch import Tensor
from torchvision._internally_replaced_utils import load_state_dict_from_url
from train.layer_modules import *
import sys
import pdb


class CustomSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input, adc_bit, loss, **kwargs):
        for module in self:
            if isinstance(module, BottleneckQuant):
                input, loss = module(input, adc_bit, loss, **kwargs)
            elif isinstance(module, QuantConv2d):
                input, loss = module(input, adc_bit, **kwargs)
            else:
                input = module(input, **kwargs)
        return input, loss


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3_quant(in_planes, out_planes, stride = 1, groups = 1, dilation = 1, config=dict(), writer=None):
    return QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=None, dilation=dilation, writer=writer, **config)


def conv1x1_quant(in_planes, out_planes, stride = 1, config=dict(), writer=None):
    return QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=None, writer=writer, **config)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckQuant(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, 
                 name='', writer=None, config_conv=dict(), config_bn=dict()):
        super().__init__()
        self.name = name
        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_quant(inplanes, width, config=config_conv, writer=writer)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3_quant(width, width, stride, groups, dilation, config=config_conv, writer=writer)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1_quant(width, planes * self.expansion, config=config_conv, writer=writer)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, adc_bit, loss):
        identity = x

        out, loss1 = self.conv1(x, adc_bit, self.name+'_conv1')
        out = self.bn1(out)
        out = self.relu(out)

        out, loss2 = self.conv2(out, adc_bit, self.name+'_conv2')
        out = self.bn2(out)
        out = self.relu(out)

        out, loss3 = self.conv3(out, adc_bit, self.name+'_conv3')
        out = self.bn3(out)

        if self.downsample is not None:
            identity, loss_d = self.downsample(x, adc_bit, 0)
        else:
            loss_d = 0

        out += identity
        out = self.relu(out)

        return out, loss1 + loss2 + loss3 + loss_d + loss


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000, zero_init_residual = False,
                 groups = 1, width_per_group = 64, replace_stride_with_dilation = None,
                 norm_layer = None, general_config=dict()):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer)
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetQuant(nn.Module):
    def __init__(self, block, layers, num_classes = 1000, zero_init_residual = False,
                 groups = 1, width_per_group = 64, replace_stride_with_dilation = None,
                 norm_layer = None, general_config=dict(), writer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        sen = general_config.pop('sen')
        self.groups = groups
        self.base_width = width_per_group
        if not sen:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = QuantConv2dSensitive(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, **general_config)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block[1], 64, layers[0],
                                       name='stack1', writer=writer, config_conv=general_config, config_bn=None)
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       name='stack2', writer=writer, config_conv=general_config, config_bn=None)
        self.layer3 = self._make_layer(block[1], 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       name='stack3', writer=writer, config_conv=general_config, config_bn=None)
        self.layer4 = self._make_layer(block[1], 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], 
                                       name='stack4', writer=writer, config_conv=general_config, config_bn=None)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if not sen:
            self.fc = nn.Linear(512 * block[1].expansion, num_classes)
        else:
            self.fc = QuantLinearSensitive(512 * block[1].expansion, num_classes, **general_config)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckQuant) or isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False, name='', writer=None, config_conv=None, config_bn=None):
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = CustomSequential(
                conv1x1_quant(self.inplanes, planes * block.expansion, stride, config=config_conv),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if config_conv:
            layers.append(
                block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, 
                      name+'_block1', writer, config_conv, config_bn)
            )
        else:
            layers.append(
                block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)
            )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if config_conv:
                layers.append(
                    block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, 
                          name=name+'_block'+str(i+1), writer=writer, config_conv=config_conv, config_bn=config_bn)
                )
            else:
                layers.append(
                    block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer) 
                )

        return CustomSequential(*layers)

    def _forward_impl(self, x, adc_bit):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        loss = 0
        x, loss = self.layer1(x, adc_bit, loss)
        x, loss = self.layer2(x, adc_bit, loss)
        x, loss = self.layer3(x, adc_bit, loss)
        x, loss = self.layer4(x, adc_bit, loss)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, loss

    def forward(self, x, adc_bit):
        return self._forward_impl(x, adc_bit)


def _resnet(block, layers, pretrained, arch, model_dir, device, custom, **kwargs):
    loc = 'cuda:{}'.format(device)
    model = arch(block, layers, **kwargs)

    if pretrained:
        if custom:
            state_dict = torch.load(model_dir, map_location=loc)['state_dict']
            new_state_dict = {}
            for name, param in state_dict.items():
                new_state_dict[name[7:]] = param
            model.load_state_dict(new_state_dict, strict=False)
        else:
            state_dict = torch.load(model_dir, map_location=loc)
            model.load_state_dict(state_dict, strict=False)

    return model


def resnet50(**kwargs):
    general_config = kwargs.get('general_config')
    pretrained = kwargs.get('pretrained')
    fp_mode = general_config.pop('fp_mode')
    device = kwargs.get('device')
    model_dir = kwargs.get('model_dir')
    custom = kwargs.get('custom')
    if fp_mode:
        return _resnet(Bottleneck, [3, 4, 6, 3], pretrained, ResNet, general_config=general_config, model_dir=model_dir, device=device, custom=custom)
    else:
        return _resnet([Bottleneck, BottleneckQuant], [3, 4, 6, 3], pretrained, ResNetQuant, general_config=general_config, model_dir=model_dir, device=device, custom=custom, writer=kwargs.get('writer'))

