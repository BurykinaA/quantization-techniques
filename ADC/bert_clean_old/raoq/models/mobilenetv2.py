from typing import Callable, Any, Optional, List, Dict
import torch
from torch import Tensor
from torch import nn
from torchvision.models._utils import _make_divisible
from torchvision._internally_replaced_utils import load_state_dict_from_url
from train.layer_modules import *
import sys
import pdb


class CustomSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input, adc_bit, loss, **kwargs):
        for module in self:
            if isinstance(module, InvertedResidualQuant) or isinstance(module, Conv2dNormActivationQuant):
                input, loss = module(input, adc_bit, loss, **kwargs)
            elif isinstance(module, QuantConv2d):
                input, loss_temp = module(input, adc_bit, **kwargs)
                loss += loss_temp
            elif isinstance(module, QuantConv2dDW):
                input, loss_temp = module(input, **kwargs)
                loss += loss_temp
            else:
                input = module(input, **kwargs)
        return input, loss


class Conv2dNormActivationQuant(nn.Sequential):
    def __init__(
        self, 
        inp: int, 
        oup: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: Optional[int] = None, 
        groups: int = 1, 
        norm_layer: Optional[Callable[..., torch.nn.Module]] = nn.BatchNorm2d, 
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU, 
        dilation: int = 1, 
        inplace: Optional[bool] = True, 
        bias: Optional[bool] = None,
        name:str = '',
        config:Dict = dict(),
        first:bool=False,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        if first:
            layers = [QuantConv2dSensitive(inp, oup, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, writer=None, x_bias=True, name=name, **config)]
        else:
            layers = [QuantConv2d(inp, oup, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, writer=None, x_bias=True, name=name, **config)]
        if norm_layer:
            layers.append(norm_layer(oup))
        if activation_layer:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        super().__init__(*layers)


    def forward(self, input, adc_bit, loss):
        for module in self:
            if isinstance(module, QuantConv2d):
                input, loss_temp = module(input, adc_bit)
                loss += loss_temp
            else:
                input = module(input)
        return input, loss


class Conv2dNormActivation(nn.Sequential):
    def __init__(
        self, 
        inp: int, 
        oup: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: Optional[int] = None, 
        groups: int = 1, 
        norm_layer: Optional[Callable[..., torch.nn.Module]] = nn.BatchNorm2d, 
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU, 
        dilation: int = 1, 
        inplace: Optional[bool] = True, 
        bias: Optional[bool] = None,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [nn.Conv2d(inp, oup, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)]
        if norm_layer:
            layers.append(norm_layer(oup))
        if activation_layer:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        super().__init__(*layers)


class InvertedResidualQuant(nn.Module):
    def __init__(
        self, 
        inp: int, 
        oup: int, 
        stride: int, 
        expand_ratio: int, 
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        name: str = '',
        config:Dict = dict()
    ):
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        config_dw = config.copy()
        config_dw['bpbs_mode'] = False
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.extend(
                [
                    QuantConv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=None, writer=None, x_bias=True, name=name+'_pw', **config),
                    norm_layer(hidden_dim),
                    nn.ReLU6(inplace=True)
                ]
            )
        layers.extend(
            [
                # dw
                QuantConv2dDW(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, dilation=1, groups=hidden_dim, bias=None, writer=None, x_bias=False, name=name+'_dw', **config),
                norm_layer(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                QuantConv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False, writer=None, x_bias=False, name=name+'_linear', **config),
                norm_layer(oup),
            ]
        )
        self.conv = CustomSequential(*layers)


    def forward(self, x: Tensor, adc_bit: int, loss) -> Tensor:
        y, loss = self.conv(x, adc_bit, loss)
        if self.use_res_connect:
            return x + y, loss
        else:
            return y, loss


class InvertedResidual(nn.Module):
    def __init__(
        self, 
        inp: int, 
        oup: int, 
        stride: int, 
        expand_ratio: int, 
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.extend(
                [
                    nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=None),
                    norm_layer(hidden_dim),
                    nn.ReLU6(inplace=True)
                ]
            )
        layers.extend(
            [
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, dilation=1, groups=hidden_dim, bias=None),
                norm_layer(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Quant(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: Optional[float] = 0.2,
        config:Dict = dict(),
    ):
        super().__init__()

        if block is None:
            block = InvertedResidualQuant

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        sen = config.pop('sen')

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        if not sen:
            features: List[nn.Module] = [
                Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            ]
        else:
            features: List[nn.Module] = [
                Conv2dNormActivationQuant(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6, name='first', config=config, first=True)
            ]
        # inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                name = str(t) + '_' + str(c) + '_' + str(n) + '_' + str(s) +'_' 
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, name=name, config=config))
                input_channel = output_channel

        # put them together
        self.features = CustomSequential(*features)

        # last several layers
        self.conv = Conv2dNormActivationQuant(
                        input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, name=name+'_last', config=config
                    )
        
        # building classifier
        if not sen:
            self.classifier = nn.Linear(self.last_channel, num_classes)
        else:
            self.classifier = QuantLinearSensitive(self.last_channel, num_classes, **config)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def _forward_impl(self, x: Tensor, adc_bit: int, freeze_scale: bool=False):
        loss = 0
        x, loss = self.features(x, adc_bit, loss)
        x, loss = self.conv(x, adc_bit, loss)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, loss


    def forward(self, x: Tensor, adc_bit: int, freeze_scale: bool=False):
        x, loss = self._forward_impl(x, adc_bit, freeze_scale)
        return x, loss


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: Optional[float] = 0.2,
    ):
        super().__init__()

        print('Multiplier:', width_mult)

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        # inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # put them together
        self.features = nn.Sequential(*features)

        # last several layers
        self.conv = Conv2dNormActivation(
                        input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
                    )

        # building classifier
        self.classifier = nn.Linear(self.last_channel, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def _forward_impl(self, x: Tensor):
        x = self.features(x)
        x = self.conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def forward(self, x: Tensor):
        return self._forward_impl(x)


def _mobilenetv2(arch, **kwargs):
    device = kwargs.pop('device')
    model_dir = kwargs.pop('model_dir')
    pretrained = kwargs.pop('pretrained')
    loc = 'cuda:{}'.format(device)
    custom = kwargs.pop('custom')
    model = arch(**kwargs)

    if pretrained:
        if custom:
            state_dict = torch.load(model_dir, map_location=loc)['state_dict']
            new_state_dict = {}
            for name, param in state_dict.items():
                new_state_dict[name[7:]] = param
            model.load_state_dict(new_state_dict)
        else:
            state_dict = torch.load(model_dir, map_location=loc)
            model.load_state_dict(state_dict, strict=False)
    return model


def mobilenet_v2(**kwargs: Any):

    general_config = kwargs.pop('general_config')
    fp_mode = general_config.pop('fp_mode')
    writer = kwargs.pop('writer')

    if fp_mode:
        return _mobilenetv2(MobileNetV2, **kwargs)
    else:
        return _mobilenetv2(MobileNetV2Quant, config=general_config, **kwargs)

    return model



