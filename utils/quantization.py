import torch

def get_qconfig_for_bitwidth(bitwidth):
    if bitwidth == 8:
        act_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.HistogramObserver.with_args(
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            )
        )
        weight_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
            )
        )
    elif bitwidth == 6:
        act_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.HistogramObserver.with_args(
                quant_min=0,
                quant_max=63,  # 2**6 - 1
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            )
        )
        weight_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
                quant_min=-32,  # -2**(6-1)
                quant_max=31,   # 2**(6-1)-1
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
            )
        )
    elif bitwidth == 4:
        act_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.HistogramObserver.with_args(
                quant_min=0,
                quant_max=15,  # 2**4 - 1
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            )
        )
        weight_qconfig = torch.ao.quantization.FakeQuantize.with_args(
            observer=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
                quant_min=-8,  # -2**(4-1)
                quant_max=7,   # 2**(4-1)-1
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
            )
        )
    else:
        raise ValueError("Unsupported bitwidth")
    return torch.quantization.QConfig(activation=act_qconfig, weight=weight_qconfig)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 
