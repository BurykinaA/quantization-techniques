import torch


def compute_kurtosis_loss(weight: torch.Tensor, target_kurtosis: float = 1.8) -> torch.Tensor:
    """Squared error between the weight kurtosis E[((W-mu)/sigma)^4] and a target."""
    mean_w = weight.mean()
    std_w = weight.std().clamp(min=1e-6)
    normalized = (weight - mean_w) / std_w
    kurtosis = (normalized ** 4).mean()
    return (kurtosis - target_kurtosis) ** 2
