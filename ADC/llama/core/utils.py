import torch


def compute_kurtosis_loss(weight: torch.Tensor, target_kurtosis: float = 1.8) -> torch.Tensor:
    # Equation 6: κ = E[((W - μ_W) / σ_W)^4]
    mean_w = weight.mean()
    std_w = weight.std()
    std_w = torch.clamp(std_w, min=1e-6)
    
    normalized = (weight - mean_w) / std_w
    kurtosis = (normalized ** 4).mean()
    
    loss = (kurtosis - target_kurtosis) ** 2
    return loss
    