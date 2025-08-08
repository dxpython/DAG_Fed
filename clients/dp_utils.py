import torch
import numpy as np
from scipy import stats


def clip_gradients(gradient, clip_bound):
    """梯度剪切"""
    grad_norm = torch.norm(gradient)
    if grad_norm > clip_bound:
        return gradient * (clip_bound / grad_norm)
    return gradient


def add_noise(gradient, sigma):
    """添加高斯噪声"""
    if sigma <= 0:
        return gradient

    noise = torch.normal(0, sigma, size=gradient.shape)
    return gradient + noise


def compute_gradient_norm(gradients):
    """计算梯度总范数"""
    total_norm = 0
    for grad in gradients.values():
        total_norm += torch.norm(grad) ** 2
    return torch.sqrt(total_norm)


def adaptive_clipping_bound(grad_norms, method='quantile', q=0.8):
    """自适应剪切阈值计算"""
    if method == 'quantile':
        return np.quantile(grad_norms, q)
    elif method == 'mean_std':
        return np.mean(grad_norms) + np.std(grad_norms)
    elif method == 'median':
        return np.median(grad_norms)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_noise_multiplier(epsilon, delta, sensitivity, steps):
    """计算噪声乘子"""
    if epsilon <= 0:
        return float('inf')

    # 基于RDP的噪声计算
    c = np.sqrt(2 * np.log(1.25 / delta))
    sigma = c * sensitivity / epsilon
    return sigma / sensitivity


def layer_importance_score(gradients, method='norm'):
    """计算层重要性分数"""
    scores = {}

    if method == 'norm':
        total_norm = sum(torch.norm(grad) for grad in gradients.values())
        for name, grad in gradients.items():
            scores[name] = torch.norm(grad) / total_norm

    elif method == 'variance':
        for name, grad in gradients.items():
            scores[name] = torch.var(grad)

    elif method == 'magnitude':
        for name, grad in gradients.items():
            scores[name] = torch.mean(torch.abs(grad))

    return scores


def privacy_budget_allocation(layer_scores, total_budget, allocation_method='proportional'):
    """隐私预算分配"""
    budget_allocation = {}

    if allocation_method == 'proportional':
        for name, score in layer_scores.items():
            budget_allocation[name] = total_budget * score

    elif allocation_method == 'inverse':
        # 重要层分配更少预算
        inv_scores = {name: 1.0 / (score + 1e-8) for name, score in layer_scores.items()}
        total_inv = sum(inv_scores.values())
        for name, inv_score in inv_scores.items():
            budget_allocation[name] = total_budget * inv_score / total_inv

    elif allocation_method == 'uniform':
        per_layer_budget = total_budget / len(layer_scores)
        for name in layer_scores.keys():
            budget_allocation[name] = per_layer_budget

    return budget_allocation


def rdp_accountant(sigma, steps, alpha=None):
    """RDP隐私损失计算"""
    if alpha is None:
        alpha = np.arange(2, 100)

    rdp = np.zeros_like(alpha, dtype=float)
    for i, a in enumerate(alpha):
        if sigma == 0:
            rdp[i] = float('inf')
        else:
            rdp[i] = a / (2 * sigma ** 2)

    return rdp * steps


def rdp_to_dp(rdp, alpha, delta):
    """RDP转换为(ε,δ)-DP"""
    eps = rdp + np.log(delta) / (alpha - 1)
    return np.min(eps[eps > 0])


def sensitivity_analysis(gradients, data_size):
    """敏感度分析"""
    sensitivity = {}

    for name, grad in gradients.items():
        # L2敏感度
        sensitivity[name] = torch.norm(grad) / data_size

    return sensitivity


def noise_calibration(epsilon, delta, sensitivity, composition_steps=1):
    """噪声校准"""
    if epsilon <= 0:
        return float('inf')

    beta = epsilon / (2 * np.sqrt(2 * composition_steps * np.log(1 / delta)))
    sigma = sensitivity / beta

    return sigma