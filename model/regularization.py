import torch

def gaussian_kl_regularization(x1_hat, P1_hat, x1_bar, P1_bar):
    """
    Compute KL divergence between two multivariate Gaussian distributions
    KL(N(x1_hat, P1_hat) || N(x1_bar, P1_bar))
    对于两个多维高斯分布 P ~ N(μ₁, Σ₁) 和 Q ~ N(μ₂, Σ₂), 它们之间的KL散度公式为:
    KL(P||Q) = 0.5 * [tr(Σ₂⁻¹Σ₁) + (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁) - k + ln(|Σ₂|/|Σ₁|)]
    
    Args:
        x1_hat: mean from deep KF [batch_size, state_dim]
        P1_hat: covariance from deep KF [batch_size, state_dim, state_dim]
        x1_bar: mean from traditional EKF [batch_size, state_dim]
        P1_bar: covariance from traditional EKF [batch_size, state_dim, state_dim]
    Returns:
        kl_loss: KL divergence loss
    """
    batch_size = x1_hat.size(0)
    state_dim = x1_hat.size(1)
    
    # 为数值稳定性添加小的对角项
    eps = 1e-6
    eye = torch.eye(state_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(x1_hat.device)
    P1_hat = P1_hat + eps * eye
    P1_bar = P1_bar + eps * eye
    
    try:
        # 计算 P1_bar 的逆矩阵
        P1_bar_inv = torch.inverse(P1_bar)
        
        # 计算均值差
        diff = (x1_bar - x1_hat).unsqueeze(-1)  # [batch_size, state_dim, 1]
        
        
        # 计算 trace(P1_bar^(-1) @ P1_hat)
        term1 = torch.bmm(P1_bar_inv, P1_hat)
        term1 = torch.diagonal(term1, dim1=1, dim2=2).sum(dim=1)  # trace
        
        # 计算 (μ1_bar - μ1_hat)^T @ P1_bar^(-1) @ (μ1_bar - μ1_hat)
        term2 = torch.bmm(torch.bmm(diff.transpose(1, 2), P1_bar_inv), diff).squeeze()
        
        # 计算 ln(|P1_bar|/|P1_hat|)
        logdet_P1_bar = torch.logdet(P1_bar)
        logdet_P1_hat = torch.logdet(P1_hat)
        term3 = logdet_P1_bar - logdet_P1_hat
        
        # 组合所有项
        kl_div = 0.5 * (term1 + term2 - state_dim + term3)
        
        # 返回批次平均值
        kl_loss = torch.mean(kl_div)
        
        return kl_loss
        
    except:
        # 如果发生数值问题，使用更稳定的版本
        return gaussian_kl_regularization_stable(x1_hat, P1_hat, x1_bar, P1_bar)

def gaussian_kl_regularization_stable(x1_hat, P1_hat, x1_bar, P1_bar):
    """
    Numerically stable version using Cholesky decomposition
    """
    batch_size = x1_hat.size(0)
    state_dim = x1_hat.size(1)
    
    # 添加抖动
    eps = 1e-6
    eye = torch.eye(state_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(x1_hat.device)
    P1_hat = P1_hat + eps * eye
    P1_bar = P1_bar + eps * eye
    
    # 使用Cholesky分解
    L_hat = torch.linalg.cholesky(P1_hat)
    L_bar = torch.linalg.cholesky(P1_bar)
    
    # 计算均值差
    diff = (x1_bar - x1_hat).unsqueeze(-1)
    
    # 使用Cholesky求解而不是直接求逆
    sol = torch.cholesky_solve(diff, L_bar)
    term2 = torch.bmm(diff.transpose(1, 2), sol).squeeze()
    
    # 计算trace项
    sol2 = torch.cholesky_solve(L_hat, L_bar)
    term1 = torch.diagonal(torch.bmm(L_hat.transpose(1, 2), sol2), dim1=1, dim2=2).sum(dim=1)
    
    # 计算log determinant
    logdet_P1_bar = 2 * torch.sum(torch.log(torch.diagonal(L_bar, dim1=1, dim2=2)), dim=1)
    logdet_P1_hat = 2 * torch.sum(torch.log(torch.diagonal(L_hat, dim1=1, dim2=2)), dim=1)
    term3 = logdet_P1_bar - logdet_P1_hat
    
    # 组合所有项
    kl_div = 0.5 * (term1 + term2 - state_dim + term3)
    
    return torch.mean(kl_div)

# 在训练循环中使用：
def training_step(self, batch):
    # 前向传播得到深度KF的估计
    x1_hat, P1_hat = self.deep_kf(batch)
    
    # 获取传统EKF的估计
    x1_bar, P1_bar = self.traditional_ekf(batch)
    
    # 计算重构损失
    recon_loss = self.reconstruction_loss(batch)
    
    # 计算KL正则化损失
    kl_loss = gaussian_kl_regularization(x1_hat, P1_hat, x1_bar, P1_bar)
    
    # 总损失
    beta = 0.1  # KL损失的权重
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss