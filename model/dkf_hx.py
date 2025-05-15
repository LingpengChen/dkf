import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import UWBPoseEncoder
from .decoder import UWBDecoder
from .regularization import gaussian_kl_regularization

class DKF(nn.Module):
    """完整的VAE模型，包括编码器和损失计算"""
    
    def __init__(self, hidden_dim=64, latent_dim=3):
        super(DKF, self).__init__()
        
        tril_elements = latent_dim * (latent_dim + 1) // 2
        
        self.encoder = UWBPoseEncoder(hidden_dim, latent_dim, tril_elements) # get mean and logL (表示covar mat Cholesky分解的下三角部分)
        self.latent_dim = latent_dim
        self.decoder = UWBDecoder()

    def reparameterize_logL(self, mu, logL):
        """
        Implements the reparameterization trick for VAE sampling
        Args:
            mu: mean vector of the latent distribution
            logL: log values of the lower triangular matrix elements
        Returns:
            z: sampled latent vectors
        """
        # Create lower triangular matrix for covariance decomposition
        def create_lower_triangular_matrix(logL):
            
            batch_size = logL.size(0)
            L = torch.zeros(batch_size, self.latent_dim, self.latent_dim).to(logL.device)
            tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0)
            
            # Fill lower triangular part with exponentiated values
            # Using exp ensures positive values on diagonal
            L[:, tril_indices[0], tril_indices[1]] = logL.exp()
            
            return L
        L = create_lower_triangular_matrix(logL)
        
        # Sample from standard normal distribution
        eps = torch.randn_like(mu)
        
        # Apply reparameterization trick: z = mu + L * eps
        # where L is the Cholesky factor of the covariance matrix
        z = mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
        return z
    
    
    def reparameterize(self, mu, covariance):
        def covariance_to_L_stable(covariance):
            """
            Numerically stable version of covariance to logL conversion
            """
            batch_size, dim, _ = covariance.size()
            
            # 添加小的对角项确保正定性
            jitter = 1e-6 * torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1).to(covariance.device)
            covariance = covariance + jitter
            
            try:
                # 尝试Cholesky分解
                L = torch.linalg.cholesky(covariance)
            except:
                # 如果失败，使用更多的jitter
                jitter = 1e-4 * torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1).to(covariance.device)
                covariance = covariance + jitter
                L = torch.linalg.cholesky(covariance)
            
            return L
        # Apply reparameterization trick: z = mu + L * eps
        # where L is the Cholesky factor of the covariance matrix
        L = covariance_to_L_stable(covariance)
        eps = torch.randn_like(mu)
        z = mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
        return z

    
    def build_covariance_matrix(self, logL):
        """
        Build covariance matrix using Cholesky decomposition
        Args:
            logL: log values of lower triangular elements
        Returns:
            covariance matrix: Σ = L * L^T
        """
        def create_lower_triangular_matrix(logL):
            
            batch_size = logL.size(0)
            L = torch.zeros(batch_size, self.latent_dim, self.latent_dim).to(logL.device)
            tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0)
            
            # Fill lower triangular part with exponentiated values
            # Using exp ensures positive values on diagonal
            L[:, tril_indices[0], tril_indices[1]] = logL.exp()
            
            return L
        
        # Create lower triangular matrix L
        L = create_lower_triangular_matrix(logL)
        
        # Compute covariance matrix: Σ = L * L^T
        covariance = torch.bmm(L, L.transpose(1, 2))
        return covariance
    
    
    def forward(self,M1, measurement_to_tag_mapping, true_pose=None):
        """
        前向传播
        
        参数:
            measurements: [batch_size, num_measurements, 5] UWB测量
            measurement_to_tag_mapping: [batch_size, num_measurements] 测量到tag的映射
            true_pose: [batch_size, 3] 真实位姿 (x, y, theta)
            LOSS: P1_hat v.s. P1_bar
        返回:
            如果提供true_pose，返回损失和采样位姿；否则只返回采样位姿
        """
        ## STEP 1: Prediction step (use encoder)
        # x1_minus = x0_bar + u1
        # P1_minus = P0_bar + Q1
        
        x1_plus, logL = self.encoder(M1, measurement_to_tag_mapping)
        
        if true_pose is not None:
        # if False:
            # 计算损失
            pose_sample = self.reparameterize_logL(x1_plus, logL)
            recon_loss = F.mse_loss(pose_sample, true_pose)
            
            total_loss = recon_loss 
            
            # return total_loss, pose_sample, x1_plus, logL
            # return total_loss, recon_loss, kl_loss, pose_sample, x1_hat, P1_hat
            return total_loss, recon_loss, None, pose_sample, None, None
            
        
        
def example_usage():
    from ..utils.data_loader import create_data_loaders
    
    data_dir = "/home/clp/catkin_ws/src/dkf/src/uwb_data"

    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(data_dir, batch_size=16)
    
    # 打印训练集和测试集大小
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 示例：从训练集中获取一个批次的数据
    for batch_idx, batch in enumerate(train_loader):
        
        true_pose = batch['robot_pose']
        measurements = batch['measurements']
        measurement_to_tag_mapping = batch['measurement_to_tag_mapping']
        
        num_measurements = measurements.size()[1]
        
        dkf_model = DKF(hidden_dim=64, latent_dim=3)
        
        # 前向传播
        loss, pose_sample, xt_plus, logL = dkf_model(measurements, measurement_to_tag_mapping, true_pose)
        
        print(f"Loss: {loss.item()}")
        
        # 输出第一个批次的结果
        print("\nBatch 0 results:")
        print(f"True pose: [x={true_pose[0,0]:.3f}, y={true_pose[0,1]:.3f}, θ={true_pose[0,2]:.3f}]")
        print(f"Estimated pose (mu): [x={xt_plus[0,0]:.3f}, y={xt_plus[0,1]:.3f}, θ={xt_plus[0,2]:.3f}]")
        print(f"Sampled pose: [x={pose_sample[0,0]:.3f}, y={pose_sample[0,1]:.3f}, θ={pose_sample[0,2]:.3f}]")
        
        # 显示一些测量值示例
        print("\nSample measurements for batch 0:")
        for i in range(min(5, num_measurements)):
            m = measurements[0, i]
            tag_id = measurement_to_tag_mapping[0, i].item()
            print(f"Measurement {i}: Anchor at ({m[0]:.1f}, {m[1]:.1f}) to Tag {tag_id} " + 
                f"at robot-relative ({m[2]:.1f}, {m[3]:.1f}), Distance: {m[4]:.3f}m")

if __name__ == "__main__":
    example_usage()