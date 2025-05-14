import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import UWBPoseEncoder
from .decoder import UWBDecoder
from .regularization import gaussian_kl_regularization

class UWBVAEForPose(nn.Module):
    """完整的VAE模型，包括编码器和损失计算"""
    
    def __init__(self, hidden_dim=64, latent_dim=3):
        super(UWBVAEForPose, self).__init__()
        
        tril_elements = latent_dim * (latent_dim + 1) // 2
        
        self.encoder = UWBPoseEncoder(hidden_dim, latent_dim, tril_elements) # get mean and logL (表示covar mat Cholesky分解的下三角部分)
        self.latent_dim = latent_dim
        self.decoder = UWBDecoder()

    # def reparameterize_logL(self, mu, logL):
    #     """
    #     Implements the reparameterization trick for VAE sampling
    #     Args:
    #         mu: mean vector of the latent distribution
    #         logL: log values of the lower triangular matrix elements
    #     Returns:
    #         z: sampled latent vectors
    #     """
    #     # Create lower triangular matrix for covariance decomposition
    #     L = self.create_lower_triangular_matrix(logL)
        
    #     # Sample from standard normal distribution
    #     eps = torch.randn_like(mu)
        
    #     # Apply reparameterization trick: z = mu + L * eps
    #     # where L is the Cholesky factor of the covariance matrix
    #     z = mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
    #     return z
    
    
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
    
    
    def forward(self, x0_bar, P0_bar, x1_bar, P1_bar, u1, Q1, M1, M2,  measurement_to_tag_mapping, true_pose=None, beta=0.01):
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
        x1_minus = x0_bar + u1
        P1_minus = P0_bar + Q1
        
        x1_plus, logL = self.encoder(M1, measurement_to_tag_mapping)
        R1_dkf = self.build_covariance_matrix(logL)
        
        ## STEP 2: Update step
        # 1 Calculate Kalman gain: K = P-(P- + R)^(-1)
        # K = torch.bmm(P1_minus, torch.inverse(P1_minus + R1_dkf))
        
        # # 添加数值稳定性处理
        # batch_size = P1_minus.size(0)
        # state_dim = P1_minus.size(1)
        
        # # 添加小的对角项来确保正定性
        # jitter = 1e-6
        # eye = torch.eye(state_dim, device=P1_minus.device).unsqueeze(0).repeat(batch_size, 1, 1)
        # P1_minus = P1_minus + jitter * eye
        # R1_dkf = R1_dkf + jitter * eye
        
        # # 使用更稳定的方式计算卡尔曼增益
        # S = P1_minus + R1_dkf  # Innovation covariance
        
        # try:
        #     # 尝试使用Cholesky分解
        #     L = torch.linalg.cholesky(S)
        #     K = torch.cholesky_solve(P1_minus.transpose(-2, -1), L).transpose(-2, -1)
        # except:
        #     # 如果Cholesky分解失败，增加更大的jitter并重试
        #     jitter = 1e-3
        #     S = P1_minus + R1_dkf + jitter * eye
        #     try:
        #         L = torch.linalg.cholesky(S)
        #         K = torch.cholesky_solve(P1_minus.transpose(-2, -1), L).transpose(-2, -1)
        #     except:
        #         # 如果还是失败，使用伪逆
        #         K = torch.bmm(P1_minus, torch.pinverse(S))
        
        # # 添加小的对角项来提高数值稳定性
        # batch_size = P1_minus.size(0)
        # state_dim = P1_minus.size(1)
        # jitter = 1e-6
        # eye = torch.eye(state_dim, device=P1_minus.device).unsqueeze(0).repeat(batch_size, 1, 1)
        # 计算创新协方差并直接使用伪逆
        # S = P1_minus + R1_dkf + jitter * eye
        S = P1_minus + R1_dkf
        K = torch.bmm(P1_minus, torch.pinverse(S))
        
        # 2 Update state estimate: x_hat = x- + K(x+ - x-)
        innovation = x1_plus - x1_minus
        x1_hat = x1_minus + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
        
        # 3 Update covariance: P_hat = (I - K)P-
        batch_size = P1_minus.size(0)
        I = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(P1_minus.device)
        P1_hat = torch.bmm(I - K, P1_minus)
        
        ## STEP 3: Calculate loss
        # 1 Calculate regular
        kl_loss = gaussian_kl_regularization(x1_hat, P1_hat, x1_bar, P1_bar)

        # 2 Decode to get reconstruction loss
        # reparameterize
        pose_sample = self.reparameterize(x1_hat, P1_hat)
        reconstructed_distances = self.decoder.decode(pose_sample, M2)
        recon_loss = self.decoder.compute_reconstruction_loss(reconstructed_distances, M2)
        
        
            # 总损失
        # beta = 0.1  # KL损失的权重
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss, pose_sample, x1_hat, P1_hat
        # if true_pose is not None:
        #     # 计算损失
        #     recon_loss = F.mse_loss(pose_sample, true_pose)
        #     # 重建距离测量
        #     # reconstructed_distances = self.decoder.decode(true_pose, M2)
        #     # KL散度
        #     # kl_loss = -0.5 * torch.sum(1 + logL - xt_plus.pow(2) - logL.exp())
        #     # 总损失
        #     # total_loss = recon_loss + kl_loss
        #     total_loss = recon_loss 
            
        #     return total_loss, pose_sample, x1_hat, P1_hat
        # else:
        #     return pose_sample

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
        
        dkf_model = UWBVAEForPose(hidden_dim=64, latent_dim=3)
        
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