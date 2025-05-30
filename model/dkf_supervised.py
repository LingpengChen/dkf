import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import UWBPoseEncoder


class DKF_supervised(nn.Module):
    """完整的VAE模型，包括编码器和损失计算"""
    
    def __init__(self, hidden_dim=64, latent_dim=3):
        super(DKF_supervised, self).__init__()
        
        tril_elements = latent_dim * (latent_dim + 1) // 2
        
        self.encoder = UWBPoseEncoder(hidden_dim, latent_dim, tril_elements) # get mean and logL (表示covar mat Cholesky分解的下三角部分)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logL):
        batch_size = mu.size(0)
        
        # 构建下三角矩阵L
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim).to(mu.device)
        tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = logL.exp()
        
        # 采样和重参数化
        eps = torch.randn_like(mu)
        z = mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
        return z
    
    def forward(self, measurements, measurement_to_tag_mapping, true_pose=None):
        """
        前向传播
        
        参数:
            measurements: [batch_size, num_measurements, 5] UWB测量
            measurement_to_tag_mapping: [batch_size, num_measurements] 测量到tag的映射
            true_pose: [batch_size, 3] 真实位姿 (x, y, theta)
            
        返回:
            如果提供true_pose，返回损失和采样位姿；否则只返回采样位姿
        """
        # 编码 logL is log of low trangular matrix
        xt_plus, logL = self.encoder(measurements, measurement_to_tag_mapping)
        
        # 采样
        pose_sample = self.reparameterize(xt_plus, logL)
        
        if true_pose is not None:
            # 计算损失
            recon_loss = F.mse_loss(pose_sample, true_pose)
            
            # KL散度
            # kl_loss = -0.5 * torch.sum(1 + logL - xt_plus.pow(2) - logL.exp())
            
            # 总损失
            # total_loss = recon_loss + kl_loss
            total_loss = recon_loss 
            
            return total_loss, pose_sample, xt_plus, logL
        else:
            return pose_sample

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
        
        dkf_model = DKF_supervised(hidden_dim=64, latent_dim=3)
        
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