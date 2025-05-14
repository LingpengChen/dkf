import torch
import torch.nn as nn
import torch.nn.functional as F

class UWBPoseEncoder(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=3, tril_elements=6):
        """
        UWB测量到机器人位姿的VAE编码器
        
        参数:
            hidden_dim: 隐藏层维度
            latent_dim: 潜变量维度，这里是3 (x, y, theta)
            tril_elements: covar mat Cholesky分解的下三角部分 维度这里是6
        """
        super(UWBPoseEncoder, self).__init__()
        
        # 1. 个体编码网络 (Φ)
        self.measurement_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),  # [x_i, y_i, m_j, n_j, d_ij]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. 自注意力机制组件
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. 注意力网络1 (按tag聚合measurements)
        self.attention_net1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 4. 注意力网络2 (聚合不同tag的特征)
        self.attention_net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 5. VAE映射层
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)  # 均值映射
        self.logL_proj = nn.Linear(hidden_dim, tril_elements)  # 对数方差映射

    def self_attention(self, features):
        """
        对特征应用自注意力机制
        
        参数:
            features: 形状为 [batch_size, num_measurements, hidden_dim] 的特征张量
            
        返回:
            更新后的特征张量
        """
        # 投影查询、键和值
        queries = self.query_proj(features)  # [batch, num_meas, hidden_dim]
        keys = self.key_proj(features)       # [batch, num_meas, hidden_dim]
        values = self.value_proj(features)   # [batch, num_meas, hidden_dim]
        
        # 计算注意力分数 (缩放点积注意力)
        scale = torch.sqrt(torch.tensor(float(queries.size(-1))))
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale  # [batch, num_meas, num_meas]
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权聚合
        attn_output = torch.matmul(attn_weights, values)  # [batch, num_meas, hidden_dim]
        
        # 残差连接
        return features + attn_output
    
    def aggregate_by_tag(self, features, measurement_to_tag_mapping):
        """
        按tag聚合特征
        
        参数:
            features: [batch_size, num_measurements, hidden_dim] 特征
            measurement_to_tag_mapping: [batch_size, num_measurements] 每个测量对应的tag ID
            
        返回:
            tag_features: [batch_size, num_tags, hidden_dim] 每个tag的特征
        """
        batch_size = features.shape[0]
        device = features.device
        
        # 计算每个测量点的注意力权重
        attn_scores = self.attention_net1(features).squeeze(-1)  # [batch, num_meas]
        
        # 获取不同tag的唯一ID
        unique_tags = []
        tag_features = []
        
        for b in range(batch_size):
            # 获取当前批次中的唯一tag ID
            unique_tag_ids = torch.unique(measurement_to_tag_mapping[b])
            unique_tags.append(unique_tag_ids)
            
            batch_tag_features = []
            for tag_id in unique_tag_ids:
                # 找到当前tag的所有测量
                mask = (measurement_to_tag_mapping[b] == tag_id)
                if not mask.any():
                    continue
                    
                # 提取相关特征和分数
                tag_measurements = features[b, mask]
                tag_scores = attn_scores[b, mask]
                
                # Softmax归一化权重
                tag_weights = F.softmax(tag_scores, dim=0)
                
                # 加权平均
                tag_feature = torch.sum(tag_weights.unsqueeze(-1) * tag_measurements, dim=0)
                batch_tag_features.append(tag_feature)
            
            # 将所有tag特征堆叠
            if batch_tag_features:
                batch_tag_features = torch.stack(batch_tag_features)
            else:
                # 如果没有标签，创建一个空张量
                batch_tag_features = torch.zeros((0, features.shape[-1]), device=device)
            
            tag_features.append(batch_tag_features)
            
        return tag_features, unique_tags
    
    def aggregate_tags(self, tag_features):
        """
        聚合所有tag特征为单一的姿态特征
        
        参数:
            tag_features: 列表，每个元素形状为 [num_tags, hidden_dim]
            
        返回:
            pose_features: [batch_size, hidden_dim] 位姿特征
        """
        batch_size = len(tag_features)
        pose_features = []
        
        for b in range(batch_size):
            if tag_features[b].shape[0] == 0:
                # 处理没有tag的情况
                pose_features.append(torch.zeros(self.measurement_encoder[-1].out_features, 
                                              device=tag_features[b].device))
                continue
                
            # 计算每个tag的注意力权重
            tag_scores = self.attention_net2(tag_features[b]).squeeze(-1)  # [num_tags]
            tag_weights = F.softmax(tag_scores, dim=0)
            
            # 加权平均
            pose_feature = torch.sum(tag_weights.unsqueeze(-1) * tag_features[b], dim=0)
            pose_features.append(pose_feature)
        
        return torch.stack(pose_features)
    
    def forward(self, measurements, measurement_to_tag_mapping):
        """
        前向传播
        
        参数:
            measurements: [batch_size, num_measurements, 5] 每个测量包含 [x_i, y_i, m_j, n_j, d_ij]
            measurement_to_tag_mapping: [batch_size, num_measurements] 每个测量对应的tag ID
            
        返回:
            mu: [batch_size, 3] 位姿均值 (x, y, theta)
            logvar: [batch_size, 3] 位姿对数方差
        """
        batch_size, num_measurements = measurements.shape[:2]
        
        # 1. 个体编码
        features = self.measurement_encoder(measurements)  # [batch, num_meas, hidden_dim]
        
        # 2. add info from other measurement处理
        features = self.self_attention(features)
        
        # 3. 按tag聚合
        tag_features, _ = self.aggregate_by_tag(features, measurement_to_tag_mapping)
        
        # 4. 聚合所有tag特征
        pose_feature = self.aggregate_tags(tag_features)  # [batch, hidden_dim]
        
        # 5. VAE映射
        mu = self.mu_proj(pose_feature)       # [batch, 3] (x, y, theta)
        logL = self.logL_proj(pose_feature)  # [batch, 3]
        
        return mu, logL
    
    # def sample(self, mu, logvar):
    #     """
    #     从分布中采样
        
    #     参数:
    #         mu: 均值
    #         logvar: 对数方差
            
    #     返回:
    #         采样的位姿
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

