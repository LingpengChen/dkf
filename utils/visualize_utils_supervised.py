import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_predictions(model, test_loader, num_samples=5, device='cuda', output_path=None):
    """可视化部分预测结果"""
    model = model.to(device)
    model.eval()
    
    plt.figure(figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # 将数据移至设备
            true_pose = batch['true_states'].to(device)
            measurements = batch['tag_anchor_distance_curr'].to(device)
            measurement_to_tag_mapping = batch['measurement_to_tag_mapping'].to(device)
            
            # 前向传播
            _, pose_sample, xt_plus, _ = model(measurements, measurement_to_tag_mapping, true_pose)
            
            # 绘制第一个样本结果
            sample_idx = 0
            
            # 当前轴对象
            ax = plt.subplot(num_samples, 2, 2*i+1)
            
            # 绘制测量值（锚点位置）
            anchor_positions = measurements[sample_idx, :, :2].cpu().numpy()
            for j, pos in enumerate(anchor_positions):
                ax.plot(pos[0], pos[1], 'bo', label='Anchor' if j==0 else "")
            
            # 绘制真实位置
            true_x, true_y = true_pose[sample_idx, 0].item(), true_pose[sample_idx, 1].item()
            ax.plot(true_x, true_y, 'go', markersize=10, label='True Position')
            
            # 绘制估计位置
            est_x, est_y = xt_plus[sample_idx, 0].item(), xt_plus[sample_idx, 1].item()
            ax.plot(est_x, est_y, 'ro', markersize=8, label='Estimated Position')
            
            # 计算误差
            error = np.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)
            
            ax.set_title(f'Sample {i+1} - Position (Error: {error:.3f}m)')
            ax.legend()
            ax.grid(True)
            
            # 绘制角度
            ax = plt.subplot(num_samples, 2, 2*i+2, projection='polar')
            
            true_theta = true_pose[sample_idx, 2].item()
            est_theta = xt_plus[sample_idx, 2].item()
            
            # 绘制真实角度
            ax.plot([true_theta, true_theta], [0, 1], 'g-', linewidth=3, label='True Angle')
            
            # 绘制估计角度
            ax.plot([est_theta, est_theta], [0, 1], 'r-', linewidth=3, label='Estimated Angle')
            
            # 计算角度误差（考虑周期性）
            angle_diff = abs(true_theta - est_theta)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            
            ax.set_title(f'Sample {i+1} - Angle (Error: {np.degrees(angle_diff):.2f}°)')
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    img_dir = os.path.join(output_path, "prediction_visualization.png")
    plt.savefig(img_dir)
    plt.show()

def plot_training_history(history, output_path=None):
    """绘制训练历史图表"""
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    # plt.plot(history['train_recon_loss'], label='train_recon_loss Loss')
    # plt.plot(history['train_kl_loss'], label='train_kl_loss Loss')
    
    plt.plot(history['val_loss'], label='Test Loss')
    # plt.plot(history['val_recon_loss'], label='val_recon_loss Loss')
    # plt.plot(history['val_kl_loss'], label='val_kl_loss Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    # 位姿误差曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['pose_error'], label='Pose Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error (m)')
    plt.title('Average Pose Error')
    plt.legend()
    
    plt.tight_layout()
    img_dir = os.path.join(output_path, "training_history.png")
    plt.savefig(img_dir)
    plt.show()