# from utils.visualize_utils import visualize_predictions
from utils.data_loader import create_data_loaders
from training_manager import TrainingManager
import torch
import os
import numpy as np
from model.dkf import DKF
import matplotlib.pyplot as plt


def visualize_predictions(model, test_loader, num_samples=5, device='cuda', output_path=None):
    """可视化部分预测结果"""
    model = model.to(device)
    model.eval()
    
    plt.figure(figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        # pose_errors = []
        
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
            
            x1_true = batch['true_states'].to(device)
            # u1_true = batch['true_inputs'].to(device)
            
            x0_bar = batch['estimated_states_prev'].to(device)
            x1_bar = batch['estimated_states_curr'].to(device)
            P0_bar = batch['covariance_prev'].to(device)
            P1_bar = batch['covariance_curr'].to(device)
            u1 = batch['measured_inputs'].to(device)
            Q1 = batch['process_noise'].to(device)
            
            M1 = batch['tag_anchor_distance_curr'].to(device)
            M2 = batch['tag_anchor_distance_next'].to(device)
            measurement_to_tag_mapping = batch['measurement_to_tag_mapping'].to(device)
            
            loss, recon_loss, kl_loss, pose_sample, x1_hat, P1_hat = model(x0_bar, P0_bar, x1_bar, P1_bar, u1, Q1, M1, M2,  measurement_to_tag_mapping, true_pose=None)
            # test_loss += loss.item()
            # test_recon_loss += recon_loss.item()
            # if kl_loss is not None:
            #     test_kl_loss += kl_loss.item()

            # if True:
            #     pose_error = torch.sqrt(torch.sum((x1_true[:, :2] - pose_sample[:, :2])**2, dim=1))
            #     pose_errors.extend(pose_error.cpu().numpy())
            # else:
            #     pose_error = torch.sqrt(torch.sum((x1_true[:, :2] - x1_hat[:, :2])**2, dim=1))
            #     pose_errors.extend(pose_error.cpu().numpy())
    
            
            # # 将数据移至设备
            # true_pose = batch['true_states'].to(device)
            # measurements = batch['tag_anchor_distance_curr'].to(device)
            # measurement_to_tag_mapping = batch['measurement_to_tag_mapping'].to(device)
            
            # # 前向传播
            # _, pose_sample, xt_plus, _ = model(measurements, measurement_to_tag_mapping, true_pose)
            # history['pose_error'].append(float(mean_pose_error))

            # 绘制第一个样本结果
            measurements = M1
            true_pose = x1_true
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
            est_x, est_y = x1_hat[sample_idx, 0].item(), x1_hat[sample_idx, 1].item()
            ax.plot(est_x, est_y, 'ro', markersize=8, label='Estimated Position')
            
            # 计算误差
            error = np.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)
            
            ax.set_title(f'Sample {i+1} - Position (Error: {error:.3f}m)')
            ax.legend()
            ax.grid(True)
            
            # 绘制角度
            ax = plt.subplot(num_samples, 2, 2*i+2, projection='polar')
            
            true_theta = true_pose[sample_idx, 2].item()
            est_theta = x1_hat[sample_idx, 2].item()
            
            # 绘制真实角度
            ax.plot([true_theta, true_theta], [0, 1], 'g-', linewidth=3, label='True Angle')
            
            # 绘制估计角度
            ax.plot([est_theta, est_theta], [0, 1], 'r-', linewidth=3, label='Estimated Angle')
            
            # 计算角度误差（考虑周期性）
            angle_diff = abs(true_theta - est_theta)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            
            ax.set_title(f'Sample {i+1} - Angle (Error: {np.degrees(angle_diff):.2f}°)')
            ax.legend(loc='upper right')
            
            # break
    
    plt.tight_layout()
    img_dir = os.path.join(output_path, "prediction_visualization.png")
    print(f"prediction_visualization saved at: {img_dir}")
    plt.savefig(img_dir)
    plt.show()


if __name__ == "__main__":
    num = 0
    torch.manual_seed(num)
    np.random.seed(num)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath( os.path.join(script_dir, "./uwb_data") )

    # 创建数据加载器
    _, _, test_loader = create_data_loaders(dataset_dir, batch_size=16)

    
    # print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 初始化模型和训练管理器
    dkf_model = DKF(hidden_dim=64, latent_dim=3)
    
    training_manager = TrainingManager(
        model_name='DKF_Model',
        save_dir='/home/clp/workspace/dkf/training_runs',
        checkpoint_frequency=5,  # 每5个epoch保存一次
        keep_last_n=5,          # 保留最近3个checkpoint
        save_best=True,          # 保存最佳模型
        pretrained_model_dir = "DKF_Model_20250515_235913"
    )

    # 可视化预测结果
    # visualize test result
    visualize_predictions(dkf_model, test_loader, num_samples=5, device=device, output_path=training_manager.run_dir)

    # 评估模型
    # evaluate_model(dkf_model, test_loader, device)