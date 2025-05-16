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
    
    dkf_xy_errors = []
    dkf_theta_errors = []
    ekf_xy_errors = []
    ekf_theta_errors = []
    
    with torch.no_grad():
        # pose_errors = []
        
        for i, batch in enumerate(test_loader):
            # print(batch['tag_anchor_distance_curr'][0][0])
            # break
            
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
            
            loss, recon_loss, kl_loss, pose_sample, x1_hat, P1_hat, x1_plus = model(x0_bar, P0_bar, x1_bar, P1_bar, u1, Q1, M1, M2,  measurement_to_tag_mapping, true_pose=None)

            dkf_xy_error = torch.sqrt(torch.sum((x1_true[:, :2] - x1_hat[:, :2])**2, dim=1))
            dkf_xy_errors.extend(dkf_xy_error.cpu().numpy()) 
            dkf_theta_error = torch.sqrt(torch.sum((x1_true[:, 2] - x1_hat[:, 2])**2))
            dkf_theta_errors.append(dkf_theta_error.cpu().numpy()) 
            
            ekf_xy_error = torch.sqrt(torch.sum((x1_true[:, :2] - x1_bar[:, :2])**2, dim=1))
            ekf_xy_errors.extend(ekf_xy_error.cpu().numpy()) 
            ekf_theta_error = torch.sqrt(torch.sum((x1_true[:, 2] - x1_bar[:, 2])**2))
            ekf_theta_errors.append(ekf_theta_error.cpu().numpy()) 
            
            # 绘制第一个样本结果
            measurements = M1
            true_pose = x1_true
            ekf_pose = x1_bar
            dkf_pose = x1_hat
            sample_idx = 0
            
            if i < num_samples:
                ax = plt.subplot(num_samples, 2, 2*i+1)
                
                # 绘制测量值（锚点位置）
                anchor_positions = measurements[sample_idx, :, 2:4].cpu().numpy()
                for j, pos in enumerate(anchor_positions):
                    ax.plot(pos[0], pos[1], 'bo', label='Anchor' if j==0 else "")
                
                # 绘制真实位置
                true_x, true_y = true_pose[sample_idx, 0].item(), true_pose[sample_idx, 1].item()
                ax.plot(true_x, true_y, 'go', markersize=10, label='True Position')
                
                # 绘制估计位置
                dfk_est_x, dfk_est_y = dkf_pose[sample_idx, 0].item(), dkf_pose[sample_idx, 1].item()
                ax.plot(dfk_est_x, dfk_est_y, 'ro', markersize=8, label='DKF Estimated Position')
                ekf_est_x, ekf_est_y = ekf_pose[sample_idx, 0].item(), ekf_pose[sample_idx, 1].item()
                ax.plot(ekf_est_x, ekf_est_y, 'bo', markersize=8, label='EKF Estimated Position')
                
                # 计算误差
                dkf_error = np.sqrt((true_x - dfk_est_x)**2 + (true_y - dfk_est_y)**2)
                ekf_error = np.sqrt((true_x - ekf_est_x)**2 + (true_y - ekf_est_y)**2)
                
                ax.set_title(f'Sample {i+1} - Position (dkf_error: {dkf_error:.3f}m, ekf_error: {ekf_error:.3f}m)')
                ax.legend()
                ax.grid(True)
                
                # 绘制角度
                ax = plt.subplot(num_samples, 2, 2*i+2, projection='polar')
                
                true_theta = true_pose[sample_idx, 2].item()
                dkf_est_theta = dkf_pose[sample_idx, 2].item()
                ekf_est_theta = ekf_pose[sample_idx, 2].item()
                
                # 绘制真实角度
                ax.plot([true_theta, true_theta], [0, 1], 'g-', linewidth=3, label='True Angle')
                
                # 绘制估计角度
                ax.plot([dkf_est_theta, dkf_est_theta], [0, 1], 'r-', linewidth=3, label='dkf Estimated Angle')
                ax.plot([ekf_est_theta, ekf_est_theta], [0, 1], 'b-', linewidth=3, label='ekf Estimated Angle')
                
                # 计算角度误差（考虑周期性）
                dkf_angle_diff = abs(true_theta - dkf_est_theta)
                dkf_angle_diff = min(dkf_angle_diff, 2*np.pi - dkf_angle_diff)
                ekf_angle_diff = abs(true_theta - ekf_est_theta)
                ekf_angle_diff = min(ekf_angle_diff, 2*np.pi - ekf_angle_diff)
                
                ax.set_title(f'Sample {i+1} - Angle (DKF Error: {np.degrees(dkf_angle_diff):.2f}°  EKF Error: {np.degrees(ekf_angle_diff):.2f}°)')
                ax.legend(loc='upper right')
                
        mean_dkf_xy_errors = np.mean(dkf_xy_errors)
        mean_dkf_theta_errors = np.mean(dkf_theta_errors)
        mean_ekf_xy_errors = np.mean(ekf_xy_errors)
        mean_ekf_theta_errors = np.mean(ekf_theta_errors)
    print(f'mean_dkf_xy_errors: {mean_dkf_xy_errors}, mean_dkf_theta_errors: {mean_dkf_theta_errors}')
    print(f'mean_ekf_xy_errors: {mean_ekf_xy_errors}, mean_ekf_theta_errors: {mean_ekf_theta_errors}')
    plt.tight_layout()
    img_dir = os.path.join(output_path, f"prediction_visualization_{NUM}.png")
    print(f"prediction_visualization saved at: {img_dir}")
    plt.savefig(img_dir)
    plt.show()


if __name__ == "__main__":
    NUM = 0
    torch.manual_seed(NUM)
    np.random.seed(NUM)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # dataset_dir = os.path.abspath( os.path.join(script_dir, "./uwb_data_low_u_noise") )
    # dataset_dir = os.path.abspath( os.path.join(script_dir, "./uwb_data_med_u_noise") )
    # dataset_dir = os.path.abspath( os.path.join(script_dir, "./uwb_data_high_u_noise") )
    dataset_dir = os.path.abspath( os.path.join(script_dir, "./uwb_data_super_high_u_noise") )

    # 创建数据加载器
    train_loader, _, test_loader = create_data_loaders(dataset_dir, batch_size=16)
    
    # print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(train_loader.dataset)}")
    
    # 初始化模型和训练管理器
    dkf_model = DKF(hidden_dim=64, latent_dim=3)
    
    training_manager = TrainingManager(
        model_name='DKF_Model',
        save_dir='/home/clp/workspace/dkf/training_runs',
        checkpoint_frequency=5,  # 每5个epoch保存一次
        keep_last_n=5,          # 保留最近3个checkpoint
        save_best=True,          # 保存最佳模型
        pretrained_model_dir = "DKF_Model_20250516_153657"
    )

    start_epoch, best_val_loss = training_manager.load_checkpoint(dkf_model, load_best=True)
    saved_history = training_manager.load_history()
    if saved_history:
        history = saved_history
    
    # 可视化预测结果
    visualize_predictions(dkf_model, test_loader, num_samples=5, device=device, output_path=training_manager.run_dir)
