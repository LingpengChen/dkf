import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import create_data_loaders
from visualize_utils import plot_training_history, visualize_predictions
from dkf import UWBVAEForPose
import json
from datetime import datetime


def evaluate_model(model, test_loader, device='cuda', save_dir='evaluation_results'):
    """
    全面评估模型性能
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 运行设备
        save_dir: 评估结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    model.eval()
    
    # 初始化评估指标
    metrics = {
        'test_loss': 0,
        'position_errors': [],
        'angle_errors': [],
        'trajectory_errors': [],  # 轨迹误差
        'prediction_times': [],   # 预测时间
        'true_positions': [],     # 真实位置
        'predicted_positions': [] # 预测位置
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # 记录预测时间
            start_time = time.time()
            
            # 将数据移至设备
            true_pose = batch['robot_pose'].to(device)
            measurements = batch['measurements'].to(device)
            measurement_to_tag_mapping = batch['measurement_to_tag_mapping'].to(device)
            
            # 前向传播
            loss, pose_sample, xt_plus, logL = model(measurements, measurement_to_tag_mapping, true_pose)
            
            # 记录预测时间
            metrics['prediction_times'].append(time.time() - start_time)
            
            # 计算位置误差(欧几里得距离)
            position_error = torch.sqrt(torch.sum((true_pose[:, :2] - xt_plus[:, :2])**2, dim=1))
            metrics['position_errors'].extend(position_error.cpu().numpy())
            
            # 计算角度误差（处理角度的周期性）
            angle_diff = torch.abs(true_pose[:, 2] - xt_plus[:, 2])
            angle_diff = torch.min(angle_diff, 2*np.pi - angle_diff)
            metrics['angle_errors'].extend(angle_diff.cpu().numpy())
            
            # 记录真实和预测位置用于轨迹可视化
            metrics['true_positions'].extend(true_pose[:, :2].cpu().numpy())
            metrics['predicted_positions'].extend(xt_plus[:, :2].cpu().numpy())
            
            metrics['test_loss'] += loss.item()
    
    # 计算平均指标
    metrics['test_loss'] /= len(test_loader)
    metrics['mean_position_error'] = np.mean(metrics['position_errors'])
    metrics['std_position_error'] = np.std(metrics['position_errors'])
    metrics['mean_angle_error'] = np.mean(metrics['angle_errors'])
    metrics['std_angle_error'] = np.std(metrics['angle_errors'])
    metrics['mean_prediction_time'] = np.mean(metrics['prediction_times']) * 1000  # 转换为毫秒
    
    # 计算误差分位数
    percentiles = [25, 50, 75, 90, 95]
    position_percentiles = np.percentile(metrics['position_errors'], percentiles)
    angle_percentiles = np.percentile(metrics['angle_errors'], percentiles)
    
    # 打印评估结果
    print("\n===== 模型评估结果 =====")
    print(f"测试损失: {metrics['test_loss']:.4f}")
    print("\n位置误差统计 (meters):")
    print(f"平均值: {metrics['mean_position_error']:.4f} ± {metrics['std_position_error']:.4f}")
    for p, v in zip(percentiles, position_percentiles):
        print(f"{p}th percentile: {v:.4f}")
    
    print("\n角度误差统计 (radians):")
    print(f"平均值: {metrics['mean_angle_error']:.4f} ± {metrics['std_angle_error']:.4f}")
    for p, v in zip(percentiles, angle_percentiles):
        print(f"{p}th percentile: {v:.4f} ({np.degrees(v):.2f}°)")
    
    print(f"\n平均预测时间: {metrics['mean_prediction_time']:.2f} ms")
    
    # 创建评估图表
    plt.figure(figsize=(15, 10))
    
    # 1. 位置误差直方图
    plt.subplot(2, 2, 1)
    plt.hist(metrics['position_errors'], bins=50, density=True, alpha=0.7)
    plt.xlabel('Position Error (m)')
    plt.ylabel('Density')
    plt.title('Position Error Distribution')
    
    # 2. 角度误差直方图
    plt.subplot(2, 2, 2)
    plt.hist(np.degrees(metrics['angle_errors']), bins=50, density=True, alpha=0.7)
    plt.xlabel('Angle Error (degrees)')
    plt.ylabel('Density')
    plt.title('Angle Error Distribution')
    
    # 3. 轨迹比较图
    plt.subplot(2, 2, 3)
    true_positions = np.array(metrics['true_positions'])
    predicted_positions = np.array(metrics['predicted_positions'])
    plt.scatter(true_positions[:, 0], true_positions[:, 1], 
               c='blue', s=1, alpha=0.5, label='True')
    plt.scatter(predicted_positions[:, 0], predicted_positions[:, 1], 
               c='red', s=1, alpha=0.5, label='Predicted')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Comparison')
    plt.legend()
    
    # 4. 误差随时间变化图
    plt.subplot(2, 2, 4)
    plt.plot(metrics['position_errors'], alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Position Error (m)')
    plt.title('Position Error over Time')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_results.png'))
    plt.close()
    
    # 保存详细的评估结果
    results_dict = {
        'test_loss': metrics['test_loss'],
        'mean_position_error': metrics['mean_position_error'],
        'std_position_error': metrics['std_position_error'],
        'mean_angle_error': metrics['mean_angle_error'],
        'std_angle_error': metrics['std_angle_error'],
        'mean_prediction_time': metrics['mean_prediction_time'],
        'position_percentiles': {str(p): v for p, v in zip(percentiles, position_percentiles)},
        'angle_percentiles': {str(p): v for p, v in zip(percentiles, angle_percentiles)}
    }
    
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    return metrics

def visualize_trajectory_3d(true_positions, predicted_positions, save_path=None):
    """
    创建3D轨迹可视化
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制真实轨迹
    ax.scatter(true_positions[:, 0], true_positions[:, 1], 
              range(len(true_positions)), c='blue', s=1, label='True')
    
    # 绘制预测轨迹
    ax.scatter(predicted_positions[:, 0], predicted_positions[:, 1], 
              range(len(predicted_positions)), c='red', s=1, label='Predicted')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Time Step')
    ax.set_title('3D Trajectory Visualization')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    data_dir = "/home/clp/catkin_ws/src/dkf/src/uwb_data"
    
    # 创建数据加载器（现在包含验证集）
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, 
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 初始化模型和训练管理器
    dkf_model = UWBVAEForPose(hidden_dim=64, latent_dim=3)
    training_manager = TrainingManager('DKF_Model')
    
    # 训练模型
    history = train_model(
        model=dkf_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=50,
        lr=1e-3,
        device=device,
        training_manager=training_manager
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    evaluate_model(dkf_model, test_loader, device)
    
    # 可视化预测结果
    visualize_predictions(dkf_model, test_loader, num_samples=5, device=device)

if __name__ == "__main__":
    main()