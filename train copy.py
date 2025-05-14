import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm
from utils.data_loader import create_data_loaders
from utils.visualize_utils import plot_training_history, visualize_predictions
from model.dkf import UWBVAEForPose
from training_manager import TrainingManager

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0, training_manager=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.training_manager = training_manager
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.training_manager:
                self.training_manager.save_best_model(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            if self.training_manager:
                self.training_manager.save_best_model(model)
            self.counter = 0


def train_model(model, train_loader, val_loader, test_loader, epochs=50, lr=1e-3, 
                device='cuda', training_manager=None):
    """
    训练模型并在每个epoch后验证
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=10, training_manager=training_manager)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'pose_error': [],
        'learning_rates': []
    }
    
    # 恢复之前的训练状态（如果存在）
    start_epoch = 0
    
    # 如果有训练管理器，尝试恢复之前的训练状态
    if training_manager:
        start_epoch, best_val_loss = training_manager.load_checkpoint(
            model, optimizer, scheduler, load_best=False)
        saved_history = training_manager.load_history()
        if saved_history:
            history = saved_history
    
    for epoch in range(start_epoch, epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                x1_true = batch['true_states'].to(device)
                u1_true = batch['true_inputs'].to(device)
                x0_bar = batch['estimated_states_prev'].to(device)
                x1_bar = batch['estimated_states_curr'].to(device)
                P0_bar = batch['covariance_prev'].to(device)
                P1_bar = batch['covariance_curr'].to(device)
                ut = batch['measured_inputs'].to(device)
                Qt = batch['process_noise'].to(device)
                
                M1 = batch['tag_anchor_distance_curr'].to(device)
                M2 = batch['tag_anchor_distance_next'].to(device)
                measurement_to_tag_mapping = batch['measurement_to_tag_mapping'].to(device)
                
                optimizer.zero_grad()
                loss, pose_sample, xt_plus, logL = model(measurements=M1, 
                                                         measurement_to_tag_mapping=measurement_to_tag_mapping, true_pose=x1_true)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        test_loss = 0
        pose_errors = []
        
        with torch.no_grad():
            # 验证集评估
            for batch in val_loader:
                x1_true = batch['true_states'].to(device)
                M1 = batch['tag_anchor_distance_curr'].to(device)
                measurement_to_tag_mapping = batch['measurement_to_tag_mapping'].to(device)
                loss, pose_sample, xt_plus, logL = model(measurements=M1, 
                                                         measurement_to_tag_mapping=measurement_to_tag_mapping, true_pose=x1_true)
                val_loss += loss.item()
                
            
            # 测试集评估
            for batch in test_loader:
                x1_true = batch['true_states'].to(device)
                M1 = batch['tag_anchor_distance_curr'].to(device)
                measurement_to_tag_mapping = batch['measurement_to_tag_mapping'].to(device)
                loss, pose_sample, xt_plus, logL = model(measurements=M1, 
                                                         measurement_to_tag_mapping=measurement_to_tag_mapping, true_pose=x1_true)
                test_loss += loss.item()
            
                pose_error = torch.sqrt(torch.sum((x1_true[:, :2] - xt_plus[:, :2])**2, dim=1))
                pose_errors.extend(pose_error.cpu().numpy())
        
        val_loss /= len(val_loader)
        test_loss /= len(test_loader)
        mean_pose_error = np.mean(pose_errors)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 更新历史记录
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['test_loss'].append(float(test_loss))
        history['pose_error'].append(float(mean_pose_error))
        history['learning_rates'].append(float(optimizer.param_groups[0]['lr']))
        

        # 保存检查点和历史记录
        if training_manager:
            is_best = val_loss < training_manager.best_val_loss
            training_manager.save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_loss, is_best)
            training_manager.save_history(history)
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    return history

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath( os.path.join(script_dir, "./uwb_data") )

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(dataset_dir, batch_size=16)

    
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 初始化模型和训练管理器
    dkf_model = UWBVAEForPose(hidden_dim=64, latent_dim=3)
    training_manager = TrainingManager(
        model_name='DKF_Model',
        save_dir='/home/clp/catkin_ws/src/dkf/training_runs',
        checkpoint_frequency=5,  # 每5个epoch保存一次
        keep_last_n=5,          # 保留最近3个checkpoint
        save_best=True,          # 保存最佳模型
        pretrained_model_dir = "DKF_Model_20250513_180748"
    )
    
    # 训练模型
    history = train_model(
        model=dkf_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=11,
        lr=1e-3,
        device=device,
        training_manager=training_manager
    )
    
    # show general training result

    # visualize train result
    plot_training_history(history, output_path=training_manager.run_dir)
    
    # 可视化预测结果
    # visualize test result
    visualize_predictions(dkf_model, test_loader, num_samples=5, device=device, output_path=training_manager.run_dir)

    # 评估模型
    # evaluate_model(dkf_model, test_loader, device)

if __name__ == "__main__":
    main()