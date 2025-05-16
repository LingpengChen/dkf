import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm
from utils.data_loader import create_data_loaders
from utils.visualize_utils import plot_training_history, visualize_predictions
from model.dkf import DKF
from training_manager import TrainingManager

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(model: DKF, train_loader, val_loader, epochs=50, lr=1e-3, 
                device='cuda', training_manager:TrainingManager =None) -> dict: 
    """
    训练模型并在每个epoch后验证
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=1e-3,
    #     weight_decay=0.01
    # )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=10)
    
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        
        'test_loss': [],
        'pose_error': [],
        'learning_rates': []
    }
    
    # 恢复之前的训练状态（如果存在）
    start_epoch = 0
    
    # 如果有训练管理器，尝试恢复之前的训练状态
    if training_manager:
        start_epoch, best_val_loss = training_manager.load_checkpoint(
            model, optimizer, scheduler, load_best=True)
        saved_history = training_manager.load_history()
        if saved_history:
            history = saved_history
    
    for epoch in range(start_epoch, epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        
        beta = 1e-5
        pose_errors = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                # print(batch['tag_anchor_distance_curr'][0][0]) tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.7827])
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
                
                # KL权重预热
                # beta = min(epoch / warmup_epochs, 1.0) * target_beta

                # 梯度裁剪
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.zero_grad()
                loss, recon_loss, kl_loss, _, x1_hat, _, _ = model(x0_bar, P0_bar, x1_bar, P1_bar, u1, Q1, M1, M2,  measurement_to_tag_mapping, true_pose=None, beta=beta)
                # loss, recon_loss, kl_loss, _, x1_hat, _ = model(x0_bar, P0_bar, x1_bar, P1_bar, u1, Q1, M1, M2,  measurement_to_tag_mapping, true_pose=None, beta=beta)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()

                pbar.set_postfix({'loss': loss.item()})
                
                pose_error = torch.sqrt(torch.sum((x1_true[:, :2] - x1_hat[:, :2])**2, dim=1))
                pose_errors.extend(pose_error.detach().cpu().numpy()) 
        
        mean_pose_error = np.mean(pose_errors)
        
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        
        
        with torch.no_grad():
            # 验证集评估
            for batch in val_loader:
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


                loss, recon_loss, kl_loss, pose_sample, x1_hat, P1_hat, _ = model(x0_bar, P0_bar, x1_bar, P1_bar, u1, Q1, M1, M2,  measurement_to_tag_mapping, true_pose=None, beta=beta)
                # loss, recon_loss, kl_loss, pose_sample, x1_hat, P1_hat = model(x0_bar, P0_bar, x1_bar, P1_bar, u1, Q1, M1, M2,  measurement_to_tag_mapping, true_pose=x1_true, beta=beta)

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()    
                
                # pose_error = torch.sqrt(torch.sum((x1_true[:, :2] - x1_hat[:, :2])**2, dim=1))
                # pose_errors.extend(pose_error.cpu().numpy())       
            
        # mean_pose_error = np.mean(pose_errors)
        
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)
                
        # 更新学习率
        scheduler.step(val_loss)
        
        # 更新历史记录
        history['train_loss'].append(float(train_loss))
        history['train_recon_loss'].append(float(train_recon_loss))
        history['train_kl_loss'].append(float(train_kl_loss))
        
        history['val_loss'].append(float(val_loss))
        history['val_recon_loss'].append(float(val_recon_loss))
        history['val_kl_loss'].append(float(val_kl_loss))
        
        history['learning_rates'].append(float(optimizer.param_groups[0]['lr']))
        history['pose_error'].append(float(mean_pose_error))
        

        # 保存检查点和历史记录
        if training_manager:
            # is_best = val_loss < training_manager.best_val_loss
            training_manager.save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss)
            training_manager.save_history(history)
        
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, train_recon_loss: {train_recon_loss:.4f}, train_kl_loss: {beta*train_kl_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, val_recon_loss: {val_recon_loss:.4f}, val_kl_loss: {beta*val_kl_loss:.4f}")
        print(f"Train pose error: {mean_pose_error:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
    return history

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # dataset_dir = os.path.abspath( os.path.join(script_dir, "./uwb_data_high_u_noise") )
    # dataset_dir = os.path.abspath( os.path.join(script_dir, "./uwb_data_med_u_noise") )
    dataset_dir = os.path.abspath( os.path.join(script_dir, "./uwb_data_low_u_noise") )

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(dataset_dir, batch_size=32)

    
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    
    # 初始化模型和训练管理器
    dkf_model = DKF(hidden_dim=64, latent_dim=3)
    training_manager = TrainingManager(
        model_name='DKF_Model',
        save_dir='/home/clp/workspace/dkf/training_runs',
        checkpoint_frequency=5,  # 每5个epoch保存一次
        keep_last_n=5,          # 保留最近3个checkpoint
        save_best=True,          # 保存最佳模型
        # pretrained_model_dir = "DKF_Model_20250516_150219"
    )
    
    # 训练模型
    history = train_model(
        model=dkf_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
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