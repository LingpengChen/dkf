import torch
import os
import json
from datetime import datetime

class TrainingManager:
    """增强版训练管理器"""
    def __init__(self, model_name, save_dir='training_runs', checkpoint_frequency=5, 
                 keep_last_n=3, save_best=True, pretrained_model_dir=None):
        self.model_name = model_name
        self.save_dir = save_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        

        if pretrained_model_dir is None:
            self.run_dir = self._create_run_dir()
            print(f"Generate new model path: {self.run_dir}")
        else:
            pretrained_model_dir = os.path.join(self.save_dir, pretrained_model_dir)
            if os.path.exists(pretrained_model_dir):
                # self.pretrained_model_dir = pretrained_model_dir
                self.run_dir = pretrained_model_dir
                print(f"Found pretrained model at: {pretrained_model_dir}")
            else:
                print(f"Pretrained model path does not exist: {pretrained_model_dir}")
                self.run_dir = self._create_run_dir()
                print(f"Generate new model path: {self.run_dir}")
        
            
        # 创建运行目录
        self.checkpoints_dir = os.path.join(self.run_dir, 'checkpoints')
        # if not os.path.exists(pretrained_model_dir):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # 文件路径
        self.best_model_path = os.path.join(self.run_dir, 'best_model.pt')
        self.history_path = os.path.join(self.run_dir, 'history.json')
        self.metadata_path = os.path.join(self.run_dir, 'metadata.json')
        
        # 初始化状态
        self.best_val_loss = float('inf')
        self.checkpoint_files = []
        self.current_epoch = 0
        
        # 加载或初始化元数据
        self._load_or_init_metadata()

    # def save_best_model(self, model):
    #     """保存最佳模型"""
    #     torch.save(model.state_dict(), self.best_model_path)

    def _create_run_dir(self):
        """创建带时间戳的运行目录"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(self.save_dir, f'{self.model_name}_{timestamp}')
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def _load_or_init_metadata(self):
        """加载或初始化元数据"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.best_val_loss = metadata.get('best_val_loss', float('inf'))
                self.current_epoch = metadata.get('current_epoch', 0)
                self.checkpoint_files = metadata.get('checkpoint_files', [])
        else:
            self._save_metadata()
    
    def _save_metadata(self):
        """保存元数据"""
        metadata = {
            'best_val_loss': self.best_val_loss,
            'current_epoch': self.current_epoch,
            'checkpoint_files': self.checkpoint_files,
            'model_name': self.model_name,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_loss, is_best=False):
        """
        保存检查点
        
        参数:
            model: PyTorch模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            val_loss: 验证损失
            is_best: 是否是最佳模型
        """
        self.current_epoch = epoch
        
        # 创建检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 定期保存checkpoint
        if epoch % self.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(
                self.checkpoints_dir, 
                f'checkpoint_epoch_{epoch}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            self.checkpoint_files.append(checkpoint_path)
            
            # 只保留最近的N个checkpoint
            if len(self.checkpoint_files) > self.keep_last_n:
                old_checkpoint = self.checkpoint_files.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
        
        # 保存最佳模型
        if self.save_best and (val_loss < self.best_val_loss or is_best):
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.best_model_path)
        
        # 更新元数据
        self._save_metadata()
    
    
    def save_history(self, history):
        """保存训练历史"""
        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=4)
    
    def load_history(self):
        """加载训练历史"""
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                return json.load(f)
        return None
    
    def load_checkpoint(self, model, optimizer=None, scheduler=None, load_best=False):
        """
        加载检查点
        
        参数:
            model: PyTorch模型
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
            load_best: 是否加载最佳模型
            
        返回:
            epoch: 加载的epoch
            val_loss: 验证损失
        """
        # 确定要加载的检查点文件
        if load_best and os.path.exists(self.best_model_path):
            checkpoint_path = self.best_model_path
        elif self.checkpoint_files:
            checkpoint_path = self.checkpoint_files[-1]  # 加载最新的检查点
        else:
            return 0, float('inf')
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['val_loss']
    