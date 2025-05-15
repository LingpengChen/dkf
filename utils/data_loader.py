import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class UWBDataset(Dataset):
    """自定义UWB定位数据集类，适配新的JSON文件格式"""

    def __init__(self, data_dir, dataset_type='train', train_ratio=0.7, val_ratio=0.15):
        """
        初始化UWB数据集

        参数:
            data_dir (str): 保存数据集文件的目录路径
            dataset_type (str): 数据集类型 ('train', 'val', 或 'test')
            train_ratio (float): 训练集占总数据的比例
            val_ratio (float): 验证集占总数据的比例
        """
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # 获取目录中的所有数据文件
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.json')]

        # 确保文件列表有序
        self.file_list.sort()

        # 计算数据集分割点
        total_files = len(self.file_list)
        train_split = int(total_files * train_ratio)
        val_split = int(total_files * (train_ratio + val_ratio))

        # 根据数据集类型选择相应的文件
        if dataset_type == 'train':
            self.file_list = self.file_list[:train_split]
        elif dataset_type == 'val':
            self.file_list = self.file_list[train_split:val_split]
        else:  # test
            self.file_list = self.file_list[val_split:]

        # 预先加载所有数据
        self.data = self._load_all_data()

    def _load_all_data(self):
        """加载所有批次文件的数据，适配新的JSON文件结构"""
        all_data = []

        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            # 提取参数和数据
            params = json_data['params']
            data = json_data['data']

            # 提取时间步数
            num_steps = len(data['true_states'])

            # [修改] 读取所有数据字段
            for step in range(num_steps):
                sample = {
                    # 当前真实状态 (位置和方向) x_true
                    'true_states': torch.tensor(data['true_states'][step], dtype=torch.float32),

                    # 前一时刻估计状态  xbar_t-1
                    'estimated_states_prev': torch.tensor(data['estimated_states_prev'][step], dtype=torch.float32),

                    # 当前时刻估计状态  xbar_t
                    'estimated_states_curr': torch.tensor(data['estimated_states_curr'][step], dtype=torch.float32),

                    # 真实控制输入  utrue_t
                    'true_inputs': torch.tensor(data['true_inputs'][step], dtype=torch.float32),

                    # 测量的控制输入  u_t
                    'measured_inputs': torch.tensor(data['measured_inputs'][step], dtype=torch.float32),

                    # 前一时刻协方差矩阵  Pbar_t-1
                    'covariance_prev': torch.tensor(data['covariance_prev'][step], dtype=torch.float32),

                    # 当前时刻协方差矩阵  Pbar_t
                    'covariance_curr': torch.tensor(data['covariance_curr'][step], dtype=torch.float32),

                    # 当前时刻tag-anchor距离测量  M_t
                    'tag_anchor_distance_curr': torch.tensor(data['tag_anchor_distance_curr'][step], dtype=torch.float32),

                    # 下一时刻tag-anchor距离测量  M_t+1
                    'tag_anchor_distance_next': torch.tensor(data['tag_anchor_distance_next'][step], dtype=torch.float32),

                    # 测量到Tag的映射 
                    'measurement_to_tag_mapping': torch.tensor(
                        data['measurement_to_tag_mapping'][step], dtype=torch.long
                    ),
                    # Q_t
                    # 'process_noise': torch.tensor(params['process_noise']),
                    'process_noise': torch.tensor(params['process_noise_normalized']),

                    # 额外元数据
                    'file_name': file_name,
                    'step': step,

                }

                # # 保存数据集参数
                # sample['params'] = {
                #     # 'num_tags': params['num_tags'],
                #     # 'num_anchors': params['num_anchors'],
                #     # 'tags_relative': params['tags_relative'],
                #     # 'anchors_global': params['anchors_global'],
                #     # 'scale_factor': params['scale_factor'],
                #     # 'initial_position': params['initial_position'],
                #     # 'measurement_noise': params['measurement_noise'],
                #     # 'measurement_noise_normalized': params['measurement_noise_normalized']
                #     'process_noise': torch.tensor(params['process_noise']),
                #     'process_noise_normalized': torch.tensor(params['process_noise_normalized']),

                # }

                all_data.append(sample)

        return all_data

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        return self.data[idx]

def create_data_loaders(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, num_workers=4):
    """
    创建训练集、验证集和测试集的数据加载器

    参数:
        data_dir (str): 数据文件目录
        batch_size (int): 批次大小
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        num_workers (int): 数据加载的工作线程数

    返回:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器
    """
    # 创建训练集、验证集和测试集
    train_dataset = UWBDataset(data_dir, dataset_type='train', train_ratio=train_ratio, val_ratio=val_ratio)
    val_dataset = UWBDataset(data_dir, dataset_type='val', train_ratio=train_ratio, val_ratio=val_ratio)
    test_dataset = UWBDataset(data_dir, dataset_type='test', train_ratio=train_ratio, val_ratio=val_ratio)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 设置数据目录
    # data_dir = "./uwb_data"  # 根据实际路径修改
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath( os.path.join(script_dir, "../uwb_data") )

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(dataset_dir, batch_size=16)

    # 打印数据集大小
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")

    # 示例：从训练集中获取一个批次的数据并打印所有数据类型
    for batch_idx, batch in enumerate(train_loader):
        print(f"\n批次 {batch_idx}:")
        
        # print(batch.keys())
        # 打印所有数据字段的形状
        print("\n数据字段形状:")
        
        print(f"from which file: {batch['file_name']}")
        print(f"Step: {batch['step']}")
        for key, value in batch.items():
            if key != 'file_name' and key != 'step':
            # if key != 'file_name' and key != 'step':
                print(f"{key}: {value.shape}")
        # batch['params']
        # for key, value in batch['params'].items():
        #     print(f"{key}: {value.shape}")
        
        # 打印样例数据
        print("\n样例数据:")
        print(f"真实状态 (第一个样本): {batch['true_states'][0]}")
        print(f"估计状态 (第一个样本): {batch['estimated_states_curr'][0]}")
        print(f"测量数据 (第一个样本第一行): {batch['tag_anchor_distance_curr'][0]}")
        print(f"测量到Tag的映射: {batch['measurement_to_tag_mapping'][0]}")

        # 打印参数信息
        # print("\n参数信息:")
        # params = batch['params']
        # print(f"params: {params}")
        # print(f"Tag数量: {params['num_tags'][0]}")
        # print(f"Anchor数量: {params['num_anchors'][0]}")
        # print(f"缩放因子: {params['scale_factor'][0]}")

        break  # 只打印第一个批次