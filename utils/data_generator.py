import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import json

from experiment_parameter import *

from ekf import ExtendedKalmanFilter

# ================== 生成仿真数据 ==================
def generate_data(x0, R, Q):
    '''
        this code is to generate robot trajectory and measurements
        x0 is robot inital pose
        R is uwb measurement noise covariance
        Q is process noise covariance of u_measurement (relative to true u input)
    '''
    true_states = [x0]
    measurements = []  # 结构：每个时间步为[num_tags, num_anchors]的矩阵（无效测量用NaN表示）
    measured_inputs = []  # 存储带噪声的输入测量
    true_inputs = []      # 存储真实输入
    current_x = x0.copy()

    # 新增：存储每个时间步的[tag位置, anchor位置, 距离]矩阵
    tag_anchor_distance_matrices = []
    measurement_to_tag_mapping = np.zeros((N-1, 36), dtype=np.int64)

    for step in range(N-1):
        # 生成随机真实控制输入
        u_true = np.array([
            np.random.uniform(-0.2, 0.2),    # 随机x方向输入
            np.random.uniform(-0.2, 0.2),    # 随机y方向输入
            np.random.uniform(-0.1, 0.1)     # 随机角度输入
        ])
        true_inputs.append(u_true)

        # 添加过程噪声，生成输入的测量值
        w = np.random.multivariate_normal(np.zeros(3), Q)
        u_measured = u_true + w
        measured_inputs.append(u_measured)

        # 使用真实输入更新状态
        next_x = current_x + u_true
        next_x[2] = next_x[2] % (2 * np.pi)
        true_states.append(next_x)

        # 计算所有Tag的全局坐标
        t_x, t_y, theta = next_x
        R_mat = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
        tags_global = (R_mat @ tags_relative.T).T + np.array([t_x, t_y])

        # 生成观测数据（每个Tag到每个Anchor的距离，添加噪声）
        z = np.full((num_tags, num_anchors), np.nan)

        # 创建当前时间步的[tag位置, anchor位置, 距离]矩阵
        tag_anchor_distance = []

        measurement_idx=0
        for tag_idx in range(num_tags):
            tag_pos = tags_global[tag_idx]  # 当前tag的全局坐标

            for anchor_idx in range(num_anchors):
                anchor_pos = anchors_global[anchor_idx]  # 当前anchor的坐标

                # 计算真实距离
                d_true = np.linalg.norm(anchor_pos - tag_pos)

                # 添加噪声
                v = np.random.normal(0, np.sqrt(R))
                d_measured = d_true + v
                z[tag_idx, anchor_idx] = d_measured

                # 存储[tag位置(x,y), anchor位置(x,y), 距离]
                tag_anchor_distance.append(np.concatenate([tags_relative[tag_idx], anchor_pos, [d_measured]]))
                measurement_to_tag_mapping[step, measurement_idx] = tag_idx

                measurement_idx+=1

        # 将当前时间步的所有tag-anchor对存储为一个矩阵
        tag_anchor_distance_matrices.append(np.array(tag_anchor_distance))

        measurements.append(z)
        current_x = next_x

    return np.array(true_states), measurements, np.array(measured_inputs), np.array(true_inputs), tag_anchor_distance_matrices, measurement_to_tag_mapping


def get_experiment_data(seed=None, initial_position=None, process_noise=None, measurement_noise=None, 
               scale_factor=10.0, output_dir="./uwb_data", filename_prefix="trajectory",P0=None):
    """
    进行一组轨迹实验得到的GT,测量 and EKF估计
    生成数据，运行EKF，并保存处理后的数据为JSON格式

    参数:
        seed: 随机种子，用于生成可重复的轨迹
        initial_position: 初始位置 [x, y, theta]，如果为None则使用默认值
        process_noise: 过程噪声水平，如果为None则使用默认值
        measurement_noise: 测量噪声水平，如果为None则使用默认值
        scale_factor: 用于归一化数据的缩放因子
        output_dir: 输出目录
        filename_prefix: 文件名前缀
    """
    # 设置随机种子（如果提供）
    if seed is not None:
        np.random.seed(seed)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # # 设置Q和R（如果提供自定义值）
    # global Q, R, x0
    # original_Q = Q.copy()
    # original_R = R
    # original_x0 = x0.copy()

    if process_noise is not None:
        Q = np.diag([process_noise, process_noise, process_noise])
    if measurement_noise is not None:
        R = measurement_noise 
    if initial_position is not None:
        x0 = np.array(initial_position)
    if P0 is not None: # robot pose covariance for t0
        P0 = P0

    print(f"生成{N}个时间步的数据...")
    true_states, measurements, measured_inputs, true_inputs, tag_anchor_distance_matrices, measurement_to_tag_mapping = generate_data(x0, R, Q)

    # 初始化EKF
    print("运行EKF算法...")
    ekf = ExtendedKalmanFilter(x0, P0, Q, R, tags_relative, anchors_global)
    estimated_states = [x0.copy()]
    covariance_matrices = [P0.copy()]  # 存储每一步的协方差矩阵

    for i in range(N-1):
        # 预测步骤，使用带噪声的输入测量
        ekf.predict(measured_inputs[i])

        # 更新步骤（传入当前时刻的所有观测）
        ekf.update(measurements[i])

        estimated_states.append(ekf.x.copy())
        covariance_matrices.append(ekf.P.copy())  # 存储当前协方差矩阵

    estimated_states = np.array(estimated_states)
    covariance_matrices = np.array(covariance_matrices)

    # 将true_inputs扩展以匹配其他数组的大小（添加一个初始零输入）
    true_inputs_full = np.vstack([np.zeros(3), true_inputs])

    # 将measured_inputs扩展以匹配其他数组的大小
    measured_inputs_full = np.vstack([np.zeros(3), measured_inputs])

    # 保存所有数据到字典中
    all_data = {
        'true_states': true_states,
        'estimated_states': estimated_states,
        'true_inputs': true_inputs_full,
        'measured_inputs': measured_inputs_full,
        'covariance_matrices': covariance_matrices,
        'measurements': measurements,
        'tag_anchor_distance_matrices': tag_anchor_distance_matrices,
        'measurement_to_tag_mapping': measurement_to_tag_mapping,
    }

    ###################################################
    # 处理数据并根据scale_factor归一化
    ###################################################
    print("处理和归一化数据...")

    # 列表数据需要特殊处理
    tag_anchor_curr = []
    tag_anchor_next = []

    # 对列表中的每个数组进行缩放
    for matrix in all_data['tag_anchor_distance_matrices'][0:N-2]:
        tag_anchor_curr.append(matrix / scale_factor)

    for matrix in all_data['tag_anchor_distance_matrices'][1:N-1]:
        tag_anchor_next.append(matrix / scale_factor)

    # 限制保存的数据量，避免JSON文件过大
    save_steps = min(1000, N-2)

    processed_data = {
        'true_states': np.column_stack([
            all_data['true_states'][1:save_steps+1, 0:2] / scale_factor,  # 位置 (x,y) 
            all_data['true_states'][1:save_steps+1, 2:3] / (2 * np.pi)    # 角度归一化到 [0,1]
        ]).tolist(),
        'estimated_states_prev': np.column_stack([
            all_data['estimated_states'][0:save_steps, 0:2] / scale_factor,
            all_data['estimated_states'][0:save_steps, 2:3] / (2 * np.pi)
        ]).tolist(),
        'estimated_states_curr': np.column_stack([
            all_data['estimated_states'][1:save_steps+1, 0:2] / scale_factor,
            all_data['estimated_states'][1:save_steps+1, 2:3] / (2 * np.pi)
        ]).tolist(),
        'true_inputs': np.column_stack([
            all_data['true_inputs'][1:save_steps+1, 0:2] / scale_factor,
            all_data['true_inputs'][1:save_steps+1, 2:3] / (2 * np.pi)
        ]).tolist(),
        'measured_inputs': np.column_stack([
            all_data['measured_inputs'][1:save_steps+1, 0:2] / scale_factor,
            all_data['measured_inputs'][1:save_steps+1, 2:3] / (2 * np.pi)
        ]).tolist(),
        'covariance_prev': (all_data['covariance_matrices'][0:save_steps] / (scale_factor*scale_factor)).tolist(),
        'covariance_curr': (all_data['covariance_matrices'][1:save_steps+1] / (scale_factor*scale_factor)).tolist(),
        # 'process_covariance': (all_data['process_covariance'][1:save_steps+1] / (scale_factor*scale_factor)).tolist(),
        'measurement_to_tag_mapping': all_data['measurement_to_tag_mapping'][0:save_steps].tolist()
    }

    # 将numpy数组转换为列表以便JSON序列化
    processed_data['tag_anchor_distance_curr'] = [matrix.tolist() for matrix in tag_anchor_curr[:save_steps]]
    processed_data['tag_anchor_distance_next'] = [matrix.tolist() for matrix in tag_anchor_next[:save_steps]]

    # 参数信息
    params = {
        'num_tags': int(num_tags),
        'num_anchors': int(num_anchors),
        'tags_relative': tags_relative.tolist(),
        'anchors_global': anchors_global.tolist(),
        'scale_factor': float(scale_factor),
        'initial_position': x0.tolist(),
        'process_noise': Q.tolist(),
        'process_noise_normalized': (Q/(scale_factor*scale_factor)).tolist(),
        'measurement_noise': float(R),
        'measurement_noise_normalized': float(R/(scale_factor*scale_factor))
    }

    # 保存为JSON格式
    json_filename = f"{filename_prefix}.json"
    json_path = os.path.join(output_dir, json_filename)

    # 合并参数和数据
    combined_data = {
        'params': params,
        'data': processed_data
    }

    with open(json_path, 'w') as f:
        json.dump(combined_data, f)

    print(f"JSON数据已保存到: {json_path}")

    # # 恢复全局参数
    # Q = original_Q
    # R = original_R
    # x0 = original_x0

    return processed_data

# 生成20个不同的轨迹
if __name__ == "__main__":
    # 创建输出目录 os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # dataset_dir = os.path.abspath( os.path.join(script_dir, "../uwb_data_low_u_noise") )
    # dataset_dir = os.path.abspath( os.path.join(script_dir, "../uwb_data_med_u_noise") )
    # dataset_dir = os.path.abspath( os.path.join(script_dir, "../uwb_data_high_u_noise") )
    dataset_dir = os.path.abspath( os.path.join(script_dir, "../uwb_data_super_high_u_noise") )
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Start to generate dataset at {dataset_dir}")

    for i in range(20):
        print(f"\n======= 生成轨迹 {i+1}/20 =======")

        # 为每个轨迹生成随机参数
        seed = i * 100  # 不同的随机种子

        # 随机初始位置 - 确保在场景范围内
        init_x = np.random.uniform(2.0, 8.0)
        init_y = np.random.uniform(2.0, 8.0)
        init_theta = np.random.uniform(0, 2*np.pi)
        initial_position = [init_x, init_y, init_theta]

        # 随机噪声参数
        P0 = np.eye(3) * 0                 # 初始协方差
        process_noise = 20*np.random.uniform(0.1, 0.2)
        measurement_noise = np.random.uniform(0.1, 0.2)

        print(f"初始位置: ({init_x:.2f}, {init_y:.2f}, {init_theta:.2f})")
        print(f"过程噪声: {process_noise:.3f}, 测量噪声: {measurement_noise:.3f}")

        # 生成并保存数据
        processed_data = get_experiment_data(
            seed=seed,
            initial_position=initial_position,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            scale_factor=10.0,
            output_dir=dataset_dir,
            filename_prefix=f"trajectory_{i+1}",
            P0=P0
        )

    print("\n所有20个轨迹数据生成完成!")