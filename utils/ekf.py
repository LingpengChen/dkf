import numpy as np
from experiment_parameter import *

# ================== EKF实现 ==================
class ExtendedKalmanFilter:
    def __init__(self, x0, P0, Q, R, tags_relative, anchors_global):
        self.x = x0.copy()          # 状态估计 [t_x, t_y, theta]
        self.P = P0.copy()          # 协方差矩阵
        self.Q = Q                  # 过程噪声协方差
        self.R = R                  # 单次观测噪声方差
        self.tags_relative = tags_relative  # Tag相对平台的位置
        self.anchors_global = anchors_global  # Anchor全局坐标

    def predict(self, u_measured):
        # 预测步骤，使用测量的输入
        self.x += u_measured
        self.P += self.Q

    def update(self, z_observed):
        # 构造有效观测的H矩阵和残差向量
        H_list = []
        residuals = []
        R_diag = []

        # 遍历所有可能的Tag-Anchor对
        for tag_idx in range(num_tags):
            tag_rel = self.tags_relative[tag_idx]
            s1, s2 = tag_rel

            # 计算当前状态下的Tag全局位置
            t_x, t_y, theta = self.x
            R_mat = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
            tag_global = R_mat @ tag_rel + np.array([t_x, t_y])

            for anchor_idx in range(num_anchors):
                z = z_observed[tag_idx, anchor_idx]
                if np.isnan(z):  # 跳过无效观测
                    continue

                # 计算预测距离和残差
                a = self.anchors_global[anchor_idx]
                delta = a - tag_global
                d_pred = np.linalg.norm(delta)
                residual = z - d_pred

                # 计算雅可比行列（对平台状态的导数）
                delta_x, delta_y = delta
                d_im0 = d_pred

                # 对平移量的导数
                H_tx = -delta_x / d_im0
                H_ty = -delta_y / d_im0

                # 对旋转角的导数
                H_theta = (delta_x * (s1 * np.sin(theta) + s2 * np.cos(theta)) +
                          delta_y * (-s1 * np.cos(theta) + s2 * np.sin(theta))) / d_im0

                H = np.array([H_tx, H_ty, H_theta])
                H_list.append(H)
                residuals.append(residual)
                R_diag.append(self.R)

        if len(residuals) == 0:  # 无有效观测
            return

        # 构造整体H矩阵和R矩阵
        H = np.stack(H_list, axis=0)  # M×3矩阵
        residuals = np.array(residuals)  # M×1向量
        R_matrix = np.diag(R_diag)       # M×M对角矩阵

        # 卡尔曼增益计算
        S = H @ self.P @ H.T + R_matrix
        K = self.P @ H.T @ np.linalg.inv(S)

        # 状态更新
        self.x += K @ residuals
        self.P = (np.eye(3) - K @ H) @ self.P