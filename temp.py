# DKF和EKF的误差值
dkf_xy = 0.059145379811525345
dkf_theta = 0.8910912275314331
ekf_xy = 0.06981628388166428
ekf_theta = 9.86917781829834

# 计算提升百分比
# (原来的值 - 新值) / 原来的值 * 100% = 提升百分比
xy_improvement = (ekf_xy - dkf_xy) / ekf_xy * 100
theta_improvement = (ekf_theta - dkf_theta) / ekf_theta * 100

print(f"XY坐标估计误差提升：{xy_improvement:.2f}%")
print(f"Theta角度估计误差提升：{theta_improvement:.2f}%")

# 具体数值对比
print("\n详细对比：")
print(f"XY误差 - DKF: {dkf_xy:.4f} vs EKF: {ekf_xy:.4f}")
print(f"Theta误差 - DKF: {dkf_theta:.4f} vs EKF: {ekf_theta:.4f}")