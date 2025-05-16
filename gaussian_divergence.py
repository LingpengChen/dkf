import numpy as np

import numpy as np
from numpy import linalg as LA
import math

# 给定数据  784.6178
x1_hat = np.array([0.6986, 0.8762, 0.4711])
P1_hat = np.array([[8.6475e-05, 6.2332e-05, 7.3319e-05],
        [6.2332e-05, 1.2736e-04, 1.4312e-04],
        [7.3319e-05, 1.4312e-04, 4.3363e-04]])

x1_bar = np.array([0.6323, 0.7848, 0.8421])
P1_bar = np.array([[ 6.7200e-05,  1.5151e-06, -9.3048e-06],
        [ 1.5151e-06,  1.1392e-04, -7.8339e-05],
        [-9.3048e-06, -7.8339e-05,  1.1817e-04]])
# x1_hat = np.array([0.5097, 0.5596, 0.3876])
# P1_hat = np.array([[8.6527e-05, 7.0419e-05, 8.0408e-05],
#                    [7.0419e-05, 1.5627e-04, 1.6713e-04],
#                    [8.0408e-05, 1.6713e-04, 3.7540e-04]])

# x1_bar = np.array([0.4549, 0.5740, 0.4145])
# P1_bar = np.array([[ 9.8075e-05, -1.0899e-05, -2.0702e-05],
#                    [-1.0899e-05,  1.6116e-04,  1.0638e-04],
#                    [-2.0702e-05,  1.0638e-04,  1.8085e-04]])

# 维度
k = len(x1_hat)

# 计算 KL 散度的各部分
# 1. tr(Σ₂⁻¹Σ₁)
P1_bar_inv = LA.inv(P1_bar)
trace_term = np.trace(P1_bar_inv @ P1_hat)

# 2. (μ₂-μ₁)ᵀΣ₂⁻¹(μ₂-μ₁)
mu_diff = x1_bar - x1_hat
quad_term = mu_diff.T @ P1_bar_inv @ mu_diff

# 3. ln(|Σ₂|/|Σ₁|)
log_det_term = np.log(LA.det(P1_bar) / LA.det(P1_hat))

# 综合计算 KL 散度
kl_divergence = 0.5 * (trace_term + quad_term - k + log_det_term)

print(f"KL(N(x1_hat, P1_hat) || N(x1_bar, P1_bar)) = {kl_divergence}")