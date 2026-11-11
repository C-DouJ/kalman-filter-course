import numpy as np

x_pred, P_pred = 10.0, 2.0**2     # 预测
z, R = 12.0, 1.0**2               # 读数

# 1. 加权最小二乘：数值求最小点，和公式对比
x = np.linspace(0, 30, 300001)
J = (x - x_pred)**2 / P_pred + (z - x)**2 / R
K = P_pred / (P_pred + R)
print(f'数值最小点 {x[J.argmin()]:.3f}，公式给出 {x_pred + K * (z - x_pred):.3f}')

# 2. 读数是个离谱的野值：平方损失 vs Huber 损失（只换掉读数那根弹簧）
def huber(r, c=1.345):
    return np.where(np.abs(r) <= c, r**2 / 2, c * np.abs(r) - c**2 / 2)
P_pred2, z_bad = 1.0**2, 18.0      # 这次预测更有把握一些，读数却差了 8 m
J_sq = (x - x_pred)**2 / P_pred2 + (z_bad - x)**2 / R
J_hub = (x - x_pred)**2 / P_pred2 + 2 * huber((z_bad - x) / np.sqrt(R))
print(f'野值 z = {z_bad}：平方损失给出 {x[J_sq.argmin()]:.2f}，Huber 损失给出 {x[J_hub.argmin()]:.2f}')

# 3. 车不动、Q = 0 时：逐步递推的卡尔曼滤波 = 一次性的加权最小二乘
rng = np.random.default_rng()
x_true, x0, P0 = 5.0, 0.0, 10.0**2
zs = x_true + rng.normal(0, np.sqrt(R), 20)
xh, P = x0, P0
for zk in zs:                     # F = 1、Q = 0、u = 0 的卡尔曼滤波
    K = P / (P + R)
    xh += K * (zk - xh)
    P *= 1 - K
batch = (x0 / P0 + zs.sum() / R) / (1 / P0 + len(zs) / R)   # 精度加权平均
print(f'逐步递推 {xh:.4f}，一次性求解 {batch:.4f}；方差 {P:.4f} vs {1 / (1 / P0 + len(zs) / R):.4f}')
