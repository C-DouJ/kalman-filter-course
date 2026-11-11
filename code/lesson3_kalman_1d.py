import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 和第 1 课同一辆小车
N, dt, u = 200, 1.0, 1.0
Q, R = 0.5**2, 2.0**2
rng = np.random.default_rng()

x_true = np.zeros(N)
for k in range(1, N):
    x_true[k] = x_true[k-1] + u*dt + rng.normal(0, np.sqrt(Q))
z = x_true + rng.normal(0, np.sqrt(R), N)   # z[0] 不用，第一次读数在 k = 1

# 卡尔曼滤波
x_hat = np.zeros(N)
P = np.zeros(N)
K = np.zeros(N)
# 初始化：给定初始估计和初始误差方差
x0_hat, P0 = 3.0, 3.0**2          # 真实起点是 0，我们猜 3 m，并承认可能差 3 m 左右
x_hat[0], P[0] = x0_hat, P0       # 对起点一无所知时，把 P0 设得很大（比如 1e6）
for k in range(1, N):
    # 预测
    x_pred = x_hat[k-1] + u*dt                  # ① 往前推一步
    P_pred = P[k-1] + Q                         # ② 不确定性变大
    # 更新
    K[k] = P_pred / (P_pred + R)                # ③ 算卡尔曼增益
    x_hat[k] = x_pred + K[k] * (z[k] - x_pred)  # ④ 用读数修正
    P[k] = (1 - K[k]) * P_pred                  # ⑤ 不确定性变小

print(f'卡尔曼滤波 RMSE = {np.sqrt(np.mean((x_hat - x_true)**2)):.2f} m')
print(f'传感器读数 RMSE = {np.sqrt(np.mean((z - x_true)**2)):.2f} m')
print(f'最后的增益 K = {K[-1]:.3f}')

t = np.arange(N)
fig, (a1, a2) = plt.subplots(2, 1, sharex=True)
a1.fill_between(t, -2*np.sqrt(P), 2*np.sqrt(P), alpha=0.2, label='±2√P')
a1.plot(t, x_hat - x_true, label='估计误差')
a1.plot(t, z - x_true, '.', label='读数误差')
a1.set_ylabel('误差 (m)'); a1.legend()
a2.plot(t[1:], K[1:])
a2.set_ylabel('K'); a2.set_xlabel('k（步）')
plt.show()
