import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

P_pred, R = 2.0**2, 1.0**2        # 预测误差的方差、读数噪声的方差
n = 100_000                       # 每种噪声做 10 万次实验
rng = np.random.default_rng()

def unit_noise(kind, size):
    """均值为 0、方差为 1，但形状各不相同的几种噪声"""
    if kind == '高斯':
        return rng.standard_normal(size)
    if kind == '均匀':
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size)
    if kind == '双峰':
        return rng.choice([-0.9, 0.9], size) + np.sqrt(0.19) * rng.standard_normal(size)
    if kind == '重尾':            # 95% 的时候噪声很小，5% 的时候放大 10 倍
        big = rng.random(size) < 0.05
        return rng.standard_normal(size) * np.where(big, 10, 1) / np.sqrt(5.95)

K_grid = np.linspace(0, 1, 21)
K_star = P_pred / (P_pred + R)    # 推导出来的最优增益
plt.plot(K_grid, (1 - K_grid)**2 * P_pred + K_grid**2 * R, 'k', label='理论 P(K)')
for kind in ['高斯', '均匀', '双峰', '重尾']:
    e_pred = np.sqrt(P_pred) * unit_noise(kind, n)   # 预测误差
    v = np.sqrt(R) * unit_noise(kind, n)             # 读数噪声
    var = [np.mean(((1 - K) * e_pred + K * v)**2) for K in K_grid]
    plt.plot(K_grid, var, 'o', ms=4, label=kind)
    best = np.mean(((1 - K_star) * e_pred + K_star * v)**2)
    print(f'{kind}: 在 K* 处实测方差 = {best:.3f}')
print(f'理论：K* = {K_star:.2f}，最小方差 = {(1 - K_star) * P_pred:.3f}')
plt.axvline(K_star, color='gray', ls=':')
plt.xlabel('K'); plt.ylabel('估计误差的方差'); plt.legend(); plt.show()
