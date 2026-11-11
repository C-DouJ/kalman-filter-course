import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

g = lambda x, m, v: np.exp(-(x - m)**2 / (2*v)) / np.sqrt(2*np.pi*v)   # 高斯密度
x = np.linspace(-10, 40, 5001)    # 把位置离散成网格，用数值方法直接算贝叶斯公式
dx = x[1] - x[0]
mean = lambda p: np.sum(x * p) * dx
var = lambda p: np.sum((x - mean(p))**2 * p) * dx

# 1. 更新 = 先验 × 似然，再归一化
x_pred, P_pred = 10.0, 2.0**2     # 先验：预测
z, R = 12.0, 1.0**2               # 读数
post = g(x, x_pred, P_pred) * g(z - x, 0, R)
post /= post.sum() * dx
K = P_pred / (P_pred + R)
print(f'更新  数值：均值 {mean(post):.3f}，方差 {var(post):.3f}')
print(f'      公式：均值 {x_pred + K*(z - x_pred):.3f}，方差 {(1 - K)*P_pred:.3f}')

# 2. 预测 = 平移 uΔt，再和过程噪声的分布做卷积
u, Q = 1.0, 0.5**2
kernel = g(np.arange(-3, 3 + dx/2, dx), 0, Q)
pred = np.convolve(post, kernel, mode='same') * dx     # 卷积：被噪声“抹宽”
print(f'预测  数值：均值 {mean(pred) + u:.3f}，方差 {var(pred):.3f}')
print(f'      公式：均值 {mean(post) + u:.3f}，方差 {var(post) + Q:.3f}')

# 3. 读数噪声不是高斯（重尾：95% 的时候很小，5% 的时候放大 10 倍），方差仍然是 R
def heavy(v, R):
    s2 = R / 5.95
    return 0.95 * g(v, 0, s2) + 0.05 * g(v, 0, 100 * s2)
x_pred, P_pred, R = 10.0, 1.0, 1.0
K = P_pred / (P_pred + R)
for z in [11.0, 18.0]:
    post_h = g(x, x_pred, P_pred) * heavy(z - x, R)
    post_h /= post_h.sum() * dx
    print(f'重尾噪声，读数 z = {z:.0f}：真正的后验均值 {mean(post_h):.2f}，卡尔曼滤波给出 {x_pred + K*(z - x_pred):.2f}')

plt.plot(x, g(x, x_pred, P_pred), label='先验（预测）')
plt.plot(x, heavy(z - x, R), label='似然（重尾噪声）')
plt.plot(x, post_h, lw=2, label='真正的后验')
plt.plot(x, g(x, x_pred + K*(z - x_pred), (1 - K)*P_pred), '--', label='卡尔曼滤波给出的高斯')
plt.xlim(5, 23); plt.xlabel('位置 x (m)'); plt.ylabel('概率密度'); plt.legend(); plt.show()
