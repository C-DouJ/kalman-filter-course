import numpy as np

rng = np.random.default_rng(606)

# 实验一：同一次更新重复 20 万次。符号约定：误差 = 估计 − 真值。
n = 200_000
e_pred = rng.normal(0, 2, n)        # 预测误差标准差 2 m
v = rng.normal(0, 1, n)             # 量测噪声标准差 1 m
r = v - e_pred

def inner(a, b):
    return np.mean(a * b)

K_sample = -inner(e_pred, r) / inner(r, r)
print(f'样本算出的增益 {K_sample:.4f}；理论值 4/(4+1) = 0.8')
for K in [0, 0.4, K_sample, 1]:
    e = e_pred + K * r
    print(f'K={K:.3f}  误差与新息的内积={inner(e, r):+.3f}  均方误差={inner(e, e):.3f}')
correction = K_sample * r
e = e_pred + correction
print(f'勾股核对：{inner(e_pred, e_pred):.3f} ≈ '
      f'{inner(correction, correction):.3f} + {inner(e, e):.3f}')

# 实验二：小车从已知位置 0 出发。四次滤波使用完全相同的一组量测值。
N = 20_000
x = np.cumsum(1 + rng.normal(0, 0.5, N))  # 每步前进 1 m，真实 Q = 0.25 m²
z = x + rng.normal(0, 2, N)               # 真实 R = 4 m²
cases = [('真实参数', 0.25, 4), ('Q 偏小', 0.0025, 4),
         ('Q 偏大', 25, 4), ('Q、R 同乘 4', 1, 16)]

for name, Q, R in cases:
    xhat, P = 0.0, 0.0
    r, S = np.empty(N), np.empty(N)
    for k, reading in enumerate(z):
        xp, Pp = xhat + 1, P + Q
        r[k] = reading - xp
        S[k] = Pp + R                     # 滤波器预测的新息方差
        K = Pp / S[k]
        xhat, P = xp + K * r[k], (1 - K) * Pp
    s = r[100:]                  # 跳过最初 100 步
    rho1 = np.corrcoef(s[:-1], s[1:])[0, 1]
    ratio = np.var(s) / np.mean(S[100:])  # 实测方差 / 预测方差
    print(f'{name}：相邻新息相关系数={rho1:+.3f}  方差比={ratio:.3f}')
