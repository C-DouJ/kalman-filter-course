rng(606, 'twister')

% 实验一：同一次更新重复 20 万次。符号约定：误差 = 估计 − 真值。
n = 200000;
e_pred = 2 * randn(n, 1);     % 预测误差标准差 2 m
v = randn(n, 1);              % 量测噪声标准差 1 m
r = v - e_pred;
inner = @(a, b) mean(a .* b);
K_sample = -inner(e_pred, r) / inner(r, r);
fprintf('样本算出的增益 %.4f；理论值 4/(4+1) = 0.8\n', K_sample)
for K = [0, 0.4, K_sample, 1]
    e = e_pred + K * r;
    fprintf('K=%.3f  误差与新息的内积=%+.3f  均方误差=%.3f\n', K, inner(e, r), inner(e, e))
end
correction = K_sample * r;
e = e_pred + correction;
fprintf('勾股核对：%.3f ≈ %.3f + %.3f\n', inner(e_pred, e_pred), inner(correction, correction), inner(e, e))

% 实验二：小车从已知位置 0 出发。四次滤波使用完全相同的一组量测值。
N = 20000;
x = cumsum(1 + 0.5 * randn(N, 1));   % 每步前进 1 m，真实 Q = 0.25 m²
z = x + 2 * randn(N, 1);            % 真实 R = 4 m²
cases = {'真实参数', 0.25, 4; 'Q 偏小', 0.0025, 4; ...
         'Q 偏大', 25, 4; 'Q、R 同乘 4', 1, 16};
for row = 1:size(cases, 1)
    name = cases{row, 1}; Q = cases{row, 2}; R = cases{row, 3};
    xhat = 0; P = 0;
    r = zeros(N, 1); S = zeros(N, 1);
    for k = 1:N
        xp = xhat + 1; Pp = P + Q;
        r(k) = z(k) - xp;
        S(k) = Pp + R;                % 滤波器预测的新息方差
        K = Pp / S(k);
        xhat = xp + K * r(k); P = (1 - K) * Pp;
    end
    s = r(101:end);          % 跳过最初 100 步
    corr_matrix = corrcoef(s(1:end-1), s(2:end));
    ratio = var(s, 1) / mean(S(101:end));
    fprintf('%s：相邻新息相关系数=%+.3f  方差比=%.3f\n', name, corr_matrix(1, 2), ratio)
end
