x_pred = 10; P_pred = 2^2;        % 预测
z = 12; R = 1^2;                  % 读数

% 1. 加权最小二乘：数值求最小点，和公式对比
x = linspace(0, 30, 300001);
J = (x - x_pred).^2 / P_pred + (z - x).^2 / R;
[~, i] = min(J);
K = P_pred / (P_pred + R);
fprintf('数值最小点 %.3f，公式给出 %.3f\n', x(i), x_pred + K * (z - x_pred));

% 2. 读数是个离谱的野值：平方损失 vs Huber 损失（只换掉读数那根弹簧）
huber = @(r, c) (abs(r) <= c) .* r.^2 / 2 + (abs(r) > c) .* (c * abs(r) - c^2 / 2);
P_pred2 = 1^2; z_bad = 18;        % 这次预测更有把握一些，读数却差了 8 m
J_sq = (x - x_pred).^2 / P_pred2 + (z_bad - x).^2 / R;
J_hub = (x - x_pred).^2 / P_pred2 + 2 * huber((z_bad - x) / sqrt(R), 1.345);
[~, i1] = min(J_sq); [~, i2] = min(J_hub);
fprintf('野值 z = %g：平方损失给出 %.2f，Huber 损失给出 %.2f\n', z_bad, x(i1), x(i2));

% 3. 车不动、Q = 0 时：逐步递推的卡尔曼滤波 = 一次性的加权最小二乘
x_true = 5; x0 = 0; P0 = 10^2;
zs = x_true + sqrt(R) * randn(20, 1);
xh = x0; P = P0;
for k = 1:numel(zs)               % F = 1、Q = 0、u = 0 的卡尔曼滤波
    K = P / (P + R);
    xh = xh + K * (zs(k) - xh);
    P = (1 - K) * P;
end
batch = (x0 / P0 + sum(zs) / R) / (1 / P0 + numel(zs) / R);   % 精度加权平均
fprintf('逐步递推 %.4f，一次性求解 %.4f\n', xh, batch);
