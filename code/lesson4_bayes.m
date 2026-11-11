g = @(x, m, v) exp(-(x - m).^2 / (2*v)) / sqrt(2*pi*v);   % 高斯密度
x = linspace(-10, 40, 5001);      % 把位置离散成网格，用数值方法直接算贝叶斯公式
dx = x(2) - x(1);
mean_of = @(p) sum(x .* p) * dx;
var_of = @(p) sum((x - sum(x .* p) * dx).^2 .* p) * dx;

% 1. 更新 = 先验 × 似然，再归一化
x_pred = 10; P_pred = 2^2;        % 先验：预测
z = 12; R = 1^2;                  % 读数
post = g(x, x_pred, P_pred) .* g(z - x, 0, R);
post = post / (sum(post) * dx);
K = P_pred / (P_pred + R);
fprintf('更新  数值：均值 %.3f，方差 %.3f\n', mean_of(post), var_of(post));
fprintf('      公式：均值 %.3f，方差 %.3f\n', x_pred + K*(z - x_pred), (1 - K)*P_pred);

% 2. 预测 = 平移 uΔt，再和过程噪声的分布做卷积
u = 1; Q = 0.5^2;
kernel = g(-3:dx:3, 0, Q);
pred = conv(post, kernel, 'same') * dx;                 % 卷积：被噪声“抹宽”
fprintf('预测  数值：均值 %.3f，方差 %.3f\n', mean_of(pred) + u, var_of(pred));
fprintf('      公式：均值 %.3f，方差 %.3f\n', mean_of(post) + u, var_of(post) + Q);

% 3. 读数噪声不是高斯（重尾：95% 的时候很小，5% 的时候放大 10 倍），方差仍然是 R
heavy = @(v, R) 0.95 * g(v, 0, R / 5.95) + 0.05 * g(v, 0, 100 * R / 5.95);
x_pred = 10; P_pred = 1; R = 1;
K = P_pred / (P_pred + R);
for z = [11, 18]
    post_h = g(x, x_pred, P_pred) .* heavy(z - x, R);
    post_h = post_h / (sum(post_h) * dx);
    fprintf('重尾噪声，读数 z = %g：真正的后验均值 %.2f，卡尔曼滤波给出 %.2f\n', z, mean_of(post_h), x_pred + K*(z - x_pred));
end

% 画图：黄 = 先验，红 = 似然，蓝 = 真正的后验，黑色虚线 = 卡尔曼滤波
cPred = [0.69 0.45 0.03]; cObs = [0.76 0.21 0.42]; cEst = [0.16 0.34 0.81];
figure('Color', 'w', 'Position', [100 100 780 420]); hold on;
plot(x, g(x, x_pred, P_pred), '-', 'Color', cPred, 'LineWidth', 1.5);
plot(x, heavy(z - x, R), '-', 'Color', cObs, 'LineWidth', 1.5);
area(x, post_h, 'FaceColor', cEst, 'FaceAlpha', 0.20, 'EdgeColor', cEst, 'LineWidth', 2.5);
plot(x, g(x, x_pred + K*(z - x_pred), (1 - K)*P_pred), 'k--', 'LineWidth', 1.5);
hold off; grid on; box off; set(gca, 'FontSize', 12); xlim([5 23]);
legend('先验（预测）', '似然（重尾噪声）', '真正的后验', '卡尔曼滤波给出的高斯', 'Location', 'northeast', 'Box', 'off');
xlabel('位置 x (m)'); ylabel('概率密度'); title('重尾噪声、读数离谱时：真正的后验与卡尔曼滤波');
