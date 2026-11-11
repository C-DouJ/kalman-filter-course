% 1. 设定：小车沿直线开，名义速度 u = 1 m/s，每 Δt = 1 s 读一次传感器
N = 200; dt = 1; u = 1;
Q = 0.5^2;        % 运动扰动的方差，σ_w = 0.5 m
R = 2^2;          % 传感器噪声的方差，σ_R = 2 m

% 2. 生成真实位置和传感器读数（真实位置只有我们知道，滤波器看不到）
x_true = zeros(1, N);
for k = 2:N
    x_true(k) = x_true(k-1) + u*dt + sqrt(Q)*randn;
end
z = x_true + sqrt(R)*randn(1, N);

% 3. 两种“偷懒”的办法
x_dr = (0:N-1) * u * dt;          % 只信预测：从起点按名义速度一路推算

% 4. 卡尔曼滤波：每一步按“谁更靠谱”把预测和读数加权（这五行怎么来的，第 2、3 课讲）
x_kf = zeros(1, N); P = 0;        % 起点已知，所以初始误差方差为 0
for k = 2:N
    x_pred = x_kf(k-1) + u*dt;    % 预测位置
    P_pred = P + Q;               % 预测的不确定性
    K = P_pred / (P_pred + R);    % 给读数的权重
    x_kf(k) = x_pred + K * (z(k) - x_pred);   % 用读数修正
    P = (1 - K) * P_pred;         % 修正后的不确定性
end

fprintf('只信预测:   RMSE = %.2f m\n', sqrt(mean((x_dr - x_true).^2)));
fprintf('只信传感器: RMSE = %.2f m\n', sqrt(mean((z - x_true).^2)));
fprintf('卡尔曼滤波: RMSE = %.2f m\n', sqrt(mean((x_kf - x_true).^2)));
fprintf('理论交叉点 k_c = R/Q = %.0f 步\n', R/Q);

% 画图：颜色和网页一致，黄 = 预测，红 = 读数，蓝 = 卡尔曼滤波
cPred = [0.69 0.45 0.03]; cObs = [0.76 0.21 0.42]; cEst = [0.16 0.34 0.81];
k = 0:N-1;
figure('Color', 'w', 'Position', [100 100 780 420]); hold on;
fill([k, fliplr(k)], sqrt(R) * [ones(1, N), -ones(1, N)], cObs, 'FaceAlpha', 0.10, 'EdgeColor', 'none');
plot(k, sqrt(k*Q), '--', 'Color', cPred, 'LineWidth', 1);
plot(k, -sqrt(k*Q), '--', 'Color', cPred, 'LineWidth', 1, 'HandleVisibility', 'off');
plot(k, z - x_true, '.', 'Color', cObs, 'MarkerSize', 12);
plot(k, x_dr - x_true, '-', 'Color', cPred, 'LineWidth', 2);
plot(k, x_kf - x_true, '-', 'Color', cEst, 'LineWidth', 2);
plot([0 N-1], [0 0], '-', 'Color', [.6 .6 .6], 'HandleVisibility', 'off');
hold off; grid on; box off; set(gca, 'FontSize', 12); xlim([0 N-1]);
legend('读数噪声的 ±σ_R 范围', '理论 ±√(kQ)', '只信传感器的误差', '只信预测的误差', '卡尔曼滤波的误差', 'Location', 'northwest', 'Box', 'off');
xlabel('k（步）'); ylabel('误差 (m)'); title('两种偷懒办法和卡尔曼滤波的误差');
