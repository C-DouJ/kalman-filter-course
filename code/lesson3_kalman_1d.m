% 和第 1 课同一辆小车
N = 200; dt = 1; u = 1;
Q = 0.5^2; R = 2^2;

x_true = zeros(1, N);
for k = 2:N
    x_true(k) = x_true(k-1) + u*dt + sqrt(Q)*randn;
end
z = x_true + sqrt(R)*randn(1, N);   % z(1) 不用，第一次读数在 k = 1

% 卡尔曼滤波
x_hat = zeros(1, N); P = zeros(1, N); K = zeros(1, N);
% 初始化：给定初始估计和初始误差方差（MATLAB 下标 1 对应 k = 0）
x0_hat = 3; P0 = 3^2;             % 真实起点是 0，我们猜 3 m，并承认可能差 3 m 左右
x_hat(1) = x0_hat; P(1) = P0;     % 对起点一无所知时，把 P0 设得很大（比如 1e6）
for k = 2:N
    % 预测
    x_pred = x_hat(k-1) + u*dt;                 % ① 往前推一步
    P_pred = P(k-1) + Q;                        % ② 不确定性变大
    % 更新
    K(k) = P_pred / (P_pred + R);               % ③ 算卡尔曼增益
    x_hat(k) = x_pred + K(k) * (z(k) - x_pred); % ④ 用读数修正
    P(k) = (1 - K(k)) * P_pred;                 % ⑤ 不确定性变小
end

fprintf('卡尔曼滤波 RMSE = %.2f m\n', sqrt(mean((x_hat - x_true).^2)));
fprintf('传感器读数 RMSE = %.2f m\n', sqrt(mean((z - x_true).^2)));
fprintf('最后的增益 K = %.3f\n', K(end));

% 画图：颜色和网页一致，蓝 = 估计，红 = 读数
cEst = [0.16 0.34 0.81]; cObs = [0.76 0.21 0.42];
t = 0:N-1;
figure('Color', 'w', 'Position', [100 100 780 560]);
subplot(2, 1, 1); hold on;
fill([t, fliplr(t)], [2*sqrt(P), fliplr(-2*sqrt(P))], cEst, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(t(2:end), z(2:end) - x_true(2:end), '.', 'Color', cObs, 'MarkerSize', 10);
plot(t, x_hat - x_true, '-', 'Color', cEst, 'LineWidth', 2);
plot([0 N-1], [0 0], '-', 'Color', [.6 .6 .6], 'HandleVisibility', 'off');
hold off; grid on; box off; set(gca, 'FontSize', 12); xlim([0 N-1]);
legend('±2√P', '读数误差', '估计误差', 'Location', 'northeast', 'Box', 'off');
ylabel('误差 (m)'); title('估计误差与滤波器自己估计的 ±2√P');
subplot(2, 1, 2);
plot(t(2:end), K(2:end), '-', 'Color', [.1 .1 .1], 'LineWidth', 2);
grid on; box off; set(gca, 'FontSize', 12); xlim([0 N-1]); ylim([0 1]);
ylabel('卡尔曼增益 K'); xlabel('k（步）');
