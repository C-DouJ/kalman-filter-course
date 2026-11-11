P_pred = 2^2; R = 1^2;            % 预测误差的方差、读数噪声的方差
n = 1e5;                          % 每种噪声做 10 万次实验

K_grid = linspace(0, 1, 21);
K_star = P_pred / (P_pred + R);   % 推导出来的最优增益
cols = [0.16 0.34 0.81; 0.69 0.45 0.03; 0.76 0.21 0.42; 0.25 0.55 0.35];   % 四种噪声各用一种颜色
marks = {'o', 's', '^', 'd'};
figure('Color', 'w', 'Position', [100 100 780 440]); hold on;
plot(K_grid, (1 - K_grid).^2 * P_pred + K_grid.^2 * R, 'k-', 'LineWidth', 2);
kinds = {'高斯', '均匀', '双峰', '重尾'};
for i = 1:numel(kinds)
    e_pred = sqrt(P_pred) * unit_noise(kinds{i}, n);   % 预测误差
    v = sqrt(R) * unit_noise(kinds{i}, n);             % 读数噪声
    vars = arrayfun(@(K) mean(((1 - K) * e_pred + K * v).^2), K_grid);
    plot(K_grid, vars, marks{i}, 'Color', cols(i, :), 'MarkerSize', 7, 'LineWidth', 1.2);
    best = mean(((1 - K_star) * e_pred + K_star * v).^2);
    fprintf('%s: 在 K* 处实测方差 = %.3f\n', kinds{i}, best);
end
fprintf('理论：K* = %.2f，最小方差 = %.3f\n', K_star, (1 - K_star) * P_pred);
yl = ylim;
plot([K_star K_star], yl, ':', 'Color', [.5 .5 .5], 'LineWidth', 1.2);
text(K_star, yl(2), sprintf('  K* = %.2f', K_star), 'VerticalAlignment', 'top', 'FontSize', 12);
hold off; grid on; box off; set(gca, 'FontSize', 12);
legend([{'理论 P(K)'}, kinds], 'Location', 'north', 'Box', 'off');
xlabel('K'); ylabel('估计误差的方差 (m^2)'); title('四种噪声，同一条抛物线');

function w = unit_noise(kind, n)
    % 均值为 0、方差为 1，但形状各不相同的几种噪声
    switch kind
        case '高斯'
            w = randn(n, 1);
        case '均匀'
            w = (2 * rand(n, 1) - 1) * sqrt(3);
        case '双峰'
            w = 0.9 * sign(rand(n, 1) - 0.5) + sqrt(0.19) * randn(n, 1);
        case '重尾'               % 95% 的时候噪声很小，5% 的时候放大 10 倍
            big = rand(n, 1) < 0.05;
            w = randn(n, 1) .* (1 + 9 * big) / sqrt(5.95);
    end
end
