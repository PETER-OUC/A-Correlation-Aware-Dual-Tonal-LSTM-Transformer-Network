clc; clear; close all;

% ==================== 全局超参数 ====================
DPI = 600;                  % 统一图像输出分辨率（可在此修改）

% ==================== 图形全局设置 ====================
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultTextFontSize', 11);

% ==================== 参数配置 ====================
env_number = 55;            % 声速剖面环境编号

Frequency_1 = 380;
Frequency_2 = 360;

% 真实运动参数（真值，用于图中标注）
CPA_range_real = 900;       % m
velocity_real  = 2.5;       % m/s
CPA_time_real = 2000;       % s

% 时间序列
Time_len = 800;
sampling_rate_1 = 2;
time_list = (1:Time_len*sampling_rate_1) / sampling_rate_1;
time_relative_to_CPA = time_list - CPA_time_real;
range_list_real = sqrt(CPA_range_real^2 + (velocity_real * time_relative_to_CPA).^2);

% ==================== 加载本征数据 ====================
environment_name = ['txt_', num2str(env_number)];
base_path = 'G:\Files\Moving\数据制作\仿真数据制作\';

eigenvalue_file    = [base_path, environment_name, '\', 'k', num2str(Frequency_1), '.txt'];
eigenfunction_file = [base_path, environment_name, '\', 'phi', num2str(Frequency_1), '.txt'];
eigenvalue_file_2    = [base_path, environment_name, '\', 'k', num2str(Frequency_2), '.txt'];
eigenfunction_file_2 = [base_path, environment_name, '\', 'phi', num2str(Frequency_2), '.txt'];

eigenvals_1 = load(eigenvalue_file);
eigenfuncs_1 = load(eigenfunction_file);
eigenvals_2 = load(eigenvalue_file_2);
eigenfuncs_2 = load(eigenfunction_file_2);

num_receivers = 41;
receiver_depth = 10;
source_depth = 20;

% 本征值/函数
k_1 = eigenvals_1(:, 2) + 1i * eigenvals_1(:, 3);
phi_1 = reshape(eigenfuncs_1(:, 1) + 1i * eigenfuncs_1(:, 2), num_receivers, []);
num_modes_1 = length(k_1);

k_2 = eigenvals_2(:, 2) + 1i * eigenvals_2(:, 3);
phi_2 = reshape(eigenfuncs_2(:, 1) + 1i * eigenfuncs_2(:, 2), num_receivers, []);
num_modes_2 = length(k_2);

% ==================== 合成声场并加噪 ====================
pressure_field_F1_sim = 1i * exp(-1i * pi / 4) / sqrt(8 * pi) * ...
    (phi_1(receiver_depth, 1:num_modes_1) .* phi_1(source_depth, 1:num_modes_1)) * ...
    (exp(-1i * k_1(1:num_modes_1) .* range_list_real) ./ ...
     sqrt(k_1(1:num_modes_1) .* range_list_real));

pressure_field_F2_sim = 1i * exp(-1i * pi / 4) / sqrt(8 * pi) * ...
    (phi_2(receiver_depth, 1:num_modes_2) .* phi_2(source_depth, 1:num_modes_2)) * ...
    (exp(-1i * k_2(1:num_modes_2) .* range_list_real) ./ ...
     sqrt(k_2(1:num_modes_2) .* range_list_real));

SNR = 10;
pressure_field_F1_sim = awgn(pressure_field_F1_sim, SNR, "measured");
pressure_field_F2_sim = awgn(pressure_field_F2_sim, SNR, "measured");

fprintf('仿真声场计算完成\n');

% ==================== 功率谱（去均值） ====================
data_F1_T_sim = abs(pressure_field_F1_sim .* conj(pressure_field_F1_sim));
data_F2_T_sim = abs(pressure_field_F2_sim .* conj(pressure_field_F2_sim));
data_F1_T_long_1 = data_F1_T_sim - mean(data_F1_T_sim);
data_F2_T_long_1 = data_F2_T_sim - mean(data_F2_T_sim);

% ==================== 估计真实 beta（单参数参考） ====================
beta_search_list = 0.8:0.002:1.3;
loss_beta = zeros(1, length(beta_search_list));
for i = 1:length(beta_search_list)
    scale = (Frequency_1 / Frequency_2)^(1 / beta_search_list(i));
    data_fit = interp1(range_list_real, range_list_real .* data_F1_T_long_1, range_list_real * scale);
    data_fit(isnan(data_fit)) = 0;
    temp = data_fit ./ (range_list_real .* scale);
    X = temp - mean(temp);
    loss_beta(i) = corr(data_F2_T_long_1(:), X(:));
end
[~, idx_beta] = max(loss_beta);
beta_real = beta_search_list(idx_beta);
fprintf('真值参考：beta=%.3f, v=%.2f m/s, r0=%.0f m, t0=%.0f s\n', ...
    beta_real, velocity_real, CPA_range_real, CPA_time_real);

% ==================== 定义搜索网格 ====================
CPA_time_search = 1500:10:2500;      % t0 搜索范围 / s
beta_search     = 0.8:0.005:1.3;    % beta 搜索范围
Rcpa_search     = 800:10:1200;      % r0 搜索范围 / m

% ==================== (a) 真实几何：已知 r0，搜索 (t0, beta) ====================
loss_a = zeros(length(CPA_time_search), length(beta_search));
for ct = 1:length(CPA_time_search)
    t_rel = time_list - CPA_time_search(ct);
    r_search = sqrt(CPA_range_real^2 + (velocity_real * t_rel).^2);
    for bt = 1:length(beta_search)
        scale = (Frequency_1 / Frequency_2)^(1 / beta_search(bt));
        data_fit = interp1(r_search, r_search .* data_F1_T_long_1, r_search * scale);
        data_fit(isnan(data_fit)) = 0;
        temp = data_fit ./ (r_search .* scale);
        X = temp - mean(temp);
        loss_a(ct, bt) = corr(data_F2_T_long_1(:), X(:));
    end
end

% ==================== (b) 径向近似：r0=0，搜索 (t0, beta) ====================
loss_b = zeros(length(CPA_time_search), length(beta_search));
for ct = 1:length(CPA_time_search)
    t_rel = time_list - CPA_time_search(ct);
    r_search = abs(velocity_real * t_rel);   % 径向近似，忽略 r0
    for bt = 1:length(beta_search)
        scale = (Frequency_1 / Frequency_2)^(1 / beta_search(bt));
        data_fit = interp1(r_search, r_search .* data_F1_T_long_1, r_search * scale);
        data_fit(isnan(data_fit)) = 0;
        temp = data_fit ./ (r_search .* scale);
        X = temp - mean(temp);
        loss_b(ct, bt) = corr(data_F2_T_long_1(:), X(:));
    end
end
% loss_a = (loss_a - min(loss_a) )./(max(loss_a)-min(loss_a));
% loss_b = (loss_b - min(loss_b) )./(max(loss_b)-min(loss_b));
% 
% loss_a = log(abs(loss_a));
% loss_b = log(abs(loss_b));

%==================== 图1：真实几何 vs 径向近似 ====================
fig_ab = figure('Name', 'Fig_ab', ...
    'Position', [150, 150, 1050, 480], 'Color', 'w', 'Renderer', 'painters');

% Muted academic palette
MUTED_RED    = [0.55 0.25 0.30];
MUTED_GOLD   = [0.75 0.65 0.35];
MUTED_BLUE   = [0.45 0.55 0.70];
MUTED_GRAY   = [0.35 0.35 0.35];

cmin = min([min(loss_a(:)),min(loss_b(:))]);
cmax =  max([max(loss_a(:)),max(loss_b(:))]);

% --- (a) 真实CPA几何 ---
ax1 = subplot(1, 2, 1);
pcolor(CPA_time_search, beta_search, loss_a');
shading interp; axis tight; box on; grid on;
hold on;

% 真值标记（红色五角星），捕获句柄用于 legend
h_true_a = plot(CPA_time_real, beta_real, 'p', 'MarkerSize', 14, ...
    'MarkerFaceColor', MUTED_RED, 'MarkerEdgeColor', 'w', 'LineWidth', 1.2);

% 查找全局最大值并标记（黄色方块），捕获句柄
[max_a, idx_max_a] = max(loss_a(:));
[idx_t_a, idx_b_a] = ind2sub(size(loss_a), idx_max_a);
t0_est_a = CPA_time_search(idx_t_a);
beta_est_a = beta_search(idx_b_a);
err_t0_a = t0_est_a - CPA_time_real;
err_beta_a = beta_est_a - beta_real;

h_est_a = plot(t0_est_a, beta_est_a, 's', 'MarkerSize', 12, ...
    'MarkerFaceColor', MUTED_GOLD, 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);

hold off;
clim([cmin, cmax]);
colormap(ax1, parula);
hcb1 = colorbar('Location', 'eastoutside');
hcb1.Label.String = 'Pearson correlation coefficient';
hcb1.Label.FontSize = 10;
xlabel(['CPA time t_{cpa} (s)', newline, '(a)']);
ylabel('Waveguide invariant \beta');
% title removed for LaTeX caption

% Legend 标注真值与估计值（含误差），白底黑字
legend([h_true_a, h_est_a], {
    sprintf('True value (%d s, %.3f)', CPA_time_real, beta_real),
    sprintf('Estimated (%d s, %.3f)', ...
        t0_est_a, beta_est_a)
}, 'Location', 'southeast', 'FontSize', 9, 'Color', 'w');

% --- (b) 径向近似 r0=0 ---
ax2 = subplot(1, 2, 2);
pcolor(CPA_time_search, beta_search, loss_b');
shading interp; axis tight; box on; grid on;
hold on;

% 真值标记（红色五角星），捕获句柄
h_true_b = plot(CPA_time_real, beta_real, 'p', 'MarkerSize', 14, ...
    'MarkerFaceColor', MUTED_RED, 'MarkerEdgeColor', 'w', 'LineWidth', 1.2);

% 查找全局最大值并标记（黄色方块），捕获句柄
[max_b, idx_max_b] = max(loss_b(:));
[idx_t_b, idx_b_b] = ind2sub(size(loss_b), idx_max_b);
t0_est_b = CPA_time_search(idx_t_b);
beta_est_b = beta_search(idx_b_b);
err_t0_b = t0_est_b - CPA_time_real;
err_beta_b = beta_est_b - beta_real;

h_est_b = plot(t0_est_b, beta_est_b, 's', 'MarkerSize', 12, ...
    'MarkerFaceColor', MUTED_GOLD, 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);

hold off;
clim([cmin, cmax]);
colormap(ax2, parula);
hcb2 = colorbar('Location', 'eastoutside');
hcb2.Label.String = 'Pearson correlation coefficient';
hcb2.Label.FontSize = 10;
xlabel(['CPA time t_{cpa} (s)', newline, '(b)']);
ylabel('Waveguide invariant \beta');
% title removed for LaTeX caption

% Legend 标注真值与估计值（含误差）
legend([h_true_b, h_est_b], {
    sprintf('True value (%d s, %.3f)', CPA_time_real, beta_real),
    sprintf('Estimated (%d s, %.3f)', ...
        t0_est_b, beta_est_b)
}, 'Location', 'southeast', 'FontSize', 9, 'Color', 'w');

% sgtitle removed for LaTeX caption

% 命令行输出误差结果（便于直接复制到论文）
fprintf('=== 图1 参数估计误差 ===\n');
fprintf('(a) True geometry :  t0_est=%d s (err=%+d s), beta_est=%.4f (err=%+.4f), rho_max=%.4f\n', ...
    t0_est_a, err_t0_a, beta_est_a, err_beta_a, max_a);
fprintf('(b) Radial approx :  t0_est=%d s (err=%+d s), beta_est=%.4f (err=%+.4f), rho_max=%.4f\n', ...
    t0_est_b, err_t0_b, beta_est_b, err_beta_b, max_b);

% 保存图1
set(fig_ab, 'PaperUnits', 'inches');
set(fig_ab, 'PaperPosition', [0 0 10.5 4.8]);
set(fig_ab, 'PaperPositionMode', 'manual');
print(fig_ab, 'fig_ab.png', '-dpng', sprintf('-r%d', DPI));
fprintf('图1 (a)(b) 已保存\n');

%% ==================== 三参数搜索：计算三维矩阵 ====================
fprintf('开始三参数搜索计算，网格规模 %d x %d x %d，请稍候...\n', ...
    length(CPA_time_search), length(Rcpa_search), length(beta_search));

loss_3d = zeros(length(CPA_time_search), length(Rcpa_search), length(beta_search));
N_total = length(CPA_time_search) * length(Rcpa_search);
count = 0;

for ct = 1:length(CPA_time_search)
    t_rel = time_list - CPA_time_search(ct);
    for rt = 1:length(Rcpa_search)
        r0 = Rcpa_search(rt);
        r_search = sqrt(r0^2 + (velocity_real * t_rel).^2);
        for bt = 1:length(beta_search)
            scale = (Frequency_1 / Frequency_2)^(1 / beta_search(bt));
            data_fit = interp1(r_search, r_search .* data_F1_T_long_1, r_search * scale);
            data_fit(isnan(data_fit)) = 0;
            temp = data_fit ./ (r_search .* scale);
            X = temp - mean(temp);
            loss_3d(ct, rt, bt) = corr(data_F2_T_long_1(:), X(:));
        end
        count = count + 1;
        if mod(count, 5000) == 0
            fprintf('  进度: %d / %d (%.1f%%)\n', count, N_total, count/N_total*100);
        end
    end
end
fprintf('三参数搜索计算完成\n');

%% ==================== 图2：三视图紧凑布局（右侧色标） ====================

% ---------- 公共数据准备 ----------
loss_3d_1 = (loss_3d -min(loss_3d(:)))/(max(loss_3d(:))- min(loss_3d(:)));
loss_3d_perm = (permute(loss_3d_1, [2, 1, 3]));
loss_3d_smooth = smooth3(loss_3d_perm, 'gaussian', 5, 1.5);
[T0_grid_mg, R0_grid_mg, Beta_grid_mg] = meshgrid(CPA_time_search, Rcpa_search, beta_search);

[~, idx_t0]   = min(abs(CPA_time_search - CPA_time_real));
[~, idx_r0]   = min(abs(Rcpa_search - CPA_range_real));
[~, idx_beta] = min(abs(beta_search - beta_real));

slice_r0_beta = squeeze(loss_3d_perm(:, idx_t0, :));
slice_t0_beta = squeeze(loss_3d_perm(idx_r0, :, :));
slice_t0_r0   = squeeze(loss_3d_perm(:, :, idx_beta));

cmin = min(loss_3d_perm(:));
cmax =  max(loss_3d_perm(:));

% ---------- 创建图窗（高度压缩） ----------
fig_all = figure('Name', 'Fig_3View_Final', ...
    'Position', [80, 80, 1300, 750], 'Color', 'w', 'Renderer', 'opengl');

% ---------- (a) 顶部：固定 β，t0-r0 平面 ----------
ax_top = axes('Parent', fig_all, 'Position', [0.28, 0.74, 0.34, 0.18]);
pcolor(ax_top, CPA_time_search, Rcpa_search, slice_t0_r0);
shading interp; axis tight; box on; grid on;
hold(ax_top, 'on');
plot(ax_top, CPA_time_real, CPA_range_real, 'p', 'MarkerSize', 11, ...
    'MarkerFaceColor', MUTED_RED, 'MarkerEdgeColor', 'w', 'LineWidth', 1);
hold(ax_top, 'off');
clim(ax_top, [cmin, cmax]);
colormap(ax_top, parula);
xlabel(ax_top, 't_{cpa} (s)', 'FontSize', 9);
ylabel(ax_top, 'r_{cpa} (m)', 'FontSize', 9);
% title removed for LaTeX caption
text(ax_top, CPA_time_real+40, CPA_range_real+10, ...
    sprintf('(%d, %d)', CPA_time_real, CPA_range_real), 'Color', 'w', 'FontSize', 8);

% ---------- (b) 左侧：固定 t0，r0-β 平面 ----------
ax_left = axes('Parent', fig_all, 'Position', [0.06, 0.14, 0.20, 0.50]);
pcolor(ax_left, beta_search, Rcpa_search, slice_r0_beta);
shading interp; axis tight; box on; grid on;
hold(ax_left, 'on');
plot(ax_left, beta_real, CPA_range_real, 'p', 'MarkerSize', 11, ...
    'MarkerFaceColor', MUTED_RED, 'MarkerEdgeColor', 'w', 'LineWidth', 1);
hold(ax_left, 'off');
clim(ax_left, [cmin, cmax]);
colormap(ax_left, parula);
xlabel(ax_left, '\beta', 'FontSize', 9);
ylabel(ax_left, 'r_{cpa} (m)', 'FontSize', 9);
% title removed for LaTeX caption
text(ax_left, beta_real+0.015, CPA_range_real+10, ...
    sprintf('(%.3f, %d)', beta_real, CPA_range_real), 'Color', 'w', 'FontSize', 8);

% ---------- (c) 中央：3D 搜索空间 ----------
ax_center = axes('Parent', fig_all, 'Position', [0.28, 0.14, 0.34, 0.50]);
hold(ax_center, 'on'); grid(ax_center, 'on'); box(ax_center, 'on');

h_slice = slice(T0_grid_mg, R0_grid_mg, Beta_grid_mg, loss_3d_smooth, ...
    CPA_time_real, CPA_range_real, beta_real);
set(h_slice, 'EdgeColor', 'none', 'FaceAlpha', 0.25);
shading interp;

threshold = 0.38;
mask_high = loss_3d_1 > threshold;
[idx_t, idx_r, idx_b] = ind2sub(size(loss_3d_1), find(mask_high));
t_high = CPA_time_search(idx_t);
r_high = Rcpa_search(idx_r);
b_high = beta_search(idx_b);
rho_high = loss_3d_1(mask_high);
if ~isempty(rho_high)
    scatter3(t_high, r_high, b_high, ...
        20 + 60*(rho_high - threshold)/(max(rho_high)-threshold), ...
        rho_high, 'filled', 'MarkerFaceAlpha', 0.85);
end
colormap(ax_center, parula); clim(ax_center, [cmin, cmax]);

plot3(CPA_time_real, CPA_range_real, beta_real, 'p', 'MarkerSize', 20, ...
    'MarkerFaceColor', MUTED_RED, 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
plot3([CPA_time_real CPA_time_real], [CPA_range_real CPA_range_real], ...
    [min(beta_search) beta_real], '--', 'Color', MUTED_RED, 'LineWidth', 1);
plot3([CPA_time_real CPA_time_real], [min(Rcpa_search) CPA_range_real], ...
    [beta_real beta_real], '--', 'Color', MUTED_RED, 'LineWidth', 1);
plot3([min(CPA_time_search) CPA_time_real], [CPA_range_real CPA_range_real], ...
    [beta_real beta_real], '--', 'Color', MUTED_RED, 'LineWidth', 1);

xlabel(ax_center, 't_{cpa} (s)', 'FontSize', 9);
ylabel(ax_center, 'r_{cpa} (m)', 'FontSize', 9);
zlabel(ax_center, '\beta', 'FontSize', 9);
% title removed for LaTeX caption
axis(ax_center, 'tight');
daspect(ax_center, [1000, 200, 0.25]);
view(ax_center, 135, 28);
camlight('headlight'); lighting gouraud; material dull;

% 色标移至右侧，彻底避免底部截断
cb = colorbar(ax_center, 'Location', 'eastoutside');
cb.Label.String = 'Pearson correlation coefficient';
cb.Label.FontSize = 9;

% ---------- (d) 右侧：固定 r0，t0-β 平面 ----------
ax_right = axes('Parent', fig_all, 'Position', [0.72, 0.14, 0.20, 0.50]);
pcolor(ax_right, beta_search, CPA_time_search, slice_t0_beta);
shading interp; axis tight; box on; grid on;
hold(ax_right, 'on');
plot(ax_right, beta_real, CPA_time_real, 'p', 'MarkerSize', 11, ...
    'MarkerFaceColor', MUTED_RED, 'MarkerEdgeColor', 'w', 'LineWidth', 1);
hold(ax_right, 'off');
clim(ax_right, [cmin, cmax]);
colormap(ax_right, parula);
xlabel(ax_right, '\beta', 'FontSize', 9);
ylabel(ax_right, 't_{cpa} (s)', 'FontSize', 9);
% title removed for LaTeX caption
text(ax_right, beta_real+0.015, CPA_time_real+30, ...
    sprintf('(%.3f, %d)', beta_real, CPA_time_real), 'Color', 'w', 'FontSize', 8);

% annotation title removed for LaTeX caption

% ---------- 保存 ----------
set(fig_all, 'PaperUnits', 'inches');
set(fig_all, 'PaperPosition', [0 0 13 7.5]);
set(fig_all, 'PaperPositionMode', 'manual');
print(fig_all, 'fig_3view_final.png', '-dpng', sprintf('-r%d', DPI), '-opengl');
fprintf('图2 (三视图最终版) 已保存\n');

%% ==================== SWellEx-96 Event S5 坐标与时间数据准备 ====================
VLA = [32 + 40.254/60, 117 + 21.620/60];  % VLA position [lat, lon]

% Load Event S5 GPS trajectory data
S5 = load('E:\Moving\数据制作\SW96_数据集\RangeEventS5\EventS5.txt');

%% ==================== 时间基转换（micro mariner -> GPS） ====================
fprintf('Converting micro mariner time to GPS time...\n');

for k = 1:length(S5)
   S5(k,3) = S5(k,3) + 1;  % Correct 1 minute (theoretical 1.05 min, limited by data precision)

   if (S5(k,3) >= 60) 
        S5(k,3) = S5(k,3) - 60;
        S5(k,2) = S5(k,2) + 1;
        if (S5(k,2) >= 24)
           S5(k,2) = S5(k,2) - 24;
           S5(k,1) = S5(k,1) + 1;  % Fixed original code bug
        end
   end
end

% Extract coordinates and calculate great-circle distance
S5Lat = S5(:,4) + S5(:,5)/60;
S5Lon = S5(:,6) + S5(:,7)/60;
VLAlat = ones(length(S5),1) * VLA(1);
VLAlon = ones(length(S5),1) * VLA(2);
S5d = distance(VLAlat, VLAlon, S5Lat, S5Lon, 'degrees');
VLAS5km = deg2km(S5d);

% Build time axis in seconds from start of Event S5
time  = S5(:,1)*1440 + S5(:,2)*60 + S5(:,3);
start = S5(1,1)*1440 + S5(1,2)*60 + S5(1,3);
tline = (time - ones(length(time),1)*start)*60;

%% ==================== 声信号与谱图参数 ====================
filename = 'J1312315.vla.21els.sio';
fs = 1500;
duration_min = 75;
npi = duration_min * 60 * fs;

% Original acoustic signal time axis (full resolution)
t_signal_sec_full = (0:npi-1)' / fs;

%% ==================== 时间对齐与插值（0.5秒间隔） ====================
fprintf('Aligning time axis, interpolation interval set to 0.5 s (2 Hz)...\n');

dt_target = 0.5;  % Target sampling interval in seconds
t_target_sec = 0:dt_target:(duration_min*60);  % 0, 0.5, 1.0, ... 4500 s

% Interpolate GPS distance onto the unified 0.5-second grid
VLAS5km_interp = interp1(tline, VLAS5km, t_target_sec, 'linear', 'extrap');

% Ensure consistent array lengths
min_len = min([length(t_target_sec), length(VLAS5km_interp)]);
t_target_sec = t_target_sec(1:min_len);
VLAS5km_interp = VLAS5km_interp(1:min_len);

%% ==================== 速度计算（0.5秒间隔，1分钟平滑） ====================
fprintf('Calculating velocity (based on 0.5s interval, 1-minute smoothing)...\n');

% Differential velocity (km/s -> m/s)
dt_interp = diff(t_target_sec); 
dr_interp = diff(VLAS5km_interp);
velocity_calc = abs((dr_interp ./ dt_interp) * 1000);  % m/s

% 1-minute moving average: 120 points at 0.5-second interval
window_pts = round(1*60 / dt_target);
velocity_smooth = movmean(velocity_calc, window_pts);

%% ==================== T-49-13 音型与频率对配置 ====================
T49 = [
    49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388;
    52, 67, 82, 97, 115, 133, 151, 169, 204, 238, 286, 341, 391;
    55, 70, 85, 100, 118, 136, 154, 172, 207, 241, 289, 344, 394;
    58, 73, 88, 103, 121, 139, 157, 175, 210, 244, 292, 347, 397;
    61, 76, 91, 106, 124, 142, 160, 178, 213, 247, 295, 350, 400
];
T49_choose = T49(1:2,:);

min_f = 250;
max_f = 400;
min_g = 3;
max_g = 50;

% NOTE: generateFreqPairs is an external helper function assumed available on path
pairs = generateFreqPairs(T49_choose, min_f, max_f, min_g, max_g);

%% ==================== 仿真参数与信号处理 ====================
Time_len = 800;
Tcpa_min = -3000;
Tcpa_max = 3000;
freq_min = 200;
freq_max = 400;
range_max = 1.5068e+04;
range_min = -range_max;
velocity_min = 1;
velocity_max = 5;
sampling_rate_1 = 2;

time_list = (1:Time_len*sampling_rate_1) / sampling_rate_1;
actual_data_length = length(time_list);

% STFT parameters
n = 1;
nett = fs * 0.5;
N = n * fs;

% Read raw acoustic signal
Signal_cpa = sioread(filename, 1, npi, 1);

% Spectrogram: window=hanning(N), overlap=N-nett, fft=N, fs=1500
% Time resolution: (N-nett)/fs = 0.5 s
[P, F, T_STFT] = spectrogram(Signal_cpa, hanning(N), N-nett, N, fs);

% Load SSP data
load("E:\Moving\数据制作\仿真数据制作\karkenc\SSP_data.mat");
SSP = SSP_data(2, 1:2:end, :);
SSP = squeeze(SSP);

% Row-wise average of non-zero elements
nonzero_counts = sum(SSP ~= 0, 2);
row_sums = sum(SSP, 2);
row_means = row_sums ./ nonzero_counts;
row_means(nonzero_counts == 0) = 0;

% Segment extraction parameters
Overlap_ratio = 0.95;
Cpa_exclusion = 1200;  % seconds, recommended 1.5~3x Time_len

samples_per_seg = round(Time_len / dt_target);
step_samples = round(samples_per_seg * (1 - Overlap_ratio));
N_total = length(T_STFT);
valid_segments = [];  % [start_idx, end_idx]

% Determine CPA time at VLA (closest point of approach)
T_cpa_vla = t_target_sec(VLAS5km_interp == min(VLAS5km_interp));

% Sliding window extraction, excluding segments near CPA
idx = 1;
while idx + samples_per_seg - 1 <= N_total
    start_idx = idx;
    end_idx = idx + samples_per_seg - 1;

    t_start = T_STFT(start_idx);
    t_end = T_STFT(end_idx);
    t_center = (t_start + t_end) / 2;

    dist_start = abs(t_start - T_cpa_vla);
    dist_center = abs(t_center - T_cpa_vla);

    if dist_start > Cpa_exclusion 
        valid_segments = [valid_segments; start_idx, end_idx];
    end

    idx = idx + step_samples;
end

num_segs = size(valid_segments, 1);
fprintf('\nTotal %d valid time segments extracted, excluding data within %.1f s of CPA=%.2f s\n', ...
    num_segs, Cpa_exclusion, T_cpa_vla);

for i_seg = 1:num_segs
    fprintf('Segment %d: Index [%d:%d], Time [%.2f:%.2f] s\n', i_seg, ...
        valid_segments(i_seg,1), valid_segments(i_seg,2), ...
        T_STFT(valid_segments(i_seg,1)), T_STFT(valid_segments(i_seg,2)));
end

%% ==================== 图3：高亮有效时段（粗线标记） ====================
fig_all = figure('Name', '', ...
    'Position', [100 100 1400 900]);

% Overall selected time range for legend annotation
if num_segs > 0
    t_sel_start = T_STFT(valid_segments(1, 1));
    t_sel_end   = T_STFT(valid_segments(end, 2));
else
    t_sel_start = NaN; 
    t_sel_end   = NaN;
end

% --- Subplot 1: Distance-Time Curve ---
subplot(2,1,1);
% All data (thin)
plot(t_target_sec, VLAS5km_interp, '-', 'Color', MUTED_BLUE, 'LineWidth', 2.0);
hold on;

% Highlight valid segments (thick)
t_seg_start = T_STFT(valid_segments(1, 1));
t_seg_end   = T_STFT(valid_segments(end, 2));
idx_highlight = (t_target_sec >= t_seg_start) & (t_target_sec <= t_seg_end);

plot(t_target_sec(idx_highlight), VLAS5km_interp(idx_highlight), '-.', 'Color', MUTED_RED, 'LineWidth', 6.0);

legend('All Data','Choose data', 'Location', 'best');
xlabel('Time (s)');
ylabel('Distance (km)');
% title removed for LaTeX caption

grid on;

% --- Subplot 2: Smoothed Velocity with Highlighted Segments ---
subplot(2,1,2);
t_vel = t_target_sec(1:end-1);

% All data (thin)
plot(t_vel, velocity_smooth, '-', 'Color', MUTED_BLUE, 'LineWidth', 2.0);
hold on;
t_seg_start = T_STFT(valid_segments(1, 1));
t_seg_end   = T_STFT(valid_segments(end, 2));
idx_highlight = (t_target_sec >= t_seg_start) & (t_target_sec <= t_seg_end);
plot(t_vel(idx_highlight), velocity_smooth(idx_highlight), '-.', 'Color', MUTED_RED, 'LineWidth', 6.0);
legend('All Data','Choose data', 'Location', 'best');

xlabel('Time (s)');
ylabel('Velocity (m/s)');
% title removed for LaTeX caption

grid on;
print(fig_all, 'choose_data.png', '-dpng', sprintf('-r%d', DPI), '-image');

%% ==================== 声速剖面图（12环境） ====================
load("E:\Moving\数据制作\仿真数据制作\远场径向\SSP_data_12.mat");

%% 筛选有效环境（水深 > 180 m）
SSP_data_depth = squeeze(SSP_data(1, :, :));               % [depth_points x env_count]
col_indices = find(max(SSP_data_depth, [], 1) > 180);
num_env = length(col_indices);

% 提取所有环境的深度与声速矩阵 [depth_points x num_env]
SSP_all   = squeeze(SSP_data(2, :, col_indices));
depth_all = squeeze(SSP_data(1, :, col_indices));

%% 绘图
fig_all = figure('Name', '', ...
    'Position', [100 100 550 700], 'Color', 'w');
hold on; box on;

% 计算非零值的平均声速剖面（按深度层逐行平均）
nonzero_mask = SSP_all ~= 0;
row_sums   = sum(SSP_all .* double(nonzero_mask), 2);
row_counts = sum(nonzero_mask, 2);
mean_SSP   = row_sums ./ row_counts;        % 各深度层非零声速均值
mean_depth = mean(depth_all, 2);            % 深度网格基本一致，直接平均

% 剔除全零深度层（无数据）
valid_mean = ~isnan(mean_SSP);
mean_SSP   = mean_SSP(valid_mean);
mean_depth = mean_depth(valid_mean);

% 绘制平均声速剖面（柔和红色粗线，唯一进入图例的曲线）
plot(mean_SSP, 0:0.5:max(max(depth_all)), '-.', 'Color', MUTED_RED, 'LineWidth', 3);
% 绘制12条环境声速剖面（深灰色细线，无图例条目）
for k = 1:num_env
    ssp = SSP_all(:, k);
    z   = depth_all(:, k);
    valid = ssp ~= 0;
    plot(ssp(valid), z(valid), '-', 'Color', MUTED_GRAY, 'LineWidth', 0.8, 'HandleVisibility', 'off');
end

%% 坐标轴与图例规范
set(gca, 'YDir', 'reverse');                % 0 m 在上，深度向下增加
xlabel('Sound Speed (m/s)', 'FontSize', 11);
ylabel('Depth (m)', 'FontSize', 11);

% 仅对平均线添加图例
legend('Mean SSP', 'Location', 'best', 'FontSize', 10);

% 坐标范围与刻度美化（参考上传图片风格）
set(gca, 'FontSize', 10, 'LineWidth', 1);
xlim([1485 1525]);
ylim([0 220]);

hold off;
print(fig_all, 'SSP_data.png', '-dpng', sprintf('-r%d', DPI), '-image');
