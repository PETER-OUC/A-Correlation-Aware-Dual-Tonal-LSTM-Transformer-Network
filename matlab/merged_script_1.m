clc;
clear ;
close all;

%% ==================== 全局超参数 ====================
DPI = 600;                  % 统一图像输出分辨率（可在此修改）

%% ==================== 图形全局设置 ====================
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextFontSize', 12);

%% ==================== 第一部分：FDM 径向速度估计 ====================
f = 300;
c0 = 1500;

f1ss = 2;%%对应t1的采样率
t1 = 1/f1ss:1/f1ss:1000;

v = 3;
t0 = 1500;
r0 = 1;
D_P1 = zeros(10,length(t1));

Nz = 81;%深度网格
fs = 10000;%采样率
z0 = 6; %声源深度位置

c = 1500;%水下声传播速度

t = t1 - t0;
r = sqrt(r0.^2 + (v.*t).^2);%计算每个时间点对应的离水听器的距离

z = 7;%水听器所在的深度位置
%% 读取克拉肯计算声场并画干涉结构
txt_name = ['txt_',num2str(55)];

num_receiver = 41;
fname1_new = ['G:\Files\Moving\数据制作\仿真数据制作\',txt_name,'\','k',num2str(f),'.txt'];
fname2_new = ['G:\Files\Moving\数据制作\仿真数据制作\',txt_name,'\','phi',num2str(f),'.txt'];
eigenv = load(fname1_new);
eigenf = load(fname2_new);
fclose all;
k_z = eigenv(:,2) + 1i*eigenv(:,3);
phi1 = reshape(eigenf(:,1) + 1i*eigenf(:,2), num_receiver, length(eigenf)/num_receiver);

nm_index = length(k_z);

clear pf
pf1 = zeros(1,length(r));
for kk = 1:nm_index
    pf = exp(-1i*pi/4)/4 * (phi1(z0,kk).*phi1(z,kk).*(exp(-1i*k_z(kk).*r))./sqrt(k_z(kk).*r));
    pf1 = pf1 + pf;
end

pz(1,:) = pf1;

%%
pzz = awgn(pz,10,'measured');
pz__1 = pzz;

T = t1;
kz = 200;
fs12 = 1;%对应选取间隔
dt = fs12:fs12:kz;

for k = 1:(length(pz__1)-kz)
    u1 = pz__1(k);
    qq = pz__1(dt+k);
    dt_z1 = (qq - u1);
    dt_z = real((dt_z1).*conj(dt_z1));
    pp(:,k) = dt_z';
end

fs1 = f1ss/fs12;%这是选取时间段内的最终采样率
f1 = (1:length(dt))/length(dt)*fs1;

for l = 1:k
    df1(:,l) = fft(pp(:,l) - max(pp(:,l))/2);
end
vs1 = f1/f*c;

% 真实径向速度
t11 = T(1:(length(pz__1)-kz));
r11 = sqrt(v.^2*(t11-t0).^2 + r0.^2);
v11 = v^2*abs(t11-t0)./r11;

% ---------- 规范绘图：FDM 速度估计 ----------
fig_fdm = figure('Color', 'w', 'Position', [100 100 900 700], 'Renderer', 'painters');

Z_fdm = abs(df1(1:round(length(vs1)/2),:)) ./ max(abs(df1(1:round(length(vs1)/2),:)));
X_fdm = t11;
Y_fdm = vs1(1:round(length(vs1)/2));

pcolor(X_fdm, Y_fdm, Z_fdm);
shading interp;
colormap(parula);
axis tight;
box on;

hold on;

% 真实速度曲线（红色实线，粗线）
plot(t11, v11, '-', 'Color', [0.55 0.25 0.30], 'LineWidth', 3);

% colorbar
hcb = colorbar('Location', 'eastoutside');
hcb.Label.String = 'Normalized Amplitude';
hcb.Label.FontSize = 12;
hcb.FontSize = 11;

% 标题与坐标轴（与上传图片一致）
% title removed for LaTeX caption
xlabel('Time (s)', 'FontSize', 13);
ylabel('v (m/s)', 'FontSize', 13);
ylim([0.2 5])
% 坐标范围与刻度（参考图片风格）

set(gca, 'LineWidth', 1.2, 'Layer', 'top', 'GridLineStyle', '-', 'GridAlpha', 0.3);

% 图例
legend('', 'True velocity', 'Location', 'best', 'FontSize', 11, 'Color', 'w');

% 保存
set(fig_fdm, 'PaperUnits', 'inches');
set(fig_fdm, 'PaperPosition', [0 0 9 7]);
set(fig_fdm, 'PaperPositionMode', 'manual');
print(fig_fdm, 'FDM_Velocity_Estimation.png', '-dpng', sprintf('-r%d', DPI), '-image');
fprintf('FDM 速度估计图已保存\\n');

%% ==================== 第二部分：双线谱干涉结构对比 ====================


% 重新加载全局设置（因前面 clear 了）
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextFontSize', 12);

%% ==================== 参数配置 ====================
env_number = 55;
Frequency_1 = 280;
Frequency_2 = 300;
DPI = 600;
CPA_range_real = 900;       % m
velocity_real  = 2.5;       % m/s
CPA_time_real = 2000;       % s

Time_len = 800;
sampling_rate_1 = 2;
time_list = (1:Time_len*sampling_rate_1) / sampling_rate_1;
time_relative_to_CPA = time_list - CPA_time_real;
range_list_real = sqrt(CPA_range_real^2 + (velocity_real * time_relative_to_CPA).^2);

%% ==================== 加载本征数据 ====================
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

k_1 = eigenvals_1(:, 2) + 1i * eigenvals_1(:, 3);
phi_1 = reshape(eigenfuncs_1(:, 1) + 1i * eigenfuncs_1(:, 2), num_receivers, []);
num_modes_1 = length(k_1);

k_2 = eigenvals_2(:, 2) + 1i * eigenvals_2(:, 3);
phi_2 = reshape(eigenfuncs_2(:, 1) + 1i * eigenfuncs_2(:, 2), num_receivers, []);
num_modes_2 = length(k_2);

%% ==================== 合成声场并加噪 ====================
pressure_field_F1_sim = 1i * exp(-1i * pi / 4) / sqrt(8 * pi) * ...
    (phi_1(receiver_depth, 1:num_modes_1) .* phi_1(source_depth, 1:num_modes_1)) * ...
    (exp(-1i * k_1(1:num_modes_1) .* range_list_real) ./ ...
     sqrt(k_1(1:num_modes_1) .* range_list_real));

pressure_field_F2_sim = 1i * exp(-1i * pi / 4) / sqrt(8 * pi) * ...
    (phi_2(receiver_depth, 1:num_modes_2) .* phi_2(source_depth, 1:num_modes_2)) * ...
    (exp(-1i * k_2(1:num_modes_2) .* range_list_real) ./ ...
     sqrt(k_2(1:num_modes_2) .* range_list_real));

% SNR = 10;
% pressure_field_F1_sim = awgn(pressure_field_F1_sim, SNR, "measured");
% pressure_field_F2_sim = awgn(pressure_field_F2_sim, SNR, "measured");

%% ==================== 功率谱（去均值） ====================
data_F1_T_sim = abs(pressure_field_F1_sim .* conj(pressure_field_F1_sim));
data_F2_T_sim = abs(pressure_field_F2_sim .* conj(pressure_field_F2_sim));
data_F1_T_long_1 = data_F1_T_sim - mean(data_F1_T_sim);
data_F2_T_long_1 = data_F2_T_sim - mean(data_F2_T_sim);

%% ==================== 估计真实 beta ====================
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
fprintf('真值参考：beta=%.3f, v=%.2f m/s, r0=%.0f m, t0=%.0f s\\n', ...
    beta_real, velocity_real, CPA_range_real, CPA_time_real);

%% ==================== 双线谱对比绘图 ====================
% 使用固定 beta = 0.986 进行伸缩变换（与代码一致）
beta_plot = 0.986;
scale = (Frequency_1 / Frequency_2)^(1 / beta_plot);
data_fit = interp1(range_list_real, range_list_real .* data_F1_T_long_1, range_list_real * scale);
data_fit(isnan(data_fit)) = 0;
temp = data_fit ./ (range_list_real .* scale);
X = temp - mean(temp);

% 归一化到 [0, 1]
norm01 = @(x) (x - min(x)) / (max(x) - min(x));
y1 = norm01(data_F1_T_long_1(:))';      % F1 原始
y2 = norm01(X(:))';                     % F1 经 beta 伸缩变换
y3 = norm01(data_F2_T_long_1(:))';      % F2 原始

% Y方向错开偏移（与图片一致）
offset1 = 0;    % 红色实线基线
offset2 = 1.0;  % 蓝色虚线基线
offset3 = 2.0;  % 黄绿色虚线基线

x_range = range_list_real(:)';

fig_dual = figure('Color', 'w', 'Position', [100 100 900 700], 'Renderer', 'painters');
hold on; box on; grid on;

% 绘制三条曲线
h1 = plot(x_range, y1 + offset1, '-', 'Color', [0.55 0.25 0.30], 'LineWidth', 2.5);
h2 = plot(x_range, y2 + offset2, '--', 'Color', [0.45 0.55 0.70], 'LineWidth', 2.5);
h3 = plot(x_range, y3 + offset3, '--', 'Color', [0.65 0.70 0.40], 'LineWidth', 2.5);

% ---------- 添加箭头（连接对应峰值） ----------
% 找 F1 原始曲线的前 3 个显著峰值
d1 = diff(y1);
peak_idx = find(d1(1:end-1) > 0 & d1(2:end) < 0) + 1;
[~, ord] = sort(y1(peak_idx), 'descend');
peak_idx = peak_idx(ord(1:min(3, length(ord))));

% 获取当前 axes 的归一化位置（用于 annotation 坐标转换）
ax = gca;
ax.Units = 'normalized';
pos = ax.Position;

% for k = 1:length(peak_idx)
%     idx = peak_idx(k);
%     x_red = x_range(idx);
%     y_red = y1(idx) + offset1;
% 
%     % 蓝色曲线对应位置（按 scale 伸缩后的距离）
%     x_blue_target = x_red * scale;
%     [~, idx_blue] = min(abs(x_range - x_blue_target));
%     x_blue = x_range(idx_blue);
%     y_blue = y2(idx_blue) + offset2;
% 
%     % 黄绿色曲线位置（与蓝色同 X）
%     x_green = x_blue;
%     y_green = y3(idx_blue) + offset3;
% 
%     % 数据坐标 → 归一化 figure 坐标
%     xn_red   = pos(1) + pos(3) * (x_red   - ax.XLim(1)) / diff(ax.XLim);
%     yn_red   = pos(2) + pos(4) * (y_red   - ax.YLim(1)) / diff(ax.YLim);
%     xn_blue  = pos(1) + pos(3) * (x_blue  - ax.XLim(1)) / diff(ax.XLim);
%     yn_blue  = pos(2) + pos(4) * (y_blue  - ax.YLim(1)) / diff(ax.YLim);
%     xn_green = pos(1) + pos(3) * (x_green - ax.XLim(1)) / diff(ax.XLim);
%     yn_green = pos(2) + pos(4) * (y_green - ax.YLim(1)) / diff(ax.YLim);
% 
%     % 红色 → 蓝色（斜向箭头）
%     annotation('arrow', [xn_red, xn_blue], [yn_red, yn_blue], ...
%         'LineWidth', 1.8, 'HeadLength', 10, 'HeadWidth', 10, 'Color', 'k');
% 
%     % 蓝色 → 黄绿色（垂直向上箭头）
%     annotation('arrow', [xn_blue, xn_green], [yn_blue, yn_green], ...
%         'LineWidth', 1.8, 'HeadLength', 10, 'HeadWidth', 10, 'Color', 'k');
% end

% ---------- 图例、标题与坐标轴 ----------
legend([h1, h2, h3], {'F_1 = 280 Hz', 'F_1 to F_2 = 300 Hz (scaled)', 'F_2 = 300 Hz'}, ...
    'Location', 'northeast', 'FontSize', 11, 'Color', 'w');

% title removed for LaTeX caption
xlabel('Range (m)', 'FontSize', 13);
ylabel('Normalized Intensity', 'FontSize', 13);

% 坐标范围与刻度（参考图片风格）
xlim([min(x_range)+0, max(x_range)+50]);
ylim([-0.2 3.2]);
set(gca, 'LineWidth', 1.2, 'Layer', 'top', 'GridLineStyle', '-', 'GridAlpha', 0.3);

% 保存
set(fig_dual, 'PaperUnits', 'inches');
set(fig_dual, 'PaperPosition', [0 0 9 7]);
set(fig_dual, 'PaperPositionMode', 'manual');
print(fig_dual, 'Dual_Line_Spectrum.png', '-dpng', sprintf('-r%d', DPI), '-image');
fprintf('双线谱干涉结构对比图已保存\\n');