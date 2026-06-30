

clc;
clear;
close all;

tic;

loop_sample_number = 4;
Sample_number = 5e3;

velocity_list = 1:0.1:5;

sampling_rate_1 = 2;
Time_len = 800;

CPA_time_list = 1200:10:3000;

beta_search_list = 0.8:0.002:1.3;

T = 1/sampling_rate_1;

sampling_rate_2 = 1;

time_list = (1:Time_len*sampling_rate_1) / sampling_rate_1;
actual_data_length = length(time_list);

% T-49-13音调模式
T49 = [
    49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388;
    52, 67, 82, 97, 115, 133, 151, 169, 204, 238, 286, 341, 391;
    55, 70, 85, 100, 118, 136, 154, 172, 207, 241, 289, 344, 394;
    58, 73, 88, 103, 121, 139, 157, 175, 210, 244, 292, 347, 397;
    61, 76, 91, 106, 124, 142, 160, 178, 213, 247, 295, 350, 400
];

T49_choose = T49(1:5,:);
min_f = 200;
max_f = 400;
min_g = 3;
max_g = 50;

pairs = generateFreqPairs(T49_choose, min_f, max_f, min_g, max_g);
num_pairs = size(pairs, 1);

% 获取所有可能用到的唯一频率（预加载目标）
unique_freqs = unique([pairs(:,1); pairs(:,2)]);

beta_min = min(beta_search_list);
beta_max = max(beta_search_list);

CPA_range_list = 800:5:1200;  

t_start = 1 / sampling_rate_1;
dt_min = min(CPA_time_list) - t_start;
dt_max = max(CPA_time_list) + t_start;

receiver_list = round([178.49,172.88,167.26,161.62,155.99,150.38,144.74,139.12,127.88,122.25,116.62,111.00,105.38,99.755,94.125]*2);

% range_min = sqrt(min(CPA_range_list)^2 + (min(velocity_list) * dt_min)^2);

range_max = sqrt(max(CPA_range_list)^2 + (max(velocity_list) * dt_max)^2);
range_min = -range_max;

T_min = -max(CPA_time_list); 
T_max =  max(CPA_time_list);

freq_min = min(min(pairs(:,1)), min(pairs(:,2)));
freq_max = max(max(pairs(:,1)), max(pairs(:,2)));

velocity_min = min(velocity_list);
velocity_max = max(velocity_list);

load("E:\\Moving\\数据制作\\仿真数据制作\\karkenc\\SSP_data_12.mat")

SSP_data_depth= squeeze(SSP_data(1, :, :));
col_indices =find(max(SSP_data_depth, [], 1) > 180);

disp(col_indices);
SSP_data = SSP_data(:,:,col_indices);

max_SSP = max(max(SSP_data(2, :, :)));
min_SSP =min(nonzeros(SSP_data(2, :, :)));

CoreNum = 20;
if isempty(gcp('nocreate'))
    parpool(CoreNum);
end

dir_name = 'E:\Moving\数据制作\仿真数据制作\远场径向\data_moving_2\';  % 新目录避免覆盖
if exist(dir_name, 'dir') == 7
    rmdir(dir_name, 's');
end
mkdir(dir_name);
SEQ_LEN = 400;
SEQ_LEN_2 = 1400;   

for env_number = 1:size(SSP_data,3)

    environment_name = ['txt_', num2str(env_number)];
    txt_name = ['txt_', num2str(env_number)];
    Water_depth = SSP_data(1, :, env_number);
    SSP = SSP_data(2, :, env_number);
    Water_depth = squeeze(Water_depth);
    SSP = squeeze(SSP);
    SSP(SSP == 0) = [];
    Water_depth = Water_depth(1:length(SSP));

    num_receivers = Water_depth(end)*2+1; 
 
    data_real_len = actual_data_length;
    channels_needed_for_data = ceil(data_real_len / SEQ_LEN);
    
    SSP_original = SSP(1:2:end);
    SSP_into = [SSP_original, zeros(1, SEQ_LEN - length(SSP_original))];
    ssp_len = length(SSP_into);
    ssp_channels_needed = ceil(ssp_len / SEQ_LEN);

    % ===== 预加载所有需要的频率数据（每个环境只执行一次）=====
    fprintf('环境 %d: 预加载 %d 个频率的本征数据...\n', env_number, length(unique_freqs));
    
    % 使用 containers.Map 存储，便于快速查找
    k_map = containers.Map('KeyType', 'double', 'ValueType', 'any');
    phi_map = containers.Map('KeyType', 'double', 'ValueType', 'any');
    num_modes_map = containers.Map('KeyType', 'double', 'ValueType', 'double');
    
    for f = unique_freqs'
        eigenvalue_file = ['E:\\Moving\\数据制作\\仿真数据制作\\karkenc\\', environment_name, '\\', 'k', num2str(f), '.txt'];
        eigenfunction_file = ['E:\\Moving\\数据制作\\仿真数据制作\\karkenc\\', environment_name, '\\', 'phi', num2str(f), '.txt'];
        
        eigenvals = load(eigenvalue_file);
        eigenfuncs = load(eigenfunction_file);
        
        k_data = eigenvals(:, 2) + 1i * eigenvals(:, 3);
        phi_data = reshape(eigenfuncs(:, 1) + 1i * eigenfuncs(:, 2), ...
                          num_receivers, length(eigenfuncs) / num_receivers);
        
        k_map(f) = k_data;
        phi_map(f) = phi_data;
        num_modes_map(f) = length(k_data);
    end
    fprintf('环境 %d: 预加载完成\n', env_number);
    % ==========================================================

    for loop_one = 1:loop_sample_number
        base_channels = 6;
        total_data_channels = 5;
        total_label_channels = 5;
        
        attempt_multiplier = 4;  
        attempts = ceil(Sample_number * attempt_multiplier);
        
        all_data_cells = cell(attempts, 1);
        all_data_2_cells = cell(attempts, 1);
        all_label_cells = cell(attempts, 1);
        is_valid = false(attempts, 1);
        
        % ===== 关键修改1：用cell数组存储距离值（parfor兼容）=====
        range_begin_cells = cell(attempts, 1);
        
        rng('shuffle');
        parfor sample_one = 1:attempts 
            temp_data = zeros(SEQ_LEN, 5);
            temp_data_2 = zeros(SEQ_LEN_2, 2);
            temp_label = zeros(SEQ_LEN, total_label_channels);
            
            % 随机选择频率对
            pair_idx = randi(num_pairs, 1, 1);
            Frequency_1 = pairs(pair_idx, 1);
            Frequency_2 = pairs(pair_idx, 2);
        
            velocity_real = velocity_list(randi(numel(velocity_list), 1, 1));
            CPA_time = CPA_time_list(randi(numel(CPA_time_list), 1, 1)) * (2 * randi([0, 1]) - 1);
            CPA_range = CPA_range_list(randi(numel(CPA_range_list), 1, 1));

            time_relative_to_CPA = time_list - CPA_time;
            range_list_real = sqrt(CPA_range^2 + (velocity_real * time_relative_to_CPA).^2);
            range_list_begin = range_list_real(1);
            if CPA_time < 0
                range_list_begin = -range_list_begin;
            end
            
            
            receiver_depth = receiver_list(randi(numel(receiver_list), 1, 1)) + randi([-5, 5]);
            source_depth = 108+randi([-3, 3]);
            
            if  receiver_depth > num_receivers
                receiver_depth = num_receivers - 1;
            end
            
            % 获取本征数据
            k_1 = k_map(Frequency_1);
            phi_1 = phi_map(Frequency_1);
            num_modes_1 = num_modes_map(Frequency_1);
            k_2 = k_map(Frequency_2);
            phi_2 = phi_map(Frequency_2);
            num_modes_2 = num_modes_map(Frequency_2);
            
            pressure_field_F1 = 1i * exp(-1i * pi / 4) / sqrt(8 * pi) * ...
                (phi_1(receiver_depth, 1:num_modes_1) .* ...
                 phi_1(source_depth, 1:num_modes_1)) * ...
                (exp(-1i * k_1(1:num_modes_1) .* range_list_real) ./ ...
                 sqrt(k_1(1:num_modes_1) .* range_list_real));
            
            pressure_field_F2 = 1i * exp(-1i * pi / 4) / sqrt(8 * pi) * ...
                (phi_2(receiver_depth, 1:num_modes_2) .* ...
                 phi_2(source_depth, 1:num_modes_2)) * ...
                (exp(-1i * k_2(1:num_modes_2) .* range_list_real) ./ ...
                 sqrt(k_2(1:num_modes_2) .* range_list_real));
            
            % temp_data_2(:, 1) =real(pressure_field_F1./abs(pressure_field_F1));
            % temp_data_2(:, 2) =real(pressure_field_F2./abs(pressure_field_F2));

            
            data_F1_T_long = pressure_field_F1 .* conj(pressure_field_F1);
            data_F2_T_long = pressure_field_F2 .* conj(pressure_field_F2);
            
            data_F1_T_long = data_F1_T_long - mean(data_F1_T_long);
            data_F2_T_long = data_F2_T_long - mean(data_F2_T_long);
            
            SNR = randi([5, 10]);
            
            loss_1 = zeros(1, length(beta_search_list));
            
            for beta_one = 1:length(beta_search_list)
                data_F1_T_fit_F2 = interp1(range_list_real, range_list_real .* data_F1_T_long, ...
                                          range_list_real * (Frequency_1 / Frequency_2)^(1 / beta_search_list(beta_one)));
                data_F1_T_fit_F2(isnan(data_F1_T_fit_F2)) = 0;
                temp_1 = data_F1_T_fit_F2 ./ (range_list_real .* (Frequency_1 / Frequency_2)^(1 / beta_search_list(beta_one)));
                X_F1 = temp_1 - mean(temp_1);
                loss_1(1, beta_one) = corr(data_F2_T_long(:), X_F1(:));
            end
            
            [~, index_beta] = max(loss_1);
            beta_choose_best = beta_search_list(index_beta);
            if beta_choose_best < 0.85 || beta_choose_best > 1.2
                all_data_cells{sample_one} = [];
                all_label_cells{sample_one} = [];
                all_data_2_cells{sample_one} = [];
                is_valid(sample_one) = false;
                range_begin_cells{sample_one} = [];
                continue;
            end
            
            pressure_field_F1 = awgn(pressure_field_F1, SNR, "measured");
            pressure_field_F2 = awgn(pressure_field_F2, SNR, "measured");

            kz=1400;
            fs12=1;%对应选取间
            dt=fs12:fs12:kz;
            
            pressure_field_F1_temp = pressure_field_F1./abs(pressure_field_F1);
            pressure_field_F1_temp = pressure_field_F1_temp(dt+1) -pressure_field_F1_temp(1);
            pressure_field_F1_temp = real(pressure_field_F1_temp.*conj(pressure_field_F1_temp));
            pressure_field_F1_temp = pressure_field_F1_temp -mean(pressure_field_F1_temp);

            pressure_field_F2_temp = pressure_field_F2./abs(pressure_field_F2);
            pressure_field_F2_temp = pressure_field_F2_temp(dt+1) -pressure_field_F2_temp(1);
            pressure_field_F2_temp = real(pressure_field_F2_temp.*conj(pressure_field_F2_temp));
            pressure_field_F2_temp = pressure_field_F2_temp -mean(pressure_field_F2_temp);

            temp_data_2(:, 1) =mapminmax(pressure_field_F1_temp,-1,1);
            temp_data_2(:, 2) =mapminmax(pressure_field_F2_temp,-1,1);
            data_F1_T_long = abs(pressure_field_F1 .* conj(pressure_field_F1));
            data_F2_T_long = abs(pressure_field_F2 .* conj(pressure_field_F2));
            
            data_F1_T_long = data_F1_T_long - mean(data_F1_T_long);
            data_F2_T_long = data_F2_T_long - mean(data_F2_T_long);
            
            data_F1_T_fit_F2 = interp1(range_list_real, range_list_real .* data_F1_T_long, ...
                                      range_list_real * (Frequency_1 / Frequency_2)^(1 / beta_choose_best));
            data_F1_T_fit_F2(isnan(data_F1_T_fit_F2)) = 0;
            temp_1 = data_F1_T_fit_F2 ./ (range_list_real .* (Frequency_1 / Frequency_2)^(1 / beta_choose_best));
            
            zero_ratio = sum(temp_1 == 0) / length(temp_1);
            if zero_ratio > 0.5 || any(isnan(temp_1)) || any(isinf(temp_1))
                all_data_cells{sample_one} = [];
                all_data_2_cells{sample_one} = [];
                all_label_cells{sample_one} = [];
                is_valid(sample_one) = false;
                range_begin_cells{sample_one} = [];
                continue;
            end
            
            norm_freq1 = 2 * (Frequency_1 - freq_min) / (freq_max - freq_min) - 1;
            norm_freq2 = 2 * (Frequency_2 - freq_min) / (freq_max - freq_min) - 1;         
            temp_data(:, 1) = norm_freq1 * ones(SEQ_LEN, 1);
            temp_data(:, 2) = norm_freq2 * ones(SEQ_LEN, 1);
            
            data_F1_real = mapminmax(abs(pressure_field_F1.*2), -1, 1);
            data_F2_real = mapminmax(abs(pressure_field_F2.*2), -1, 1);

            mask = SSP_into ~= 0;
            ssp_normalized = zeros(size(SSP_into));
            ssp_normalized(mask) = 2 * (SSP_into(mask) - min_SSP) / (max_SSP-min_SSP) - 1;
            
            temp_data(:, 3) = data_F1_real(1:4:end)';
            temp_data(:, 4) = data_F2_real(1:4:end)';
            temp_data(:, 5) = ssp_normalized';
            
            norm_beta = 2 * (beta_choose_best - beta_min) / (beta_max - beta_min) - 1;
            norm_T = 2 * (CPA_time - T_min) / (T_max - T_min) - 1;
            norm_range = 2 * (range_list_begin - range_min) / (range_max - range_min) - 1;
            norm_velocity = 2 * (velocity_real - velocity_min) / (velocity_max - velocity_min) - 1;

            mask = temp_1 ~= 0;
            data_F1_T_long_norm = zeros(size(temp_1));
            data_F1_T_long_norm(mask) = mapminmax(temp_1(mask), -1, 1);
            data_F2_T_long_norm = mapminmax(data_F2_T_long, -1, 1);
            
            temp_label(:, 1) = data_F1_T_long_norm(1:4:end);
            temp_label(:, 2) = data_F2_T_long_norm(1:4:end);
            temp_label(:, 3) = norm_T * ones(SEQ_LEN, 1);
            temp_label(:, 4) = norm_range * ones(SEQ_LEN, 1);
            temp_label(:, 5) = norm_velocity * ones(SEQ_LEN, 1);
            
            all_data_cells{sample_one} = temp_data;
            all_data_2_cells{sample_one} = temp_data_2;
            all_label_cells{sample_one} = temp_label;
            is_valid(sample_one) = true;
            range_begin_cells{sample_one} = range_list_begin;  % 保存原始距离值
        end
        
        %% ===== 关键修改2：分层抽样逻辑（替换原有的先到先得）=====
        valid_indices_all = find(is_valid);
        n_valid = length(valid_indices_all);
        fprintf('批次 %d: 有效候选样本 %d 个，开始分层抽样...\n', loop_one, n_valid);
        
        if n_valid < Sample_number
            error('有效样本不足 %d 个（仅 %d 个），请增大 attempt_multiplier', Sample_number, n_valid);
        end
        
        % 提取所有有效样本的距离值（从cell转换为数组）
        range_values = zeros(n_valid, 1);
        for i = 1:n_valid
            idx = valid_indices_all(i);
            if ~isempty(range_begin_cells{idx})
                range_values(i) = range_begin_cells{idx};
            end
        end
        
        range_min_actual = min(range_values);
        range_max_actual = max(range_values);
        
        % 分层抽样设置
        num_bins = 25;  % 分成25个区间（可根据需要调整，如20或50）
        bin_edges = linspace(range_min_actual, range_max_actual, num_bins + 1);
        [~, ~, bin_indices] = histcounts(range_values, bin_edges);
        
        % 计算每个区间的目标配额（均匀分配）
        quota_per_bin = floor(Sample_number / num_bins);
        remainder = Sample_number - quota_per_bin * num_bins;
        
        selected_indices = [];
        
        for b = 1:num_bins
            in_bin = find(bin_indices == b);
            candidates = valid_indices_all(in_bin);
            
            if isempty(candidates)
                fprintf('警告：区间 %d 无样本（距离 %.1f-%.1f）\n', b, bin_edges(b), bin_edges(b+1));
                continue;
            end
            
            % 该区间目标数：基础配额 + 前remainder个区间各多1个
            target_count = quota_per_bin + (b <= remainder);
            
            if length(candidates) >= target_count
                % 样本充足：随机选取目标数量
                pick = candidates(randperm(length(candidates), target_count));
                selected_indices = [selected_indices; pick];
            else
                % 样本不足：全选，后续从其他区间补充
                selected_indices = [selected_indices; candidates];
                fprintf('区间 %d 样本不足：目标%d，实际%d\n', b, target_count, length(candidates));
            end
        end
        
        % 补充不足部分（如果有）
        current_count = length(selected_indices);
        if current_count < Sample_number
            deficit = Sample_number - current_count;
            remaining = setdiff(valid_indices_all, selected_indices);
            if length(remaining) >= deficit
                extra = remaining(randperm(length(remaining), deficit));
                selected_indices = [selected_indices; extra];
                fprintf('从剩余候选中补充 %d 个样本\n', deficit);
            else
                error('即使使用所有候选，样本仍不足（缺%d个）', deficit);
            end
        elseif current_count > Sample_number
            % 如果超了（因余数分配），随机剔除多余
            selected_indices = selected_indices(randperm(current_count, Sample_number));
        end
        
        valid_indices = selected_indices;
        
        % 输出分层后统计（调试用）
        selected_ranges = zeros(Sample_number, 1);
        for i = 1:Sample_number
            idx = valid_indices(i);
            selected_ranges(i) = range_begin_cells{idx};
        end
        fprintf('分层完成：距离范围 [%.1f, %.1f]，均值 %.1f，标准差 %.1f\n', ...
            min(selected_ranges), max(selected_ranges), mean(selected_ranges), std(selected_ranges));
        %% ===== 分层抽样结束 =====
        
        data_list = zeros(Sample_number, SEQ_LEN, 5);
        data_list_2 = zeros(Sample_number, SEQ_LEN_2, 2);
        label_list = zeros(Sample_number, SEQ_LEN, total_label_channels);
        
        for i = 1:Sample_number
            idx = valid_indices(i);
            data_list(i, :, :) = all_data_cells{idx};
            label_list(i, :, :) = all_label_cells{idx};
            data_list_2(i, :, :) = all_data_2_cells{idx};
        end
        
        writeNPY(data_list, [dir_name,'env_', num2str(env_number, '%02d'), '_data_list_a_', num2str(loop_one, '%02d'),'_' ,num2str(Sample_number/1000, '%04d'), 'e3', '.npy']);
        writeNPY(data_list_2, [dir_name,'env_', num2str(env_number, '%02d'), '_data_list_b_', num2str(loop_one, '%02d'),'_' ,num2str(Sample_number/1000, '%04d'), 'e3', '.npy']);
        writeNPY(label_list, [dir_name,'env_', num2str(env_number, '%02d'),'_label_list_',num2str(loop_one, '%02d'),'_' ,num2str(Sample_number/1000, '%04d'),'e3', '.npy']);
        
        toc;
        fprintf('批次 %d 完成，保存 %d 个样本（强制距离均匀分布）\n', loop_one, Sample_number);
    end
    
    fprintf('环境 %d 全部完成\n', env_number);
end

fprintf('所有环境处理完成！输出目录：%s\n', dir_name);
toc;