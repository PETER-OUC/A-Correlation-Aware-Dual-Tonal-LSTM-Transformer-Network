function freqPairs = generateFreqPairs(T49_choose, min_freq, max_freq, min_gap, max_gap)
% GENERATEFREQPAIRS 基于T49矩阵生成满足约束条件的频率对 (f1 > f2)
% 
% 输入:
%   T49_choose - 频率矩阵 (如 5×13 矩阵)
%   min_freq   - 最小频率限制
%   max_freq   - 最大频率限制  
%   min_gap    - 最小频率间隔
%   max_gap    - 最大频率间隔
%
% 输出:
%   freqPairs  - N×2 矩阵，每行是一个满足条件的频率对 [f1, f2]
%                其中 f1 > f2（f1为高频，f2为低频），且间隔在指定范围内
%                间隔 = f1 - f2，满足 min_gap <= 间隔 <= max_gap

    % 1. 将矩阵转换为唯一且排序的向量（去重并排序）
    allFreqs = unique(sort(T49_choose(:)));
    
    % 2. 筛选在频率范围内的值
    validFreqs = allFreqs(allFreqs >= min_freq & allFreqs <= max_freq);
    
    % 3. 检查是否有足够的频率点
    if length(validFreqs) < 2
        freqPairs = zeros(0, 2);
        warning('满足频率范围的点少于2个，无法生成组合');
        return;
    end
    
    % 4. 生成所有可能的两两组合（只考虑 f1 > f2 的情况）
    n = length(validFreqs);
    pairs = [];
    
    % 向量化实现：使用网格生成组合
    [F1, F2] = meshgrid(validFreqs, validFreqs);
    gap = F1 - F2;  % 计算间隔：f1 - f2
    
    % 5. 应用间隔约束筛选
    % 条件：min_gap <= gap <= max_gap 且 gap > 0 (确保 f1 > f2)
    mask = (gap >= min_gap) & (gap <= max_gap) & (gap > 0);
    
    % 6. 提取满足条件的对
    f1_valid = F1(mask);  % 高频
    f2_valid = F2(mask);  % 低频
    
    % 7. 组合成输出矩阵 [高频, 低频]
    freqPairs = [f1_valid, f2_valid];
    
    % 8. 按第一列降序（高频从大到小），然后第二列降序排序
    freqPairs = sortrows(freqPairs, [-1, -2]);
    
    fprintf('共找到 %d 组满足条件的频率对 (f1 > f2)\n', size(freqPairs, 1));
end
