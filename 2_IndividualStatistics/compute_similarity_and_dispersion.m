
%% FIGURE 2: Individual similarity and gradient dispersion
% Purpose:
%   1. Calculate the similarity between individual and group-average gradients.
%   2. Calculate the global dispersion of individual gradients.
%   3. Calculate network-specific gradient dispersion.
%   4. Compare ASD and HC groups and generate violin plots.
%
% Inputs:
%   - Participant lists for ASD and HC.
%   - Individual cerebellar gradient CIFTI files.
%   - Group-average gradient CIFTI files.
%   - Cerebellar 7-network parcellation.
%
% Outputs:
%   - MAT files containing individual similarity and dispersion measures.
%   - Violin plots comparing ASD and HC.
%
% Notes:
%   - Similarity is calculated within each diagnostic group's own group-average map.
%   - Dispersion is quantified using median absolute deviation (MAD).
%   - Network IDs 1-6 are analyzed; network 7 (limbic) is excluded.

clear; clc;

%% 0. Configuration
dataset = 'ABIDE1';
base_dir = fullfile('..', dataset, 'maps', 'cerebellum');

list_dir = fullfile('.', 'sublist');
asd_list = readtable(fullfile(list_dir, ...
    'ABIDE1_ASD_flist.csv'), 'ReadVariableNames', false);
hc_list = readtable(fullfile(list_dir, ...
    'ABIDE1_HC_flist.csv'), 'ReadVariableNames', false);

groups = {'ASD', 'controls'};
participant_lists = {asd_list, hc_list};
group_labels = {'ASD', 'HC'};

cerebellum_rows = 70862:88714;
n_vertices = numel(cerebellum_rows);
gradient_index = 1;
network_ids = 1:6;
network_names = {'VN', 'SMN', 'DAN', 'VAN', 'FPN', 'DMN'};

% Set this path to the GRETNA installation on your own computer.
% addpath(genpath('PATH_TO_GRETNA'));

% Load the network parcellation (network IDs are assumed to be 1-7).
network_file = 'HCP_group_7Networks_100candidatevox.dscalar.nii';
network_cifti = ft_read_cifti(network_file);
network_labels = network_cifti.dscalar(cerebellum_rows, 1);

%% 1. Similarity: individual gradient vs. group-average gradient
% Calculate Pearson correlation within each group.

similarity = cell(1, 2);

for g = 1:2
    group_dir = fullfile(base_dir, groups{g});
    group_avg_dir = fullfile(group_dir, 'group_averaged');

    group_file = fullfile(group_avg_dir, 'Grad1-100.dtseries.nii');
    group_data = ft_read_cifti(group_file);
    group_gradient = group_data.dtseries(cerebellum_rows, gradient_index);

    participant_list = participant_lists{g};
    r_values = zeros(height(participant_list), 1);

    for i = 1:height(participant_list)
        subject_id = string(participant_list{i, 1});
        subject_dir = fullfile(group_dir, char(subject_id));
        individual_file = fullfile(subject_dir, ...
            'Corrected_Grad1-3.dtseries.nii');

        individual_data = ft_read_cifti(individual_file);
        individual_gradient = individual_data.dtseries( ...
            cerebellum_rows, gradient_index);

        r_values(i) = corr(individual_gradient, group_gradient, ...
            'Rows', 'complete');
    end

    similarity{g} = r_values;
end

asd_r_values = similarity{1};
hc_r_values = similarity{2};

save(fullfile(base_dir, 'asd_r_values.mat'), ...
    'asd_r_values');
save(fullfile(base_dir, 'hc_r_values.mat'), ...
    'hc_r_values');

%% 1.1. Plot similarity
[~, p_similarity] = ttest2(asd_r_values, hc_r_values);
fprintf('Similarity: p = %.4g\n', p_similarity);

plot_two_group_violin(asd_r_values, hc_r_values, ...
    'Similarity', [0.4, 1], 0.2e0, p_similarity);

%% 2. Global dispersion: MAD across cerebellar vertices
% Use the aligned individual gradients, as in the original analysis.

dispersion = cell(1, 2);

for g = 1:2
    group_dir = fullfile(base_dir, groups{g});
    participant_list = participant_lists{g};
    mad_values = zeros(height(participant_list), 1);

    for i = 1:height(participant_list)
        subject_id = string(participant_list{i, 1});
        subject_dir = fullfile(group_dir, char(subject_id));
        individual_file = fullfile(subject_dir, ...
            'Corrected_Grad1-3.dtseries.nii');

        individual_data = ft_read_cifti(individual_file);
        individual_gradient = individual_data.dtseries( ...
            cerebellum_rows, gradient_index);

        mad_values(i) = mad(individual_gradient, 0);
    end

    dispersion{g} = mad_values;
end

asd_distance = dispersion{1};
hc_distance = dispersion{2};

save(fullfile(base_dir, 'asd_distance.mat'), ...
    'asd_distance');
save(fullfile(base_dir, 'hc_distance.mat'), ...
    'hc_distance');

%% 2.1. Plot global dispersion
[~, p_dispersion] = ttest2(asd_distance, hc_distance);
fprintf('Global dispersion: p = %.4g\n', p_dispersion);

plot_two_group_violin(asd_distance, hc_distance, ...
    'Dispersion (MAD)', [1e-4, 1.6e-4], 0.2e-4, p_dispersion);

%% 3. Network-specific dispersion
% Calculate MAD separately for each network, excluding network ID 7.
% Do not use zero to identify non-network vertices, because zero can be
% a valid gradient value.

network_dispersion = cell(1, 2);

for g = 1:2
    group_dir = fullfile(base_dir, groups{g});
    participant_list = participant_lists{g};
    group_values = nan(height(participant_list), numel(network_ids));

    for i = 1:height(participant_list)
        subject_id = string(participant_list{i, 1});
        subject_dir = fullfile(group_dir, char(subject_id));
        individual_file = fullfile(subject_dir, ...
            'Corrected_Grad1-3.dtseries.nii');

        individual_data = ft_read_cifti(individual_file);
        individual_gradient = individual_data.dtseries( ...
            cerebellum_rows, gradient_index);

        for n = 1:numel(network_ids)
            in_network = network_labels == network_ids(n);
            values = individual_gradient(in_network);
            values = values(~isnan(values));

            if ~isempty(values)
                group_values(i, n) = mad(values, 0) * 10000;
            end
        end
    end

    network_dispersion{g} = group_values;
end

asd_7net_distance = network_dispersion{1};
hc_7net_distance = network_dispersion{2};

save(fullfile(base_dir, 'asd_7net_distance.mat'), ...
    'asd_7net_distance');
save(fullfile(base_dir, 'hc_7net_distance.mat'), ...
    'hc_7net_distance');

%% 3.1. Compare network-specific dispersion
p_network = nan(1, numel(network_ids));

for n = 1:numel(network_ids)
    [~, p_network(n)] = ttest2( ...
        asd_7net_distance(:, n), hc_7net_distance(:, n), ...
        'Vartype', 'unequal');
    fprintf('%s: p = %.4g\n', network_names{n}, p_network(n));
end

% GRETNA violin plot: two groups per network.
% Requires gretna_plot_violin to be available on the MATLAB path.
figure('Position', [100, 100, 1200, 600]);
gretna_plot_violin( ...
    {asd_7net_distance, hc_7net_distance}, ...
    group_labels, network_names, 'meanstdfill');

ax = gca;
set(ax, 'LineWidth', 2.5, 'FontName', 'Arial', 'FontSize', 20);
hold on;

color_asd = [226, 115, 134] / 255;
color_hc = [0.7, 0.7, 0.7];

% Color the violin patches according to their group position.
patches = findobj(gca, 'Type', 'patch');
for i = 1:numel(patches)
    center_x = mean(get(patches(i), 'XData'));
    group_position = round(center_x);

    if mod(group_position, 2) == 1
        set(patches(i), 'FaceColor', color_asd);
    else
        set(patches(i), 'FaceColor', color_hc);
    end
end

% Add significance annotations where p < 0.05.
for n = 1:numel(network_ids)
    if p_network(n) < 0.05
        asd_values = asd_7net_distance(:, n);
        hc_values = hc_7net_distance(:, n);
        max_value = max([asd_values; hc_values], [], 'omitnan');

        x_center = (n - 1) * 2 + 1.5;
        line_y = max_value * 1.15;
        text_y = max_value * 1.18;

        if p_network(n) < 0.001
            stars = '***';
        elseif p_network(n) < 0.01
            stars = '**';
        else
            stars = '*';
        end

        line([x_center - 0.8, x_center + 0.8], ...
            [line_y, line_y], 'Color', 'k', 'LineWidth', 2);
        text(x_center, text_y, stars, ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 24, 'FontWeight', 'bold', ...
            'Color', 'k', 'FontName', 'Arial');
    end
end

hold off;

%% Local function: plot two-group violin plot
function plot_two_group_violin(asd_values, hc_values, ...
    y_label, y_limits, y_tick_step, p_value)

    data = {asd_values, hc_values};
    color_asd = [226, 115, 134] / 255;
    color_hc = [0.7, 0.7, 0.7];

    figure('Position', [100, 100, 350, 500]);
    gretna_plot_violin(data, {'ASD', 'HC'}, {y_label}, 'meanstdfill');

    ax = gca;
    patches = findobj(ax, 'Type', 'patch');

    % Patch ordering can depend on the GRETNA/MATLAB version.
    % Inspect the plot if the colors appear swapped.
    if numel(patches) >= 2
        set(patches(1), 'FaceColor', color_hc);
        set(patches(2), 'FaceColor', color_asd);
    end

    lines = findobj(ax, 'Type', 'line');
    set(lines, 'Color', 'k');

    hold on;
    if p_value < 0.05
        max_value = max([asd_values(:); hc_values(:)], [], 'omitnan');
        line_y = max_value * 1.15;
        text_y = max_value * 1.17;

        if p_value < 0.001
            stars = '***';
        elseif p_value < 0.01
            stars = '**';
        else
            stars = '*';
        end

        line([1, 2], [line_y, line_y], ...
            'Color', 'k', 'LineWidth', 2);
        text(1.5, text_y, stars, ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 20, 'FontWeight', 'bold', ...
            'FontName', 'Arial');
    end

    ylim(y_limits);
    yticks(y_limits(1):y_tick_step:y_limits(2));
    set(ax, 'XTick', [1, 2], 'XTickLabel', {}, ...
        'LineWidth', 2.5, 'FontName', 'Arial', 'FontSize', 20);
    legend off;
    hold off;
end
