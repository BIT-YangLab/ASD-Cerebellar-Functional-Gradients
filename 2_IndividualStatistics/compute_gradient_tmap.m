% =========================================================================
% Purpose:
%   Perform vertex-wise group comparisons of cerebellar Gradient 1 between
%   individuals with ASD and healthy controls, and generate t-maps before
%   and after FDR correction.
%
% Inputs:
%   - Participant lists for the ASD and control groups.
%   - Individual ComBat-corrected gradient CIFTI files.
%   - CIFTI header information.
%   - The FDR correction function.
%
% Outputs:
%   - Grad1_t-map.dtseries.nii: uncorrected vertex-wise t-statistics.
%   - Grad1_fdr_t-map.dtseries.nii: t-statistics retained after FDR
%     correction at q < 0.05; other vertices are set to zero.
%
% Notes:
%   - Positive t-values indicate higher Gradient 1 values in ASD than in
%     controls, assuming the group order used in the t-test below.
%   - FDR correction is applied across cerebellar vertices.
% =========================================================================

%% Configuration
dataset = 'ABIDE1';
base_dir = fullfile('..', 'results', ...
    dataset, 'maps', 'cerebellum');

sublist_dir = fullfile('.', 'sublist');
list_files = {
    fullfile(sublist_dir, 'ABIDE1_ASD_flist.csv')
    fullfile(sublist_dir, 'ABIDE1_HC_flist.csv')
};

groups = {'ASD', 'controls'};
gradient_index = 1;
cerebellum_rows = 70862:88714;
n_vertices = numel(cerebellum_rows);
n_cifti_rows = 96854;
n_components = 100;
fdr_threshold = 0.05;

%% Load CIFTI header information
load('./headers/cifti_base_info.mat', 'cifti_base_info');
cifti_base_info.dtseries = zeros(n_cifti_rows, n_components);
cifti_base_info.time = linspace(0, n_components - 1, n_components);

%% Load participant lists
asd_list = readtable(list_files{1}, 'ReadVariableNames', false);
hc_list = readtable(list_files{2}, 'ReadVariableNames', false);

%% Load individual gradients
asd_grad = zeros(n_vertices, height(asd_list));
hc_grad = zeros(n_vertices, height(hc_list));

for g = 1:2
    if g == 1
        participant_list = asd_list;
        gradient_data = asd_grad;
    else
        participant_list = hc_list;
        gradient_data = hc_grad;
    end

    for i = 1:height(participant_list)
        subject_id = char(participant_list{i, 1});
        subject_dir = fullfile(base_dir, groups{g}, subject_id);
        gradient_file = fullfile(subject_dir, ...
            'Corrected_Grad1-3.dtseries.nii');

        cifti_data = ft_read_cifti(gradient_file);
        gradient_data(:, i) = ...
            cifti_data.dtseries(cerebellum_rows, gradient_index);
    end

    if g == 1
        asd_grad = gradient_data;
    else
        hc_grad = gradient_data;
    end
end

%% Vertex-wise independent-samples t-tests
t_value = zeros(n_vertices, 1);
p_value = ones(n_vertices, 1);

for v = 1:n_vertices
    [~, p_value(v), ~, stats] = ttest2(asd_grad(v, :), hc_grad(v, :));
    t_value(v) = stats.tstat;
end

%% Save uncorrected t-map
group_dir = fullfile(base_dir, 'group_averaged');
if ~exist(group_dir, 'dir')
    mkdir(group_dir);
end

tmap_data = cifti_base_info;
tmap_data.dtseries(cerebellum_rows, 1) = t_value;

ft_write_cifti(fullfile(group_dir, 'Grad1_t-map'), ...
    tmap_data, 'parameter', 'dtseries');

%% FDR correction
addpath('./functions/tools');
[~, ~, q_values] = fdr(p_value);

significant = q_values < fdr_threshold;
fdr_t_value = t_value .* significant;

%% Save FDR-corrected t-map
fdr_map_data = cifti_base_info;
fdr_map_data.dtseries(cerebellum_rows, 1) = fdr_t_value;

ft_write_cifti(fullfile(group_dir, 'Grad1_fdr_t-map'), ...
    fdr_map_data, 'parameter', 'dtseries');
