% Authors: Guoyuan Yang, Xinyu Wu, and Wenjing Zhu
% Institution: Beijing Institute of Technology (BIT)
%
% =========================================================================
% Overview
% =========================================================================
% Purpose:
%   This script computes individual and group-averaged functional gradients
%   of the cerebellum based on cerebello-cortical functional connectivity.
%   It also performs site-effect harmonization of individual gradients
%   using ComBat.
%
% Main procedures:
%   1. Compute individual cerebello-cortical similarity matrices.
%   2. Derive individual functional gradients using BrainSpace.
%   3. Harmonize individual gradients across sites using ComBat,
%      adapted from the MDD_ConnectomeGradient pipeline.
%   4. Compute group-averaged cerebello-cortical similarity matrices.
%   5. Derive group-averaged functional gradients.
%
% Inputs:
%   - Preprocessed resting-state fMRI data in CIFTI format.
%   - Participant lists for the ASD and control groups.
%   - A cortical mask and CIFTI header information.
%   - Individual similarity matrices or aligned gradients, where required.
%   - Participant covariates, including site, diagnosis, age, and mean FD.
%   - MATLAB functions for preprocessing, similarity computation,
%     CIFTI I/O, BrainSpace, and ComBat.
%
% Outputs:
%   - Individual cerebello-cortical similarity matrices.
%   - Individual functional gradients and variance-explained values.
%   - ComBat-harmonized individual gradients.
%   - Group-averaged similarity matrices for the ASD and control groups.
%   - Group-averaged functional gradients and variance-explained values.
%
% Notes:
%   - Input data and output directories must be configured before execution.
%   - Subject-level data are expected to be obtained from authorized sources.
%   - The analysis assumes consistent subject ordering across participant
%     lists, covariate tables, and gradient data.
% =========================================================================


%% Shared configuration

project_dir = '.';
dataset = 'ABIDE1';

data_dir = fullfile('..', 'data', dataset);
results_dir = fullfile('..', 'results', dataset);
sublist_dir = fullfile(project_dir, 'sublist');
mask_file = fullfile(project_dir, 'masks', ...
    'fslr_downsample_900mesh_parcellation.dlabel.nii');
header_file = fullfile(project_dir, 'headers', ...
    'cifti_base_info.mat');

tools_dir = fullfile(project_dir, 'functions', 'tools');
similarity_dir = fullfile(project_dir, ...
    'functions', 'compute_similarity');
brainspace_dir = 'PATH_TO_BRAINSPACE_MATLAB';

category = ["ASD"; "controls"];
group_list_files = ["ASD_subject_list.csv"; ...
                    "HC_subject_list.csv"];

kernel = 'cs';
approach = 'le';
n_components = 100;

cifti_n_rows = 96854;
cerebellum_rows = 70862:88714;
n_cerebellar_vertices = numel(cerebellum_rows);


%% 1. Compute individual cerebello-cortical similarity matrices

addpath(tools_dir);
addpath(similarity_dir);

mask_data = ft_read_cifti(mask_file);
cortical_mask = mask_data.dlabel ~= 0;

for g = 1:numel(category)
    list = readtable(fullfile(sublist_dir, ...
        group_list_files(g)), 'ReadVariableNames', false);

    for i = 1:height(list)
        subject_id = char(list{i, 1});

        data_file = fullfile(data_dir, ...
            [subject_id, 'task-rest_DCANBOLDProc_v4.0.0_Atlas.dtseries.nii']);

        subject_dir = fullfile(results_dir, ...
            category(g), subject_id);

        if ~exist(subject_dir, 'dir')
            mkdir(subject_dir);
        end

        [x, x_sub] = preprocess_data_abide_cifti( ...
            data_file, subject_id, subject_dir, cortical_mask);

        c = compute_similarity_cifti(x, x_sub);

        save(fullfile(subject_dir, 'corr-matrix.mat'), ...
            'c', '-v7.3');
    end
end


%% 2. Compute individual functional gradients

addpath(genpath(brainspace_dir));
addpath(tools_dir);

load(header_file, 'cifti_base_info');
cifti_base_info.dtseries = zeros(cifti_n_rows, n_components);

gm = GradientMaps('kernel', kernel, ...
    'approach', approach, 'n_components', n_components);

for g = 1:numel(category)
    list = readtable(fullfile(sublist_dir, ...
        group_list_files(g)), 'ReadVariableNames', false);

    for i = 1:height(list)
        subject_id = char(list{i, 1});

        subject_dir = fullfile(results_dir, ...
            category(g), subject_id);

        S = load(fullfile(subject_dir, 'corr-matrix.mat'), 'c');
        result = gm.fit(S.c);

        gradient_data = cifti_base_info;
        gradient_data.dtseries(cerebellum_rows, :) = ...
            result.gradients{1};

        ft_write_cifti(fullfile(subject_dir, 'Grad1-100'), ...
            gradient_data, 'parameter', 'dtseries');

        save_file_spmd(fullfile(subject_dir, ...
            'Grad1-100_variexpl.mat'), ...
            result.lambda{1}, 'vari_expl');
    end
end


%% 3. Harmonize individual gradients using ComBat
% Reference:
%   Xia et al., MDD_ConnectomeGradient repository.
%   Original implementation:
%   https://github.com/mingruixia/MDD_ConnectomeGradient/blob/main/0_GradientCalculation/a_analysis_pipeline.m
%
% Adaptation:
%   The ComBat harmonization procedure was adapted from the
%   referenced pipeline for cerebellar functional gradients.
%   The implementation was modified to accommodate the present
%   study's data structure, covariates, and CIFTI outputs.

addpath(genpath(combat_dir));
addpath(fullfile(project_dir, 'functions', 'tools'));

base_dir_path = fullfile(old_results_dir, dataset, ...
    'maps', 'cerebellum');

% Participant lists and covariates
list1 = readtable(fullfile(sublist_dir, ...
    [dataset, '_ASD_flist.csv']), 'ReadVariableNames', false);
list2 = readtable(fullfile(sublist_dir, ...
    [dataset, '_HC_flist.csv']), 'ReadVariableNames', false);

asd = readtable(fullfile(sublist_dir, ...
    [dataset, '_ASD_finfos.csv']));
controls = readtable(fullfile(sublist_dir, ...
    [dataset, '_HC_finfos.csv']));
sub_info = [asd; controls];

site_data = sub_info.SITE;
group_data = sub_info.DX_GROUP;
age_data = sub_info.AGE_AT_SCAN;
fd_data = sub_info.fd_mean;

n_subjects = height(sub_info);
n_gradients = 3;
gradient_emb_reordered = cell(n_gradients, 1);

% Collect aligned individual gradients
for j = 1:n_gradients
    all_indiv = zeros(n_cerebellar_vertices, n_subjects);
    n = 1;

    for y = 1:2
        list = list1;
        if y == 2
            list = list2;
        end

        for i = 1:height(list)
            subject_id = char(list{i, 1});
            indiv_path = fullfile(base_dir_path, ...
                category(y), subject_id);

            grad_file = fullfile(indiv_path, ...
                'Aligned_Grad1-100.dtseries.nii');
            grad_data = ft_read_cifti(grad_file);

            all_indiv(:, n) = ...
                grad_data.dtseries(cerebellum_rows, j);
            n = n + 1;
        end
    end

    gradient_emb_reordered{j} = all_indiv;
end

% Harmonize gradients across sites
gradient_emb_correct = cell(n_gradients, 1);

for j = 1:n_gradients
    gradient_emb_correct{j} = combat( ...
        gradient_emb_reordered{j}, site_data', ...
        [group_data, age_data, fd_data], 'parametric');
end

save(fullfile(base_dir_path, 'Corrected_gradient1-3.mat'), ...
    'gradient_emb_correct');

% Write corrected gradients back to individual CIFTI files
load(header_file, 'cifti_base_info');
cifti_base_info.dtseries = zeros(cifti_n_rows, n_components);
cifti_base_info.time = linspace(0, n_components - 1, n_components);

n = 1;

for y = 1:2
    list = list1;
    if y == 2
        list = list2;
    end

    for i = 1:height(list)
        subject_id = char(list{i, 1});
        indiv_path = fullfile(base_dir_path, ...
            category(y), subject_id);

        tmp_base_info = cifti_base_info;

        for j = 1:n_gradients
            tmp_base_info.dtseries(cerebellum_rows, j) = ...
                gradient_emb_correct{j}(:, n);
        end

        ft_write_cifti(fullfile(indiv_path, ...
            'Corrected_Grad1-3'), tmp_base_info, ...
            'parameter', 'dtseries');

        n = n + 1;
    end
end


%% 4. Compute group-averaged similarity matrices

base_dir_path = fullfile(nodilated_results_dir, ...
    dataset, 'maps', 'cerebellum');

list1 = readtable(fullfile(sublist_dir, ...
    [dataset, '_ASD_flist.csv']), 'ReadVariableNames', false);
list2 = readtable(fullfile(sublist_dir, ...
    [dataset, '_HC_flist.csv']), 'ReadVariableNames', false);

for y = 1:2
    list = list1;
    if y == 2
        list = list2;
    end

    group_c_avg = zeros( ...
        n_cerebellar_vertices, n_cortical_features);

    for i = 1:height(list)
        subject_id = char(list{i, 1});
        matrix_file = fullfile(base_dir_path, ...
            category(y), subject_id, 'corr-matrix.mat');

        S = load(matrix_file, 'c');
        group_c_avg = group_c_avg + S.c / height(list);
    end

    group_path = fullfile(base_dir_path, ...
        category(y), 'group_averaged');

    if ~exist(group_path, 'dir')
        mkdir(group_path);
    end

    save(fullfile(group_path, 'corr-matrix.mat'), ...
        'group_c_avg', '-v7.3');
end


%% 5. Compute group-averaged functional gradients

addpath(genpath(brainspace_dir));
addpath(fullfile(project_dir, 'functions', 'tools'));

base_dir_path = fullfile(old_results_dir, ...
    dataset, 'maps', 'cerebellum');

load(header_file, 'cifti_base_info');
cifti_base_info.dtseries = zeros(cifti_n_rows, n_components);
cifti_base_info.time = linspace(0, n_components - 1, n_components);

gm = GradientMaps('kernel', kernel, ...
    'approach', approach, 'n_components', n_components);

for y = 1:2
    group_path = fullfile(base_dir_path, ...
        category(y), 'group_averaged');

    S = load(fullfile(group_path, 'corr-matrix.mat'), ...
        'group_c_avg');

    group = gm.fit(S.group_c_avg);

    scree_plot(group.lambda{1});

    tmp_base_info = cifti_base_info;
    tmp_base_info.dtseries(cerebellum_rows, :) = ...
        group.gradients{1};

    ft_write_cifti(fullfile(group_path, 'Grad1-100'), ...
        tmp_base_info, 'parameter', 'dtseries');

    save_file_spmd(fullfile(group_path, ...
        'Grad1-100_variexpl.mat'), ...
        group.lambda{1}, 'vari_expl');
end
