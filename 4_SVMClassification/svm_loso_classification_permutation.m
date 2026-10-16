%% Leave-one-site-out SVM analysis and permutation testing
% Description:
%   Perform leave-one-site-out (LOSO) SVM classification using cerebellar
%   Gradient 1 data, then run label-permutation analyses to estimate the
%   null distribution of classification accuracy and ROC-related metrics.
%
% Input:
%   ./sublist/old_sublist/ABIDE1_ASD_flist.csv
%   ./sublist/old_sublist/ABIDE1_HC_flist.csv
%       Subject lists for ASD and healthy controls.
%   ./sublist/old_sublist/ABIDE1_ASD_finfos.csv
%   ./sublist/old_sublist/ABIDE1_HC_infos.csv
%       Subject metadata, including site, diagnosis, age, and mean FD.
%   ../old_preprocess_results/NC/ABIDE1/maps/cerebellum/
%       Individual CIFTI files containing corrected cerebellar gradients.
%
% Output:
%   ../old_preprocess_results/NC/ABIDE1/svmtrain/leave_one_site_out/
%       total_accuracy.mat, model_results.mat, haufe_weight.mat,
%       abs_haufe_weight.mat, roc_curve.mat, total_accuracy_null.mat,
%       and roc_curve_null.mat.
%
% Dependencies:
%   LIBSVM for MATLAB, SurfStat, FieldTrip CIFTI I/O, and the project
%   functions in ./functions/tools and ./functions/roc.
%
% Notes:
%   - Configure the software paths below for your own environment.
%   - This script retains the original analysis workflow; see the README
%     and accompanying documentation for methodological considerations.
%   - The covariate regression is performed before site-wise splitting,
%     matching the original implementation.
%
% Configure these paths for your local installation.
libsvm_path = 'PATH_TO_LIBSVM_MATLAB';
surfstat_path = 'PATH_TO_SURFSTAT';

addpath(libsvm_path);
addpath(surfstat_path);
addpath('./functions/tools');
addpath('./functions/roc');

list1 = readtable('./sublist/ABIDE1_ASD_list.csv', 'Delimiter',',','ReadVariableNames',false);
list2 = readtable('./sublist/ABIDE1_HC_list.csv', 'Delimiter',',','ReadVariableNames',false);

ASD_list = list1;
controls_list = list2;

category=["ASD";"controls"];

indiv_num = size(ASD_list, 1) + size(controls_list,1);
indiv_data = zeros(17853, indiv_num);

fprintf('Loading data...\n');
frst = 0;

x = 1;
for y = 1:2
    if y == 1
            list = list1;
        else
            list = list2;
    end
    for i = 1:size(list, 1)
    tmp_indiv_data = ft_read_cifti(cell2mat(fullfile('../results/ABIDE1/maps/cerebellum', category(y,:), list{i,1}, 'Corrected_Grad1-3.dtseries.nii')));

    indiv_data(:, x) = zscore(tmp_indiv_data.dtseries(70862:88714, 1));
%     indiv_data(:, x) = tmp_indiv_data.dtseries(70862:88714, 1);
    show_progress(x, indiv_num, frst); frst = 1;
    x = x + 1;
    end
end

% Regressing out movement
asd = readtable('./sublist/ABIDE1_ASD_infos.csv');
controls = readtable('./sublist/ABIDE1_HC_infos.csv');
sub_info = [asd; controls];

site_data = sub_info.SITE; 

group_data = sub_info.DX_GROUP; 
age_data = sub_info.AGE_AT_SCAN;        
fd_data = sub_info.fd_mean;      

% Covs = [age_data, sex_data/2+0.5, -sex_data/2+0.5, fd_data];%ABIDE2
Covs = [age_data, fd_data];%ABIDE1

M = 1 + term(Covs); 
indiv_data = indiv_data';
slm = SurfStatLinMod(indiv_data, M);
indiv_data_residual = indiv_data - repmat(slm.coef(1, :), indiv_num, 1);
for i = 1: size(Covs, 2)
    indiv_data_residual = indiv_data_residual - repmat(Covs(:, i), 1, size(indiv_data, 2)) .* repmat(slm.coef(i+1, :), indiv_num, 1);
end

indiv_data = indiv_data_residual;
indiv_data = indiv_data';


svm_max_round = 100;

sites = unique(site_data);
num_sites = length(sites);

total_accuracy = zeros(num_sites, svm_max_round);
model_results = cell(num_sites, svm_max_round);
haufe_weight = zeros(num_sites, svm_max_round, 17853);
abs_haufe_weight = zeros(num_sites, svm_max_round, 17853);
TP_Array = cell(num_sites, svm_max_round);
FP_Array = cell(num_sites, svm_max_round);

for site = 1:num_sites

    test_site = sites(site);

    fprintf('\nTesting site: %d\n', test_site);

    train_ind = find(site_data ~= test_site);
    valid_ind = find(site_data == test_site);

    train_group = group_data(train_ind);
    valid_group = group_data(valid_ind);

    for round = 1:svm_max_round

        % Balance training data
        ASD_train_all = train_ind(train_group == 1);
        controls_train_all = train_ind(train_group == 2);

        n_train = min(length(ASD_train_all), length(controls_train_all));

        ASD_trainind = randsample(ASD_train_all, n_train);
        controls_trainind = randsample(controls_train_all, n_train);

        train_ind_round = [ASD_trainind; controls_trainind];

        indiv_rank = randperm(length(train_ind_round));
        train_ind_round = train_ind_round(indiv_rank);

        train_label = group_data(train_ind_round);
        train_label(train_label == 1) = -1;
        train_label(train_label == 2) = 1;

        % Training
        train_data = zeros(length(train_ind_round), 17853);

        for i = 1:length(train_ind_round)
            train_data(i, :) = indiv_data(:, train_ind_round(i));
        end

        % finding best c
        c_values = -10:2:10;
        best_c = 1;
        best_acc = 0;

        for i = 1:length(c_values)

            fprintf('Training SVM with C = 2^%f\n', c_values(i));

            cg = svmtrain(train_label, train_data, ...
                ['-t 0 -v 5 -c ', num2str(2^c_values(i))]);

            if cg > best_acc
                best_acc = cg;
                best_c = 2^c_values(i);
            end

        end

        fprintf('Best C : %f, Best Accuracy : %f\n', best_c, best_acc);

        % Fit SVM
        fprintf('Fitting SVM...\n');

        model = svmtrain(train_label, train_data, ...
            ['-t 0 -c ', num2str(best_c)]);

        % Validation
        valid_data = zeros(length(valid_ind), 17853);

        for i = 1:length(valid_ind)
            valid_data(i, :) = indiv_data(:, valid_ind(i));
        end

        valid_label = group_data(valid_ind);
        valid_label(valid_label == 1) = -1;
        valid_label(valid_label == 2) = 1;

        fprintf('Predicting...\n');

        [predict_label, accuracy, test] = ...
            svmpredict(valid_label, valid_data, model);

        total_accuracy(site, round) = accuracy(1);

        fprintf('Site %d, Round %d, accuracy = %f\n', ...
            test_site, round, total_accuracy(site, round));

        [~, ~, TP_Array{site, round}, FP_Array{site, round}] = ...
            AUC_TP_FP_Calculate(test, valid_label);

        model_results{site, round} = model;

        % Haufe normalized weight
        w = sum(model.sv_coef .* model.SVs);
        w = w * cov(train_data);
        w = w / sqrt(sum(w.^2));
        w = w / norm(w);

        haufe_weight(site, round, :) = w;
        abs_haufe_weight(site, round, :) = abs(w);

    end
end


% Saving results
base_dir_path = fullfile('..', 'results', 'ABIDE1', ...
    'svmtrain', 'leave_one_site_out');

mkdir(base_dir_path);

save(fullfile(base_dir_path, 'total_accuracy.mat'), 'total_accuracy', '-v7.3');
save(fullfile(base_dir_path, 'model_results.mat'), 'model_results', '-v7.3');
save(fullfile(base_dir_path, 'haufe_weight.mat'), 'haufe_weight', '-v7.3');
save(fullfile(base_dir_path, 'abs_haufe_weight.mat'), 'abs_haufe_weight', '-v7.3');
save(fullfile(base_dir_path, 'roc_curve.mat'), 'TP_Array', 'FP_Array', '-v7.3');

%% Step 2: Run LOSO label-permutation tests
% Reuse the software paths configured at the beginning of this script.
% Prepare input data for the permutation analysis.
list1 = readtable('./sublist/ABIDE1_ASD_flist.csv', 'Delimiter',',','ReadVariableNames',false);
list2 = readtable('./sublist/ABIDE1_HC_flist.csv', 'Delimiter',',','ReadVariableNames',false);

ASD_list = list1;
controls_list = list2;

category=["ASD";"controls"];

indiv_num = size(ASD_list, 1) + size(controls_list,1);
indiv_data = zeros(17853, indiv_num);

fprintf('Loading data...\n');
frst = 0;

x = 1;
for y = 1:2
    if y == 1
        list = list1;
    else
        list = list2;
    end

    for i = 1:size(list, 1)

        tmp_indiv_data = ft_read_cifti(cell2mat(fullfile( ...
            '../results/ABIDE1/maps/cerebellum', ...
            category(y,:), list{i,1}, ...
            'Corrected_Grad1-3.dtseries.nii')));

        indiv_data(:, x) = zscore(tmp_indiv_data.dtseries(70862:88714, 1));

        show_progress(x, indiv_num, frst);
        frst = 1;
        x = x + 1;

    end
end


% Regressing out movement
asd = readtable('./sublist/ABIDE1_ASD_finfos.csv');
controls = readtable('./sublist/ABIDE1_HC_infos.csv');
sub_info = [asd; controls];

site_data = sub_info.SITE;
group_data = sub_info.DX_GROUP;
age_data = sub_info.AGE_AT_SCAN;
fd_data = sub_info.fd_mean;

Covs = [age_data, fd_data];

M = 1 + term(Covs);
indiv_data = indiv_data';
slm = SurfStatLinMod(indiv_data, M);

indiv_data_residual = indiv_data - repmat(slm.coef(1, :), indiv_num, 1);

for i = 1:size(Covs, 2)
    indiv_data_residual = indiv_data_residual ...
        - repmat(Covs(:, i), 1, size(indiv_data, 2)) ...
        .* repmat(slm.coef(i+1, :), indiv_num, 1);
end

indiv_data = indiv_data_residual;
indiv_data = indiv_data';


% Running LOSO permutations

site_data = sub_info.SITE;
group_data = sub_info.DX_GROUP;

sites = unique(site_data);
num_sites = length(sites);

best_c = 0.000977;
svm_max_round = 1000;

total_accuracy_null = zeros(num_sites, svm_max_round);
AUC_Array_null = cell(num_sites, svm_max_round);
TP_Array_null = cell(num_sites, svm_max_round);
FP_Array_null = cell(num_sites, svm_max_round);

for site = 1:num_sites

    test_site = sites(site);

    train_ind = find(site_data ~= test_site);
    valid_ind = find(site_data == test_site);

    train_data_all = indiv_data(:, train_ind)';
    valid_data_all = indiv_data(:, valid_ind)';

    train_group = group_data(train_ind);
    valid_group = group_data(valid_ind);

    for round = 1:svm_max_round

        % Randomly permute labels while preserving class numbers
        train_label = train_group(randperm(length(train_group)));
        valid_label = valid_group(randperm(length(valid_group)));

        % Convert labels: ASD = -1, HC = 1
        train_label(train_label == 1) = -1;
        train_label(train_label == 2) = 1;

        valid_label(valid_label == 1) = -1;
        valid_label(valid_label == 2) = 1;

        % Shuffle subjects
        train_rank = randperm(length(train_label));
        train_data = train_data_all(train_rank, :);
        train_label = train_label(train_rank);

        valid_rank = randperm(length(valid_label));
        valid_data = valid_data_all(valid_rank, :);
        valid_label = valid_label(valid_rank);

        % Fit SVM
        model = svmtrain(train_label, train_data, ...
            ['-t 0 -c ', num2str(best_c)]);

        % Validation
        [predict_label, accuracy, test] = svmpredict( ...
            valid_label, valid_data, model);

        total_accuracy_null(site, round) = accuracy(1);

        [AUC_Array_null{site, round}, ~, ...
            TP_Array_null{site, round}, FP_Array_null{site, round}] = ...
            AUC_TP_FP_Calculate(test, valid_label);

    end

    fprintf('Site %d/%d finished.\n', site, num_sites);

end

% Saving results

base_dir_path = fullfile('..', 'results',  ...
    'ABIDE1', 'svmtrain', 'leave_one_site_out');

if ~exist(base_dir_path, 'dir')
    mkdir(base_dir_path);
end

save(fullfile(base_dir_path, 'total_accuracy_null.mat'), ...
    'total_accuracy_null', '-v7.3');

save(fullfile(base_dir_path, 'roc_curve_null.mat'), ...
    'AUC_Array_null', 'TP_Array_null', 'FP_Array_null', '-v7.3');