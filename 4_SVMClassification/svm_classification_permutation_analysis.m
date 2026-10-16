%% SVM classification and permutation analysis
% Description:
%   Part 1: Train and evaluate a linear SVM on ABIDE-I cerebellar gradient data.
%           Tune the regularization parameter C by 5-fold cross-validation,
%           then save accuracy, models, ROC coordinates, and Haufe weights.
%   Part 2: Run label-permutation analyses on ABIDE-II data using a fixed C,
%           then save null-model performance and weight results.
%
% Inputs:
%   ./headers/cifti_base_info.mat
%       CIFTI reference information.
%   ./sublist/old_sublist/
%       Subject lists and participant covariate tables for ABIDE-I and ABIDE-II.
%   ../old_preprocess_results/NC/ABIDE1/maps/cerebellum/
%       Individual ABIDE-I corrected Gradient 1 CIFTI files.
%   ../old_preprocess_results/NC/ABIDE2/maps/cerebellum/
%       Individual ABIDE-II corrected Gradient 1 CIFTI files.
%   LIBSVM and SurfStat MATLAB toolboxes; local helper functions in ./functions.
%
% Outputs:
%   ../old_preprocess_results/NC/ABIDE1/svmtrain/new_after_combat/
%       Repeated validation accuracy, trained models, Haufe weights, and ROC data.
%   ../old_preprocess_results/NC/ABIDE2/svmtrain/after_combat/regress_bestc/
%       Permutation-test accuracy, models, Haufe weights, and AUC values.
%
% Before running:
%   - Set LIBSVM_PATH and SURFSTAT_PATH below if needed.
%   - Ensure all data paths and helper functions are available.
%   - This script retains the original analysis design; see notes accompanying
%     the code for methodological issues that should be checked separately.
% Set these paths to your local installations if they are not already on the MATLAB path.

LIBSVM_PATH = '';   % e.g., '/path/to/libsvm-3.24/matlab'
SURFSTAT_PATH = ''; % e.g., '/path/to/surfstat'
if ~isempty(LIBSVM_PATH), addpath(LIBSVM_PATH); end
if ~isempty(SURFSTAT_PATH), addpath(SURFSTAT_PATH); end
addpath('./functions/tools');
addpath('./functions/roc');

load('./headers/cifti_base_info.mat');
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
    tmp_indiv_data = ft_read_cifti(cell2mat(fullfile('../results/ABIDE1/maps/cerebellum', category(y,:), list{i,1}, 'Corrected_Grad1-3.dtseries.nii')));

    indiv_data(:, x) = zscore(tmp_indiv_data.dtseries(70862:88714, 1));
    show_progress(x, indiv_num, frst); frst = 1;
    x = x + 1;
    end
end


% Regress age and mean framewise displacement (and sex for ABIDE-II)
asd = readtable('./sublist/ABIDE1_ASD_infos.csv');
controls = readtable('./sublist/ABIDE1_HC_infos.csv');
sub_info = [asd; controls];
group_data = sub_info.DX_GROUP; 
sex_data = sub_info.SEX;
age_data = sub_info.AGE_AT_SCAN;        
fd_data = sub_info.fd_mean;      
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
% best_c = 0.000977;

total_accuracy = zeros(svm_max_round, 1);
model_results = cell(svm_max_round, 1);
haufe_weight = zeros(svm_max_round, 17853);
TP_Array = cell(1, svm_max_round);
FP_Array = cell(1, svm_max_round);

sublist_value1 = string(list1{:, 1});
sublist_value2 = string(list2{:, 1});
sublist_value = [sublist_value1;sublist_value2];

for round = 1: svm_max_round
    % training data.
    ASD_ind = 1:size(ASD_list, 1);
    controls_ind = 1:size(controls_list, 1);

    ASD_trainind = randsample(size(ASD_list, 1), floor(0.8 * size(ASD_list, 1)));
    ASD_trainlist = ASD_list(ASD_trainind, :);

    controls_trainind = randsample(size(controls_list, 1), size(ASD_trainlist, 1));
    controls_trainlist = controls_list(controls_trainind, :);

    train_list = [ASD_trainlist; controls_trainlist];
    train_label = [-ones(size(ASD_trainlist, 1), 1); ones(size(controls_trainlist, 1), 1)];

    indiv_rank = randperm(size(train_list, 1));
    train_list = train_list(indiv_rank, :);
    train_label = train_label(indiv_rank, :);

    % Sample some of the rest data as validation data.
    ASD_validind = setdiff(ASD_ind', ASD_trainind);
    ASD_validlist = ASD_list(ASD_validind, :);

%     [~, controls_trainind] = ismember(controls_trainlist, controls_list.FILE_ID, 'rows');
    controls_validind = setdiff(controls_ind', controls_trainind);
    controls_validlist = controls_list(controls_validind, :);

    valid_list = [ASD_validlist; controls_validlist];
    valid_label = [-ones(size(ASD_validlist, 1), 1); ones(size(controls_validlist, 1), 1)];

    indiv_rank = randperm(size(valid_list, 1));
    valid_list = valid_list(indiv_rank, :);
    valid_label = valid_label(indiv_rank, :);

    % Training
    train_data = zeros(size(train_list, 1), 17853);
    
    for i = 1: size(train_list, 1)
        d1 = indiv_data(:, sublist_value == string(train_list{i, 1}));
        train_data(i, :) = d1;
    end

    % Select C using 5-fold cross-validation
    c_values = -10:0.2:10;
    best_c = ones(1, 1);
    best_acc = 0;

    for i = 1: length(c_values)
        fprintf('Training SVM with C = 2^%f\n',c_values(i));
        cg(i) = svmtrain(train_label, train_data, ['-t 0 -v 5 -c ', num2str(2^c_values(i))]);

        if(cg(i)>best_acc)
            best_acc = cg(i);
            best_c = 2^c_values(i);
        end
    end

    fprintf('Best C : %f, Best Accuracy : %f\n',best_c, best_acc);

    % Fit the final SVM model
    fprintf('Fitting SVM...\n');
    model = svmtrain(train_label, train_data, ['-t 0 -c ', num2str(best_c)]);

    % Prepare validation data
    valid_data = zeros(size(valid_list, 1), 17853);
    for i = 1: size(valid_list, 1)
        d2 = indiv_data(:, sublist_value == string(valid_list{i, 1}));
        valid_data(i, :) = d2;
    end

    fprintf('Predicting...\n');
    [predict_label, accuracy, test] = svmpredict(valid_label, valid_data, model);
    total_accuracy(round, 1) = accuracy(1);
    fprintf('the accuracy is %f\n', total_accuracy(round, 1));

    [~, ~, TP_Array{round}, FP_Array{round}] = AUC_TP_FP_Calculate(test, valid_label);
    model_results{round} = model;

    % Calculate normalized Haufe weights
    w = sum(model.sv_coef .* model.SVs);
    w = w * cov(train_data);
    w = w / sqrt(sum(w.^2));
    w = w / norm(w);
    haufe_weight(round, :) = w;
    abs_haufe_weight(round, :) = abs(w); % Using absolute value.

end

% Save results
base_dir_path = '../results/ABIDE1/svmtrain/after_combat';

mkdir(base_dir_path);

save(fullfile(base_dir_path, 'total_accuracy.mat'), 'total_accuracy', '-v7.3');
save(fullfile(base_dir_path, 'model_results.mat'), 'model_results', '-v7.3');
save(fullfile(base_dir_path, 'haufe_weight.mat'), 'haufe_weight', '-v7.3');
save(fullfile(base_dir_path, 'abs_haufe_weight.mat'), 'abs_haufe_weight', '-v7.3');
save(fullfile(base_dir_path, 'roc_curve.mat'), 'TP_Array', 'FP_Array', '-v7.3');

%% Part 2: ABIDE-II label-permutation analysis
% Note: The original implementation permutes training and validation labels
% separately. Confirm that this matches the intended permutation-test design
% before interpreting the resulting null distribution.
% LIBSVM_PATH and SURFSTAT_PATH are configured at the beginning of this script.

% relationship
load('./headers/cifti_base_info.mat');
list1 = readtable('./sublist/ABIDE2_ASD_list.csv', 'Delimiter',',','ReadVariableNames',false);
list2 = readtable('./sublist/ABIDE2_HC_list.csv', 'Delimiter',',','ReadVariableNames',false);

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
    tmp_indiv_data = ft_read_cifti(cell2mat(fullfile('../results/ABIDE2/maps/cerebellum', category(y,:), list{i,1}, 'fCorrected_Grad1-3.dtseries.nii')));

    indiv_data(:, x) = zscore(tmp_indiv_data.dtseries(70862:88714, 1));
    show_progress(x, indiv_num, frst); frst = 1;
    x = x + 1;
    end
end


% Regress age and mean framewise displacement (and sex for ABIDE-II)
asd = readtable('./sublist/ABIDE2_ASD_infos.csv');
controls = readtable('./sublist/ABIDE2_HC_infos.csv');
sub_info = [asd; controls];
group_data = sub_info.DX_GROUP; 
sex_data = sub_info.SEX;
age_data = sub_info.AGE_AT_SCAN;        
fd_data = sub_info.fd_mean;      

Covs = [age_data, sex_data/2+0.5, -sex_data/2+0.5, fd_data];

M = 1 + term(Covs); 
indiv_data = indiv_data';
slm = SurfStatLinMod(indiv_data, M);
indiv_data_residual = indiv_data - repmat(slm.coef(1, :), indiv_num, 1);
for i = 1: size(Covs, 2)
    indiv_data_residual = indiv_data_residual - repmat(Covs(:, i), 1, size(indiv_data, 2)) .* repmat(slm.coef(i+1, :), indiv_num, 1);
end

indiv_data = indiv_data_residual;
indiv_data = indiv_data';


best_c = 0.000977;
svm_max_round = 1000;

total_accuracy_null = zeros(svm_max_round, 1);
model_results_null = cell(svm_max_round, 1);
haufe_weight_null = zeros(svm_max_round, 17853);
AUC_Array = cell(1, svm_max_round);


sublist_value1 = string(list1{:, 1});
sublist_value2 = string(list2{:, 1});
sublist_value = [sublist_value1;sublist_value2];

for round = 1:svm_max_round 

    ASD_ind = 1:size(ASD_list, 1); 
    controls_ind = 1:size(controls_list, 1); 
 
    % training data
    ASD_trainind = randsample(size(ASD_list, 1), ...
        floor(0.8 * size(ASD_list, 1))); 
    
    ASD_trainlist = ASD_list(ASD_trainind, :); 
 
    controls_trainind = randsample(size(controls_list, 1), ...
        size(ASD_trainlist, 1)); 
    
    controls_trainlist = controls_list(controls_trainind, :); 
 
    train_list = [ASD_trainlist; controls_trainlist]; 

    % Permute original labels while preserving class numbers
    train_label = [ones(size(ASD_trainlist, 1), 1); ...
                   2 * ones(size(controls_trainlist, 1), 1)];
    
    train_label = train_label(randperm(length(train_label)));

    % Convert labels: ASD = -1, HC = 1
    train_label(train_label == 1) = -1; 
    train_label(train_label == 2) = 1; 
 
    indiv_rank = randperm(size(train_list, 1)); 
    train_list = train_list(indiv_rank, :); 
    train_label = train_label(indiv_rank, :); 
 
    % Prepare validation data data
    ASD_validind = setdiff(ASD_ind', ASD_trainind); 
    ASD_validlist = ASD_list(ASD_validind, :); 
 
    controls_validind = setdiff(controls_ind', controls_trainind); 
    controls_validlist = controls_list(controls_validind, :); 
 
    valid_list = [ASD_validlist; controls_validlist]; 

    % Permute original labels while preserving class numbers
    valid_label = [ones(size(ASD_validlist, 1), 1); ...
                   2 * ones(size(controls_validlist, 1), 1)];
    
    valid_label = valid_label(randperm(length(valid_label)));

    % Convert labels: ASD = -1, HC = 1
    valid_label(valid_label == 1) = -1; 
    valid_label(valid_label == 2) = 1; 
 
    indiv_rank = randperm(size(valid_list, 1)); 
    valid_list = valid_list(indiv_rank, :); 
    valid_label = valid_label(indiv_rank, :); 
 
    % Training
    train_data = zeros(size(train_list, 1), 17853); 
    
    for i = 1:size(train_list, 1) 
        d1 = indiv_data(:, ...
            sublist_value == string(train_list{i, 1})); 
        
        train_data(i, :) = d1; 
    end 
 
    % Fit the final SVM model 
    fprintf('Fitting SVM...\n'); 
    
    model = svmtrain(train_label, train_data, ...
        ['-t 0 -c ', num2str(best_c)]); 
 
    % Prepare validation data
    valid_data = zeros(size(valid_list, 1), 17853); 
    
    for i = 1:size(valid_list, 1) 
        d2 = indiv_data(:, ...
            sublist_value == string(valid_list{i, 1})); 
        
        valid_data(i, :) = d2; 
    end 
 
    fprintf('Predicting...\n'); 
    
    [predict_label, accuracy, test] = ...
        svmpredict(valid_label, valid_data, model); 
    
    total_accuracy_null(round, 1) = accuracy(1); 
 
    [AUC_Array{round}, ~, TP_Array{round}, FP_Array{round}] = ...
        AUC_TP_FP_Calculate(test, valid_label); 
    
    model_results_null{round} = model; 
 
    % Haufe normalized weight
    w = sum(model.sv_coef .* model.SVs); 
    w = w * cov(train_data); 
    w = w / sqrt(sum(w.^2)); 
    w = w / norm(w); 
    
    haufe_weight_null(round, :) = abs(w); 
    abs_haufe_weight_null(round, :) = abs(w); 

end 

% Save results
base_dir_path = '../results/ABIDE2/svmtrain/after_combat/regress_bestc';
mkdir(base_dir_path);

save(fullfile(base_dir_path, 'total_accuracy_null.mat'), 'total_accuracy_null', '-v7.3');
save(fullfile(base_dir_path, 'model_results_null.mat'), 'model_results_null', '-v7.3');
save(fullfile(base_dir_path, 'haufe_weight_null.mat'), 'haufe_weight_null', '-v7.3');
save(fullfile(base_dir_path, 'abs_haufe_weight_null.mat'), 'abs_haufe_weight_null', '-v7.3');
save(fullfile(base_dir_path, 'roc_curve_null.mat'), 'AUC_Array', '-v7.3');


