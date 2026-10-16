%% gene_pls_cerebellar_gradient.m
% =========================================================================
%  Gene association analysis of cerebellar cortical-gradient t-maps via PLS
% -------------------------------------------------------------------------
%  Description:
%    Step 1 - Project a surface-based (CIFTI) group-level gradient t-map
%              onto a cerebellar ROI atlas (Nettekoven 32 ROIs) by averaging
%              t-statistics within each ROI.
%    Step 2 - Fit Partial Least Squares (PLS) regression between ROI-wise
%              gene expression (predictors X) and the ROI-level gradient
%              effect (response Y); both X and Y are z-scored.
%    Step 3 - Permutation-test (n=1000) the variance explained by the first
%              1..10 PLS components to choose the dimensionality.
%    Step 4 - Bootstrap (n=1000) the PLS gene weights to obtain a stable,
%              signed Z-ranked gene list per component, suitable for downstream
%              enrichment tools (e.g. GOrilla).
%
%  Reference: Morgan et al. PNAS, 2019.
%  Adapted by Ziteng Han, 2024.
%  Adapted by Ruoqi Yang, 2025.
%  Adapted by Wenjing Zhu, 2026.
%
%  Dependencies (must be on the MATLAB path):
%    - Statistics and Machine Learning Toolbox  (plsregress, zscore, corr)
%    - FieldTrip                                (ft_read_cifti)
%    - CIFTIO / HCP Workbench MATLAB tools      (cifti_read)
%
%  Inputs (configured in the USER CONFIGURATION block below):
%    - cerebellar ROI atlas  (.dscalar.nii) : atlas_file
%    - group-level t-map      (.dtseries.nii): tmap_file
%    - gene expression table  (.csv)         : gene_csv_file
%                                              (1st col = ROI ID, rest = genes)
%
%  Outputs (written under save_dir):
%    - roi_tmap .mat             : ROI-averaged t-statistics
%    - PLS{1,2,3}_ROIscores.csv  : PLS component scores per ROI
%    - PLS{1,2,3}_geneWeights.csv: ranked genes (gene name, original column
%                                  index, bootstrap Z-score)
%    - Diagnostic figures: cumulative variance explained in Y, PLS1 vs
%      t-statistic scatter, permutation p-value vs number of components.
%
%  Usage:
%    Edit the paths and CIFTI row ranges in USER CONFIGURATION, then run.
% =========================================================================


%% USER CONFIGURATION -----------------------------------------------------
% Group / project layout.
base_dir_path = '/path/to/your/project/maps/cerebellum';   % project root
category      = ["ASD"; "controls"];                        % group labels
group_path    = fullfile(base_dir_path, category(2,:), 'group_averaged');

% Cerebellar ROI atlas. Default is the Nettekoven 32-ROI dilated atlas;
atlas_file = '/path/to/atl-NettekovenAsym32_space-HCPGrayord_dseg_std_dilate.2.dscalar.nii';

% Group-level gradient t-map on the HCP grayordinate surface.
tmap_file = fullfile(group_path, 'Grad1_t-map.dtseries.nii');

% Gene expression table: first column = ROI identifier, remaining columns =
% genes (rows align 1:1 with the ROI atlas above).
gene_csv_file = '/path/to/gene_expression_data_32_ori_new.csv';

% Output directory for bootstrap-ranked gene lists.
save_dir = fullfile(fileparts(gene_csv_file), 'ROI32', 'ori_tmap_new');

% CIFTI row ranges for the cerebellar vertices, indexed in the HCP
% grayordinate template used by both files. Adjust if you swap templates.
cereb_rows = 5878:23730;     % rows of atlas.cdata belonging to cerebellum
tmap_rows  = 70862:88714;    % rows of tmap.dtseries belonging to cerebellum
% -------------------------------------------------------------------------


%% Translate tmap to ROI level
roi  = cifti_read(atlas_file);
cere_roi = roi.cdata(cereb_rows, :);
roi_ids  = unique(cere_roi);
roi_ids(roi_ids == 0) = [];
num_rois = length(roi_ids);

tmap      = ft_read_cifti(char(tmap_file));
tmap_data = tmap.dtseries(tmap_rows, 1);

roi_tmap = zeros(num_rois, 1);

% extract mean t-statistic within each ROI
for i = 1:num_rois
    roi_id = roi_ids(i);

    % find all vertices for current ROI
    voxel_indices = find(cere_roi == roi_id);

    % average t-statistic across vertices of this ROI
    roi_voxel_data = tmap_data(voxel_indices, :);
    roi_tmap(i, :) = mean(roi_voxel_data, 1);

    fprintf('deal with ROI %d: include %d vertices\n', roi_id, length(voxel_indices));
end
roi_tmap_file = fullfile(group_path, 'ori_tmap_32roi.mat');
save(roi_tmap_file, 'roi_tmap');


%% Load parcellation, t-maps, gene
X_table = readtable(gene_csv_file, 'VariableNamingRule', 'preserve');

geneData   = X_table(:, 2:end);
geneMatrix = table2array(geneData);
columnNames = X_table.Properties.VariableNames;
geneNames   = columnNames(2:end);

tmap_struct = load(roi_tmap_file);
tmap_roi    = tmap_struct.roi_tmap;

X = zscore(geneMatrix);   % Predictor variables (genes x ROIs, z-scored)
Y = zscore(tmap_roi);     % Response variable (ROI gradient effect, z-scored)


%% Step1: perform full PLS and plot variance in Y explained by top 15 components
% Typically top 2 or 3 components explain a large part of the variance.
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X, Y);
dim = 15;

figure
plot(1:dim, cumsum(100*PCTVAR(2,1:dim)), '-o', ...
    'LineWidth', 1.5, 'Color', [140/255, 0, 0]);
set(gca, 'Fontsize', 14)
xlabel('Number of PLS components', 'FontSize', 14);
ylabel('Percent Variance Explained in Y', 'FontSize', 14);
grid on

% correlation of PLS component 1 with t-statistic
figure
plot(XS(:,1), tmap_roi, 'r.')
[R, p] = corr(XS(:,1), tmap_roi);
xlabel('XS scores for PLS component 1', 'FontSize', 14);
ylabel('t-statistic', 'FontSize', 14);
grid on


%% Step2: permutation testing to assess significance of PLS result as a function of
% the number of components (dim) included
rep = 1000;
for dim = 1:10
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X, Y, dim);
    temp      = cumsum(100*PCTVAR(2,1:dim));
    Rsquared  = temp(dim);

    for j = 1:rep
        order = randperm(size(Y,1));
        Yp    = Y(order, :);

        [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X, Yp, dim);

        temp   = cumsum(100*PCTVAR(2,1:dim));
        Rsq(j) = temp(dim);
    end

    dim
    R(dim) = Rsquared
    p(dim) = length(find(Rsq >= Rsquared)) / rep
end
figure
plot(1:dim, p, 'ok', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('Number of PLS components', 'FontSize', 14);
ylabel('p-value', 'FontSize', 14);
grid on


%% Step3: bootstrap to get the gene list
mkdir(save_dir);

genes      = geneNames;          % imported from gene CSV above
geneindex  = 1:length(genes);

% number of bootstrap iterations
bootnum = 1000;

% fit PLS with 3 components
dim = 3;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X, Y, dim);

% store region IDs and weights in descending order of weight per component
[R1, p1] = corr([XS(:,1), XS(:,2)], tmap_roi);

% align PLS components with the desired direction for interpretability
if R1(1,1) < 0    % flip sign if needed for interpretability
    stats.W(:,1) = -1 * stats.W(:,1);
    XS(:,1)      = -1 * XS(:,1);
end
if R1(2,1) < 0    % flip sign if needed for interpretability
    stats.W(:,2) = -1 * stats.W(:,2);
    XS(:,2)      = -1 * XS(:,2);
end

[PLS1w, x1] = sort(stats.W(:,1), 'descend');
PLS1ids     = genes(x1);
geneindex1  = geneindex(x1);
[PLS2w, x2] = sort(stats.W(:,2), 'descend');
PLS2ids     = genes(x2);
geneindex2  = geneindex(x2);
[PLS3w, x3] = sort(stats.W(:,3), 'descend');
PLS3ids     = genes(x3);
geneindex3  = geneindex(x3);

% write out component scores per ROI
csvwrite(fullfile(save_dir, 'PLS1_ROIscores.csv'), XS(:,1));
csvwrite(fullfile(save_dir, 'PLS2_ROIscores.csv'), XS(:,2));
csvwrite(fullfile(save_dir, 'PLS3_ROIscores.csv'), XS(:,3));

% containers for (ordered) weights from all bootstrap runs
PLS1weights = [];
PLS2weights = [];
PLS3weights = [];

% start bootstrap
for i = 1:bootnum
    i
    myresample = randsample(size(X,1), size(X,1), true);
    res(i,:)   = myresample;            % store resampling indices
    Xr         = X(myresample, :);      % resampled predictors
    Yr         = Y(myresample, :);      % resampled response

    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(Xr, Yr, dim);

    % --- PLS1 ---
    temp  = stats.W(:,1);               % extract PLS1 weights
    newW  = temp(x1);                   % order as in the initial PLS
    if corr(PLS1w, newW) < 0            % align sign across bootstrap runs
        newW = -1 * newW;
    end
    PLS1weights = [PLS1weights, newW];

    % --- PLS2 ---
    temp  = stats.W(:,2);
    newW  = temp(x2);
    if corr(PLS2w, newW) < 0
        newW = -1 * newW;
    end
    PLS2weights = [PLS2weights, newW];

    % --- PLS3 ---
    temp  = stats.W(:,3);
    newW  = temp(x3);
    if corr(PLS3w, newW) < 0
        newW = -1 * newW;
    end
    PLS3weights = [PLS3weights, newW];
end

% standard deviation of weights across bootstrap runs
PLS1sw = std(PLS1weights');
PLS2sw = std(PLS2weights');
PLS3sw = std(PLS3weights');

% bootstrap ratios (weight / bootstrap SE)
temp1 = PLS1w ./ PLS1sw';
temp2 = PLS2w ./ PLS2sw';
temp3 = PLS3w ./ PLS3sw';

% sort bootstrap Z and gene names
[Z1, ind1]  = sort(temp1, 'descend');
PLS1        = PLS1ids(ind1);
geneindex1  = geneindex1(ind1);
[Z2, ind2]  = sort(temp2, 'descend');
PLS2        = PLS2ids(ind2);
geneindex2  = geneindex2(ind2);
[Z3, ind3]  = sort(temp3, 'descend');
PLS3        = PLS3ids(ind3);
geneindex3  = geneindex3(ind3);

% write ranked gene lists (feed the first column of these CSVs into GOrilla)
fid1 = fopen(fullfile(save_dir, 'PLS1_geneWeights.csv'), 'w');
for i = 1:length(genes)
    fprintf(fid1, '%s, %d, %f\n', PLS1{i}, geneindex1(i), Z1(i));
end
fclose(fid1);

fid2 = fopen(fullfile(save_dir, 'PLS2_geneWeights.csv'), 'w');
for i = 1:length(genes)
    fprintf(fid2, '%s, %d, %f\n', PLS2{i}, geneindex2(i), Z2(i));
end
fclose(fid2);

fid3 = fopen(fullfile(save_dir, 'PLS3_geneWeights.csv'), 'w');
for i = 1:length(genes)
    fprintf(fid3, '%s, %d, %f\n', PLS3{i}, geneindex3(i), Z3(i));
end
fclose(fid3);
