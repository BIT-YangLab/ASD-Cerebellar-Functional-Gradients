%% NeuroSynth meta-analysis post-processing
% Description:
%   Post-process NeuroSynth decoding results by filtering terms using a
%   predefined cognitive atlas, calculating one-sided p-values, selecting
%   significant terms, and visualizing them with a word cloud.
%
% Input:
%   ABIDE1_fdr_neg_decoding_results.txt
%       NeuroSynth decoding results, including Feature and cluster_0.
%   Neurosynth_cognition_atlas_filter.mat
%       Predefined cognitive atlas terms (filter_terms_concepts).
%
% Output:
%   neg_results.xlsx
%       Filtered significant terms with z-scores and p-values.
%   Word cloud figure
%       Visualization of the filtered terms, weighted by their z-scores.
%
% Notes:
%   - Input file paths are specified relative to the working directory.
%   - This script requires MATLAB functions including readtable, intersect,
%     normcdf, writetable, and wordcloud.
%   - q_values must be defined by an appropriate multiple-comparison
%     correction procedure before being used in the results table.

%% Step 1: Load decoding results
dir_path = '.\neurosynth-master\examples';
decoding_data = readtable(fullfile(dir_path, ...
    'ABIDE1_fdr_neg_decoding_results.txt'));

decoding_data = table(decoding_data.Feature, decoding_data.cluster_0, ...
    'VariableNames', {'Feature', 'zscore'});

%% Step 2: Filter terms using a predefined cognitive atlas
load('Neurosynth_cognition_atlas_filter.mat');

% Ensure the filter terms are stored as a cell array
if ~iscell(filter_terms_concepts)
    filter_terms_concepts = cellstr(filter_terms_concepts);
end

% Convert terms to strings for comparison
decoding_str = string(decoding_data.Feature);
filter_terms_str = string(filter_terms_concepts);

% Identify terms shared by both lists
[common_terms, idx_decoding, idx_filter] = intersect( ...
    decoding_str, filter_terms_str, 'stable');

% Extract matching terms and their z-scores
filter_features = decoding_data.Feature(idx_decoding);
filter_z = decoding_data.zscore(idx_decoding);

filter_data = table(filter_features, filter_z, ...
    'VariableNames', {'Feature', 'zscore'});

%% Step 3: Sort and organize results
% Sort terms by z-score in descending order
[z_sorted, sort_idx] = sort(filter_data.zscore, 'descend');
sorted_features = filter_data.Feature(sort_idx);

meta_data = table(sorted_features, z_sorted, ...
    'VariableNames', {'Feature', 'zscore'});

%% Step 4: Calculate one-sided p-values
% Calculate p-values for positive z-scores
positive_idx = meta_data.zscore > 0;
positive_z = meta_data.zscore(positive_idx);
positive_features = meta_data.Feature(positive_idx);

positive_results = table(positive_features, positive_z, ...
    'VariableNames', {'Feature', 'zscore'});

p_value_positive = 1 - normcdf(positive_z);

% Assign p = 1 to non-positive z-scores
p_value_non_sig = ones(sum(~positive_idx), 1);

% Restore p-values to the original order of meta_data
all_p_values = ones(height(meta_data), 1);
all_p_values(positive_idx) = p_value_positive;

%% Step 5: Select significant terms
% Apply the significance threshold (p < 0.05)
significant_idx = all_p_values < 0.05;

significant_features = meta_data.Feature(significant_idx);
significant_z = meta_data.zscore(significant_idx);
p_value = all_p_values(significant_idx);

significant_results = table(significant_features, significant_z, p_value, ...
    'VariableNames', {'Feature', 'zscore', 'p_value'});

%% Step 6: Export significant results
final_results_neg = significant_results;

fprintf('Filtering results:\n');
fprintf('Number of significant terms: %d\n', height(significant_results));
fprintf('Number of predefined atlas terms: %d\n', numel(filter_terms_concepts));
fprintf('Number of retained terms: %d\n', height(final_results_neg));

writetable(final_results_neg, 'neg_results.xlsx');

%% Step 7: Prepare word cloud data
% Sort terms by z-score in descending order
[final_z_sorted, sort_idx] = sort(final_results_neg.zscore, 'descend');
final_features_sorted = final_results_neg.Feature(sort_idx);

wordcloud_data = table(final_features_sorted, final_z_sorted, ...
    'VariableNames', {'Word', 'Weight'});

%% Step 8: Generate word cloud
figure;

red_color = [208/255, 56/255, 78/255];
blue_color = [52/255, 139/255, 192/255];

wordcloud(wordcloud_data, 'Word', 'Weight', ...
    'Shape', 'oval', 'Color', blue_color, ...
    'HighlightColor', blue_color);
