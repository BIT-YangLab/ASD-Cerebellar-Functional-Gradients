function [x, x_sub] = preprocess_data_abide_cifti(file_dir_path, testee_ID, tmp_path, fsLR_900mesh_mask_dlabel)
% =========================================================================
% Purpose:
%   Preprocess CIFTI time series by spatially smoothing, extracting cortical
%   and cerebellar data, and standardizing each vertex's time series.
%
% Inputs:
%   file_dir_path                 - Input CIFTI dtseries file.
%   testee_ID                     - Subject ID (retained for compatibility).
%   tmp_path                      - Directory for temporary smoothed files.
%   fsLR_900mesh_mask_dlabel      - Cortical mask.
%
% Outputs:
%   x                             - Standardized cortical time series
%                                   (time points x cortical vertices).
%   x_sub                         - Standardized cerebellar time series
%                                   (time points x cerebellar vertices).
% =========================================================================

left_surface_file = './surface/Conte69.L.very_inflated.32k_fs_LR.surf.gii';
right_surface_file = './surface/Conte69.R.very_inflated.32k_fs_LR.surf.gii';

data_file_path = char(file_dir_path);
[~, file_name, file_ext] = fileparts(data_file_path);
smoothed_file = fullfile(tmp_path, ['smooth_', file_name, file_ext]);

% Spatial smoothing (6 mm FWHM)
system(['wb_command -cifti-smoothing ', data_file_path, ...
    ' 2.547965 2.547965 COLUMN ', smoothed_file, ...
    ' -left-surface ', left_surface_file, ...
    ' -right-surface ', right_surface_file]);

% Load smoothed CIFTI data
data_struct = ft_read_cifti(smoothed_file);
data = data_struct.x0;

% Extract cortical and cerebellar time series
x = data(1:64984, :);
x = x(fsLR_900mesh_mask_dlabel ~= 0, :)';

x_sub = data(70862:88714, :)';

% Demean and standardize each vertex's time series
x = detrend(x, 'constant');
x = x ./ std(x, 0, 1);

x_sub = detrend(x_sub, 'constant');
x_sub = x_sub ./ std(x_sub, 0, 1);

% Replace NaN values with zero
x(isnan(x)) = 0;
x_sub(isnan(x_sub)) = 0;

end