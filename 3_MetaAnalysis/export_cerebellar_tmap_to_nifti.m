%% Export cerebellar t-maps for NeuroSynth analysis
% Description:
%   1. Generate positive and negative binary CIFTI maps from a group-level
%      Gradient 1 t-map.
%   2. Extract cerebellar data from a CIFTI map and export it as a NIfTI
%      volume for subsequent NeuroSynth analysis.
%
% Input:
%   Grad1_fdr_t-map.dtseries.nii
%       Group-level Gradient 1 t-map.
%   cifti_base_info
%       CIFTI template containing spatial and brainordinate information.
%   CortexSubcortex_ColeAnticevic_NetPartition_woSubcorGSR_netassignments_LR.dscalar.nii
%       CIFTI reference file used for spatial coordinate mapping.
%   Atlas_ROIs.2.nii.gz
%       NIfTI reference volume defining the target image geometry.
%
% Output:
%   Grad1_fdr_t-map_positive.dtseries.nii
%       Binary map of positive t-values.
%   Grad1_fdr_t-map_negative.dtseries.nii
%       Binary map of negative t-values.
%   fdr_negative_Grad*.nii.gz
%       NIfTI volume containing the selected cerebellar gradient data.
%
% Notes:
%   - Update input paths and the gradient index before running.
%   - Requires FieldTrip CIFTI I/O functions and FreeSurfer MRIread/MRIwrite.
%   - The cerebellar brainordinate range and coordinate mapping are retained
%     from the original implementation.

%% Step 1: Generate positive and negative binary CIFTI maps
t_data = ft_read_cifti(char(fullfile(group_path, ...
    'Grad1_fdr_t-map.dtseries.nii')));

% Create binary masks based on the sign of the t-values
t_positive = double(t_data.dtseries > 0);
t_negative = double(t_data.dtseries < 0);

% Export positive t-map
tmp_base_info = cifti_base_info;
tmp_base_info.dtseries(:, 1) = t_positive(:, 1);

ft_write_cifti(char(fullfile(group_path, ...
    'Grad1_fdr_t-map_positive')), tmp_base_info, ...
    'parameter', 'dtseries');

% Export negative t-map
tmp_base_info = cifti_base_info;
tmp_base_info.dtseries(:, 1) = t_negative(:, 1);

ft_write_cifti(char(fullfile(group_path, ...
    'Grad1_fdr_t-map_negative')), tmp_base_info, ...
    'parameter', 'dtseries');

%% Step 2: Convert the cerebellar CIFTI map to NIfTI
% Directory containing the input CIFTI maps
dir_path = fullfile('..', 'results', 'ABIDE1', 'maps', 'cerebellum', ...
    'controls', 'group_averaged');

% Load the CIFTI reference and selected negative t-map
cifti_base_info = ft_read_cifti( ...
    'PATH_TO_CIFTI_REFERENCE.dscalar.nii');

grad_file = cifti_read(fullfile(dir_path, ...
    'Grad1_fdr_t-map_negative.dtseries.nii'));

% For the positive t-map, replace the filename above with:
% 'Grad1_fdr_t-map_positive.dtseries.nii'

% Load the target NIfTI reference volume
nifti_base_info = MRIread('PATH_TO_NIFTI_REFERENCE.nii.gz');

% Select cerebellar brainordinates
cerebellum_idx = 70862:88714;
cerebellum_pos = cifti_base_info.pos(cerebellum_idx, :);

% Map CIFTI coordinates to NIfTI voxel indices
x = 63 + cerebellum_pos(:, 2) / 2;
y = 45 - cerebellum_pos(:, 1) / 2;
z = 36 + cerebellum_pos(:, 3) / 2;

pos = sub2ind([109, 91, 91], x, y, z);

% Insert the selected gradient values into the NIfTI volume
data = zeros(109, 91, 91);
data(pos) = grad_file.cdata(cerebellum_idx, grad);

nifti_base_info.vol = data;

% Export the negative t-map for the selected gradient
output_file = fullfile(dir_path, ...
    ['fdr_negative_Grad', num2str(grad), '.nii.gz']);

MRIwrite(nifti_base_info, output_file);

% For the positive t-map, change the input and output filenames accordingly.
