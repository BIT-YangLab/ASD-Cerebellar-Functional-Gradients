% =========================================================================
% Purpose:
%   Plot cerebellar functional gradient distributions and selected network
%   centroids for ASD and control groups.
%
% Inputs:
%   - Group-averaged gradient CIFTI files.
%   - Cerebellar network parcellation CIFTI file.
%   - Network color table (if network-specific colors are used).
%
% Outputs:
%   - Scatter plots of Gradient 2 versus Gradient 1.
%   - Selected network centroids and between-group centroid distances.
% =========================================================================

%% Configuration
dataset = 'ABIDE1';

base_dir = fullfile('..', 'results',  ...
    dataset, 'maps', 'cerebellum');

parcellation_file = 'HCP_group_7Networks_100candidatevox.dscalar.nii';

group_names = {'controls', 'ASD'};
group_dirs = {
    fullfile(base_dir, 'controls', 'A_old_results', 'group_averaged')
    fullfile(base_dir, 'ASD', 'A_old_results', 'group_averaged')
};

% Cerebellar vertex indices in the full CIFTI data
start_pos = 70862;
end_pos = 88714;

% Networks whose centroids are displayed
nets_to_show = [2, 3, 4];

% Coordinate limits: set these for the selected dataset
x_limits = [-4.5e-4, 3.5e-4];
y_limits = [-3e-4, 4e-4];

%% Load network labels
parcellation = ft_read_cifti(parcellation_file);
network_labels = parcellation.dscalar(start_pos:end_pos, 1);

%% Load group gradients and calculate network centroids
n_groups = numel(group_names);
gradients = cell(n_groups, 1);
centroids = cell(n_groups, 1);

for g = 1:n_groups
    grad_file = cifti_read(fullfile( ...
        group_dirs{g}, 'Grad1-100.dtseries.nii'));

    grad1 = grad_file.cdata(start_pos:end_pos, 1);
    grad2 = grad_file.cdata(start_pos:end_pos, 2);

    % Columns: Gradient 2 (x-axis), Gradient 1 (y-axis)
    gradients{g} = [grad2, grad1];

    centroids{g} = nan(7, 2);

    for net = 1:7
        idx = network_labels == net;

        if any(idx)
            centroids{g}(net, :) = mean( ...
                gradients{g}(idx, :), 1, 'omitnan');
        end
    end
end

%% Plot group scatter plots
figure('Position', [0, 0, 800, 350]);

for g = 1:n_groups
    subplot(1, 2, g);

    scatter(gradients{g}(:, 1), gradients{g}(:, 2), ...
        30, [0.8, 0.8, 0.8], '.');

    hold on;

    for net = nets_to_show
        xy = centroids{g}(net, :);

        if all(isfinite(xy))
            if g == 1
                marker_color = 'w';
            else
                marker_color = 'k';
            end

            scatter(xy(1), xy(2), 40, marker_color, 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        end
    end

    xlim(x_limits);
    ylim(y_limits);
    grid off;

    ax = gca;
    ax.LineWidth = 2;
    ax.FontSize = 14;
    ax.TickDir = 'out';

    title(group_names{g});

    if g == 2
        set(ax, 'YTick', [], 'YColor', 'none');
    end

    hold off;
end

%% Calculate centroid distances between groups
fprintf('\nCentroid distances between ASD and Controls:\n');

for net = nets_to_show
    control_xy = centroids{1}(net, :);
    asd_xy = centroids{2}(net, :);

    if all(isfinite(control_xy)) && all(isfinite(asd_xy))
        distance = norm(asd_xy - control_xy) * 10000;
        fprintf('Network %d: distance = %.4f\n', net, distance);
    end
end