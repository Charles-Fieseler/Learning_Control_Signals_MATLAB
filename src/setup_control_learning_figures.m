warning('This is a script and will modify global variables!')
disp('Make sure that you have set the variable dat_struct')

% First get a baseline model as a preprocessor
settings_base = settings_ideal;
settings_base.augment_data = 0;
settings_base.autocorrelation_noise_threshold = 0.5;
my_model_base = CElegansModel(dat_struct, settings_base);
n = my_model_base.dat_sz(1);
m = my_model_base.dat_sz(2);

% Loop through sparsity and rank to get a set of control signals
num_iter = 100;
max_rank = 10;
% max_rank = 15; % For the supplement
[all_U1, all_acf, all_nnz] = ...
    sparse_residual_analysis_max_over_iter(my_model_base,...
    num_iter, 1:max_rank, 'acf');
% Register the lines across rank
[registered_lines, registered_names] = ...
    sparse_residual_line_registration(all_U1, all_acf, my_model_base);

%---------------------------------------------
% Get names and points to cluster for ALL plots
%---------------------------------------------
% rank_to_plot = max_rank;
rank_to_plot = 10;
cluster_xy = zeros(length(registered_lines), 1);
final_xy = zeros(rank_to_plot,2);
final_names = cell(rank_to_plot,1);
for i = 1:length(registered_lines)
    xy = registered_lines{i};
    this_name = strjoin(registered_names{i}, ';');
    these_ind = xy{:,'which_rank'}==rank_to_plot;
    if any(these_ind) % Used for the next plot
        final_xy(xy{these_ind, 'which_line_in_rank'},:) = ...
            [i; xy{these_ind, 'this_y'}];
        final_names{xy{these_ind, 'which_line_in_rank'}} = this_name;
    end
    cluster_xy(i) = max(xy{:, 'this_y'});
end

num_clusters = 3;
[idx, centroids] = kmeans(cluster_xy, num_clusters, 'Replicates', 10);
[~, tmp] = max(cluster_xy);
real_clust_idx = idx(tmp);
[~, tmp] = min(cluster_xy);
noise_clust_idx = idx(tmp);

real_clust = find(idx==real_clust_idx);
gray_clust = find((idx~=real_clust_idx).*(idx~=noise_clust_idx));
noise_clust = find(idx==noise_clust_idx);

%---------------------------------------------
% Text names
%---------------------------------------------
n_lines = length(registered_lines);
max_plot_rank = 10;
text_xy = zeros(n_lines, 2);
text_names = cell(n_lines, 1);
for i = 1:length(registered_lines)
    xy = registered_lines{i};
    this_name = strjoin(registered_names{i}, ';');
    % Whether or not it shows up early enough
    if max_plot_rank < max_rank
        these_ranks = xy{:,'which_rank'};
        these_ranks_ind = (these_ranks<=max_plot_rank);
        if ~any(these_ranks_ind)
            text_names{i} = '';
            continue
        end
    else
        these_ranks_ind = true(size(xy{:,'which_rank'}));
    end
    % Whether or not to place a name by the line
    if size(xy,1) > 1 && ~ismember(i, noise_clust)
        text_xy(i, :) = [xy{1,'which_rank'}, xy{1,'this_y'}];
        text_names{i} = this_name;
    else
        text_xy(i, :) = [xy{1,'which_rank'}, xy{1,'this_y'}];
        text_names{i} = this_name;
%         text_names{i} = '';
    end
end

