function [co_occurrence, these_names, my_model_td, all_dat, out] = ...
    zimmer_dynamic_clustering(...
    filename, my_features, calc_final_clusters, use_PCA, use_findpeaks, to_plot)
%% Set defaults
if ~exist('my_features', 'var')
    my_features = {'Correlation','L2_distance','FN','FP','Spikes'};
end
if ~exist('calc_final_clusters', 'var')
    calc_final_clusters = false;
end
if ~exist('use_PCA', 'var')
    use_PCA = false;
end
if ~exist('use_findpeaks', 'var')
    use_findpeaks = true;
end
if ~exist('to_plot', 'var')
    to_plot = false;
end
%% Build the model
settings_ideal = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'augment_data', 7,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','func_DMDc',...
    'global_signal_subset', {{'DT','VT','REV','FWD','SLOW'}},...
    'autocorrelation_noise_threshold', 0.5,...
    'lambda_sparse',0);
settings_ideal.global_signal_mode = 'ID_binary_transitions';

if contains(filename, 'PreLet')
    settings_ideal.global_signal_subset = {'turn', 'Rev', 'For', 'Q'};
end

% Build model and get data
my_model_td = CElegansModel(filename, settings_ideal);
%% Analyze all neurons
num_neurons = my_model_td.original_sz(1);
reconstruct_dat = real(...
    my_model_td.AdaptiveDmdc_obj.calc_reconstruction_control());
if use_findpeaks
    f = @(x) abs(TVRegDiff(x, 5, 1e-4, [], [], [], [], false, false));
else
    f = @(x) my_model_td.flat_filter(x, 10);
end
threshold_vec = linspace(0.1, 1.5, 5);

all_fp = zeros(num_neurons, 1);
all_fn = all_fp;
true_pos = all_fp;
all_ind_fp = false(my_model_td.original_sz);
all_ind_fn = all_ind_fp;
all_spikes = all_fp;
used_thresholds = all_fp;

for i = 1:num_neurons
    this_dat = my_model_td.dat(i,:);
    this_recon = reconstruct_dat(i,:);
    % Analyze the derivatives
    % Get false detections
    if use_findpeaks
        this_dat = f(this_dat);
        this_recon = f(this_recon);
        used_thresholds(i) = 1;
    else
        all_thresholds = zeros(length(threshold_vec),1);
        this_dat = abs(gradient(f(this_dat)));
        this_recon = abs(gradient(f(this_recon)));
        f_detect = @(x) minimize_false_detection(this_dat, ...
            this_recon, x, 0.5, true);
        all_vals = all_thresholds;
        for i2 = 1:length(threshold_vec)
            [all_thresholds(i2), all_vals(i2)] = ...
                fminsearch(f_detect, threshold_vec(i2));
        end
        [~, ind] = min(all_vals);
        used_thresholds(i) = all_thresholds(ind);
    end
    
    % Use partial detection of false positives/negatives
    [all_fp(i), all_fn(i), all_spikes(i),...
        tmp_fp, tmp_fn, true_pos(i)] = ...
        calc_false_detection(this_dat, this_recon, used_thresholds(i),...
        [], [], false, true);
    all_ind_fp(i,tmp_fp) = true;
    all_ind_fn(i,tmp_fn) = true;
end

% Diagnostics regarding metrics
all_corr = my_model_td.calc_correlation_matrix([], 'linear');
all_corr = all_corr(1:num_neurons);
ind_to_keep = ~(all_spikes==0);

names = my_model_td.get_names(1:num_neurons, true);
% names{1} = '1';
% names{49} = '49';

this_dat = real(my_model_td.dat - mean(my_model_td.dat,2));
[snr_vec, dat_signal, dat_noise] = calc_snr(this_dat);
max_variance = var(dat_signal-mean(dat_signal,2), 0, 2) ./...
    var(this_dat, 0, 2);
max_variance = max_variance(1:num_neurons);
snr_vec = snr_vec(1:num_neurons);
this_recon = real(reconstruct_dat - mean(reconstruct_dat,2));
% all_dist = vecnorm(this_dat - this_recon, 2, 2);
L2_dist = vecnorm(dat_signal - this_recon, 2, 2);
L2_dist = L2_dist(1:num_neurons)./...
    vecnorm(my_model_td.dat(1:num_neurons), 2, 2);

disp('Finished analyzing all neurons')
%% Build the feature set
raw_corr = my_model_td.calc_correlation_matrix();
raw_corr = raw_corr(1:num_neurons);
rng(2)
if any(ismember(my_features, 'dtw'))
    % Dynamic Time Warping distance
    n = num_neurons;
    dtw_dist = zeros(n,1);
    for i = 1:n
        dtw_dist(i) = dtw(dat_signal(i,:), this_recon(i2,:));
    end
else
    dtw_dist = zeros(size(L2_dist));
end
all_dat = table(real(raw_corr), real(all_corr), L2_dist, dtw_dist,...
    all_fn, all_fp, true_pos, all_spikes,...
    all_fn./all_spikes, all_fp./all_spikes, true_pos./all_spikes,...
    snr_vec, max_variance,...
    'VariableNames', {...
    'Raw_Correlation', 'Correlation', 'L2_distance', 'dtw',...
    'FN', 'FP', 'TP', 'Spikes',...
    'FN_norm', 'FP_norm', 'TP_norm',...
    'SNR', 'max_variance'});
to_cluster_dat = all_dat{:, my_features};
neurons_with_activity = logical(ind_to_keep.*(snr_vec>median(snr_vec)));
to_cluster_dat = to_cluster_dat(neurons_with_activity,:);

% figure; plot(to_cluster_dat(1,:)); xticklabels(my_features);xtickangle(60)
% Normalize the data
mappedX = whiten(to_cluster_dat);
if use_PCA
    % Just the rows of the U matrix using PCA
    [r, ~, ~, U] = optimal_truncation(mappedX, 'rank');
    if r <= size(mappedX, 2)
        mappedX = U(:,1:r);
    else
        warning('Features are full-rank; no dimensionality reduction applied')
    end
end
% mappedX(:,3:end) = mappedX(:,3:end)./2; % These columns have repeats

these_names = names(neurons_with_activity);

disp('Finished building feature set')
%% Do ensemble clustering
%---------------------------------------------
% Build the Co-occurrence matrix using K-means
%---------------------------------------------
settings.which_metrics = {'silhouette', 'gap'};
settings.total_m = 1000;
settings.max_clusters = 10;
settings.cluster_func = @(X,K)(kmeans(X, K, 'emptyaction','singleton',...
    'replicate',1));

[co_occurrence, all_cluster_evals] = ensemble_clustering(mappedX, settings);

if calc_final_clusters
    %---------------------------------------------
    % Hierarchical clustering on the co-occurence matrix (and plot)
    %---------------------------------------------
    % Get 'best' number of clusters
    E = evalclusters(co_occurrence,...
        'linkage', 'silhouette', 'KList', 1:settings.max_clusters);
    if to_plot
        figure;
        plot(E)
    end
    k = E.OptimalK;

    idx = @(X) cluster(linkage(X,'Ward'), 'maxclust',k);
    [fig, c] = cluster_and_imagesc(co_occurrence, idx, these_names, []);
    title(sprintf('Number of clusters: %d', k))
    
    % Dendrogram
    tree = linkage(co_occurrence,'Ward');
    figure()
    cutoff = median([tree(end-k+1,3) tree(end-k+2,3)]);
    [H,T,outperm] = dendrogram(tree, 15, ...
        'Orientation','left','ColorThreshold',cutoff);
    tree_names = cell(length(outperm),1);
    for i = 1:length(outperm)
        tree_names{outperm==i} = strjoin(these_names(T==i), ';');
    end
    yticklabels(tree_names)
    title(sprintf('Dendrogram with %d clusters', k))
end

%% Save things for export
if nargout > 4
    if calc_final_clusters
        out = struct(...
            'neurons_with_activity', neurons_with_activity,...
            'all_cluster_evals', all_cluster_evals,...
            'dendrogram', H,...
            'E', E,...
            'cluster_figure', fig,...
            'final_indices', c);
    else
        out = struct(...
            'neurons_with_activity', {neurons_with_activity},...
            'all_cluster_evals', {all_cluster_evals},...
            'all_ind_fp', all_ind_fp,...
            'all_ind_fn', all_ind_fn);
    end
end

end
