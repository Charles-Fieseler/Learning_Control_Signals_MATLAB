
%% Define folder to save in and data folders
to_save = false;
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper_local\Supplement\Supplemental_figures\';
my_viewpoint = [0, 80];

%==========================================================================
%% Define colormaps
%---------------------------------------------
% Set up colormap for RPCA visualizations
%---------------------------------------------
num_colors = 16;
my_cmap_RPCA = brewermap(num_colors,'OrRd');
my_cmap_RPCA(1,:) = ones(1,3);
colormap(my_cmap_RPCA)
my_cmap_RPCA(end,:) = zeros(1,3);
set(0, 'DefaultFigureColormap', my_cmap_RPCA)
% caxis([-0.5,1.0])

%---------------------------------------------
% Set up colormap for 3d visualizations
%---------------------------------------------
num_possible_roles = 4;
my_cmap_3d = colormap(parula(num_possible_roles));
my_cmap_3d = [my_cmap_3d; zeros(1,3)];
my_cmap_3d(4,:) = [249, 222, 12]./256; % Make yellow more visible

% Dict for switching between simple label plots and bar-graph-matching
% colors
my_cmap_dict = containers.Map(...
    {1, 2, 3, 4, 5},... %original: NOSTATE, REV, VT, DT, FWD
    {5, 4, 1, 2, 3}); %Want: VT, DT, FWD, REV, NOSTATE

close all
%==========================================================================
%% Define the base settings
[all_filenames, num_type_1] = get_Zimmer_filenames();

filename_ideal = all_filenames{5};
dat_ideal = importdata(filename_ideal);
num_neurons = size(dat_ideal.traces,2);

settings_ideal = define_ideal_settings();
%==========================================================================
error('You probably do not want to run the entire file')


%% SECTION 1: Controller learning
%% Setup: General
all_figs = cell(7,1);

dat_struct = dat_ideal;

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
% max_rank = 25; % For the supplement
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
n_lines = length(registered_lines);
cluster_xy = zeros(n_lines, 1);
final_xy = zeros(rank_to_plot,2);
final_names = cell(rank_to_plot,1);
for i = 1:n_lines
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
%---------------------------------------------

%% Setup: Full path of a signal
disp('Getting full path of for a single rank...')
rank_to_plot = 5;
settings = struct('r_ctr', rank_to_plot, 'num_iter', num_iter);    

[all_U_path, all_A, all_B, ~] = ...
    sparse_residual_analysis(my_model_base, settings);

my_model_simple = my_model_base;
my_model_simple.set_simple_labels();
my_model_simple.remove_all_control();
my_model_simple.calc_all_control_signals();
ctr = my_model_simple.control_signal;
%% Plot 1a: Full path of a good signal (REV)
%---------------------------------------------
% Two rows to this figure:
%   Top = 5 panels with increasing sparsity for a signal
%   Bottom = autocorrelation across the entire sparsity iteration

tspan = 100:1000; % decided by hand, SAME AS ABOVE FIGURE
all_figs{1} = figure('DefaultAxesFontSize', 16);

% Plot 5 example traces, with acf titles
% which_neur = 'AVAR';
which_neur = 'AVAL';
registration_ind = find(contains(text_names, which_neur), 1);
this_line = registered_lines{registration_ind};
rank_to_plot_table = find(this_line{:, 'which_rank'} == rank_to_plot);
if isempty(rank_to_plot_table)
    error('Signal not found in path matrix; rerun sparse_residual_analysis')
end
rev_ind = this_line{rank_to_plot_table, 'which_line_in_rank'};
rev_learned = all_U1{rank_to_plot}(rev_ind, :);

% Get previously calculated autocorrelation
rev_acf = all_acf{rank_to_plot}(rev_ind, :);
[~, max_ind] = max(rev_acf);
iter_to_plot = sort([1, 25, 50, 100, max_ind]);
title_str = 'Iteration %d';
title_str_special = 'Iteration %d (Maximum)';
which_max = find(iter_to_plot==max_ind);

% Calculate number of nonzeros, i.e. sparsity
rev_nnz = zeros(size(rev_acf));
for i=1:length(rev_acf)
    tmp = all_U_path{i}(rev_ind, :);
    rev_nnz(i) = round(nnz(tmp)/numel(tmp), 2);
end

opt = {'color', my_cmap_3d(4,:), 'LineWidth', 2}; % Match paper colors

% Panels 1-5: Varying sparsity
for i = 1:length(iter_to_plot)
    this_iter = iter_to_plot(i);
    subplot(2,5,i)
%     plot(remove_isolated_spikes(all_U_path{this_iter}(rev_ind, tspan)),...
%         opt{:})
    plot(all_U_path{this_iter}(rev_ind, tspan),opt{:})
    if i==which_max
        title(sprintf(title_str_special, this_iter))
    else
        title(sprintf(title_str, this_iter))
    end
    xlim([0 length(tspan)])
    ylim([0 1])
    xlabel('Time')
    if i==1
        ylabel('Amplitude')
    end
end

% Panel 6 i.e. the second row
subplot(2,1,2)
plot(rev_acf, 'LineWidth', 2)
hold on
for i = 1:length(iter_to_plot)
    x = iter_to_plot(i);
    plot(x, rev_acf(x), 'ko', 'LineWidth', 3);
end
title(sprintf('Signals that actuate neuron %s', which_neur))
xlabel('Fraction nonzero frames')
ylabel('Autocorrelation')
xticks(iter_to_plot)
xticklabels(rev_nnz(iter_to_plot))

% Next figure!
% Panel 1 (expert signal)
all_figs{2} = figure('DefaultAxesFontSize', 16);
opt = {'color', my_cmap_3d(4,:), 'LineWidth', 2};
subplot(2,1,1)
plot(ctr(4,:), opt{:})
title('Expert labeled onset of Reversal')
ylabel('Amplitude')
xlabel('Time')
xlim([0 size(ctr, 2)])

% Learned signal
subplot(2,1,2)
plot(remove_isolated_spikes(rev_learned))
xlim([0 size(ctr, 2)])
title('Learned onset of Reversal')
ylabel('Amplitude')
xlabel('Time')
%---------------------------------------------
%% Plot 1b: Full path of a good signal (VT)
%---------------------------------------------
% Two rows to this figure:
%   Top = 5 panels with increasing sparsity for a signal
%   Bottom = autocorrelation across the entire sparsity iteration

tspan = 100:1000; % decided by hand, SAME AS ABOVE FIGURE
all_figs{3} = figure('DefaultAxesFontSize', 16);

% Plot 5 example traces, with acf titles
which_neur = 'SMDVR';
registration_ind = find(contains(text_names, which_neur), 1);
this_line = registered_lines{registration_ind};
rank_to_plot_table = find(this_line{:, 'which_rank'} == rank_to_plot);
if isempty(rank_to_plot_table)
    error('Signal not found in path matrix; rerun sparse_residual_analysis')
end
rev_ind = this_line{rank_to_plot_table, 'which_line_in_rank'};

rev_acf = all_acf{rank_to_plot}(rev_ind, :);
[~, max_ind] = max(rev_acf);
iter_to_plot = sort([1, 25, 50, 100, max_ind]);
title_str = 'Iteration %d';
title_str_special = 'Iteration %d (Maximum)';
which_max = find(iter_to_plot==max_ind);
% Calculate number of nonzeros, i.e. sparsity
rev_nnz = zeros(size(rev_acf));
for i=1:length(rev_acf)
    tmp = all_U_path{i}(rev_ind, :);
    rev_nnz(i) = round(nnz(tmp)/numel(tmp), 2);
end

opt = {'color', my_cmap_3d(1,:), 'LineWidth', 2}; % Match paper colors

% Panels 1-5: Varying sparsity
for i = 1:length(iter_to_plot)
    this_iter = iter_to_plot(i);
    subplot(2,5,i)
    plot(all_U_path{this_iter}(rev_ind, tspan),opt{:})
    if i==which_max
        title(sprintf(title_str_special, this_iter))
    else
        title(sprintf(title_str, this_iter))
    end
    xlim([0 length(tspan)])
    ylim([0 1])
    xlabel('Time')
    if i==1
        ylabel('Amplitude')
    end
end

% Panel 6 i.e. the second row
subplot(2,1,2)
plot(rev_acf, 'LineWidth', 2)
hold on
for i = 1:length(iter_to_plot)
    x = iter_to_plot(i);
    plot(x, rev_acf(x), 'ko', 'LineWidth', 3);
end
title(sprintf('Signals that actuate neuron %s', which_neur))
xlabel('Fraction nonzero frames')
ylabel('Autocorrelation')
xticks(iter_to_plot)
xticklabels(rev_nnz(iter_to_plot))

% Next figure!
% Panel 1 (expert signal)
all_figs{4} = figure('DefaultAxesFontSize', 16);
opt = {'color', my_cmap_3d(1,:), 'LineWidth', 2};
subplot(2,1,1)
plot(ctr(3,:), opt{:})
title('Expert labeled onset of Ventral Turn')
ylabel('Amplitude')
xlabel('Time')
xlim([0 size(ctr, 2)])

% Learned signal
subplot(2,1,2)
plot(remove_isolated_spikes(all_U1{rank_to_plot}(rev_ind, :)))
xlim([0 size(ctr, 2)])
title('Learned onset of Ventral Turn')
ylabel('Amplitude')
xlabel('Time')
%---------------------------------------------
%% Plot 1c: Full path of a decent signal (1, i.e. sensory)
%---------------------------------------------
% Two rows to this figure:
%   Top = 5 panels with increasing sparsity for a signal
%   Bottom = autocorrelation across the entire sparsity iteration

tspan = 100:1000; % decided by hand, SAME AS ABOVE FIGURE
all_figs{5} = figure('DefaultAxesFontSize', 16);

% Plot 5 example traces, with acf titles
which_neur = '1';
registration_ind = find(contains(text_names, which_neur), 1);
this_line = registered_lines{registration_ind};
rank_to_plot_table = find(this_line{:, 'which_rank'} == rank_to_plot);
if isempty(rank_to_plot_table)
    error('Signal not found in path matrix; rerun sparse_residual_analysis')
end
rev_ind = this_line{rank_to_plot_table, 'which_line_in_rank'};

rev_acf = all_acf{rank_to_plot}(rev_ind, :);
[~, max_ind] = max(rev_acf);
iter_to_plot = sort([1, 25, 50, 100, max_ind]);
title_str = 'Iteration %d';
title_str_special = 'Iteration %d (Maximum)';
which_max = find(iter_to_plot==max_ind);
% Calculate number of nonzeros, i.e. sparsity
rev_nnz = zeros(size(rev_acf));
for i=1:length(rev_acf)
    tmp = all_U_path{i}(rev_ind, :);
    rev_nnz(i) = round(nnz(tmp)/numel(tmp), 2);
end

opt = {'color', [0.5 0.5 0.5], 'LineWidth', 2}; % Match paper colors

% Panels 1-5: Varying sparsity
for i = 1:length(iter_to_plot)
    this_iter = iter_to_plot(i);
    subplot(2,5,i)
    plot(all_U_path{this_iter}(rev_ind, tspan),opt{:})
    if i==which_max
        title(sprintf(title_str_special, this_iter))
    else
        title(sprintf(title_str, this_iter))
    end
    xlim([0 length(tspan)])
    ylim([0 1])
    xlabel('Time')
    if i==1
        ylabel('Amplitude')
    end
end

% Panel 6 i.e. the second row
subplot(2,1,2)
plot(rev_acf, 'LineWidth', 2)
hold on
for i = 1:length(iter_to_plot)
    x = iter_to_plot(i);
    plot(x, rev_acf(x), 'ko', 'LineWidth', 3);
end
title(sprintf('Signals that actuate neuron %s', which_neur))
xlabel('Fraction nonzero frames')
ylabel('Autocorrelation')
xticks(iter_to_plot)
xticklabels(rev_nnz(iter_to_plot))
%---------------------------------------------
%% Plot 1d: Full path of a bad signal (RIML)
%---------------------------------------------
% Two rows to this figure:
%   Top = 5 panels with increasing sparsity for a signal
%   Bottom = autocorrelation across the entire sparsity iteration

tspan = 100:1000; % decided by hand, SAME AS ABOVE FIGURE
all_figs{6} = figure('DefaultAxesFontSize', 16);

% Plot 5 example traces, with acf titles
which_neur = 'RIML';
registration_ind = find(contains(text_names, which_neur), 1);
this_line = registered_lines{registration_ind};
rank_to_plot_table = find(this_line{:, 'which_rank'} == rank_to_plot);
if isempty(rank_to_plot_table)
    error('Signal not found in path matrix; rerun sparse_residual_analysis')
end
rev_ind = this_line{rank_to_plot_table, 'which_line_in_rank'};

rev_acf = all_acf{rank_to_plot}(rev_ind, :);
[~, max_ind] = max(rev_acf);
iter_to_plot = sort([1, 25, 50, 100, max_ind]);
title_str = 'Iteration %d';
title_str_special = 'Iteration %d (Maximum)';
which_max = find(iter_to_plot==max_ind);
% Calculate number of nonzeros, i.e. sparsity
rev_nnz = zeros(size(rev_acf));
for i=1:length(rev_acf)
    tmp = all_U_path{i}(rev_ind, :);
    rev_nnz(i) = round(nnz(tmp)/numel(tmp), 2);
end

opt = {'color', [0 0 0], 'LineWidth', 2}; % Match paper colors

% Panels 1-5: Varying sparsity
for i = 1:length(iter_to_plot)
    this_iter = iter_to_plot(i);
    subplot(2,5,i)
    plot(all_U_path{this_iter}(rev_ind, tspan),opt{:})
    if i==which_max
        title(sprintf(title_str_special, this_iter))
    else
        title(sprintf(title_str, this_iter))
    end
    xlim([0 length(tspan)])
    ylim([0 1])
    xlabel('Time')
    if i==1
        ylabel('Amplitude')
    end
end

% Panel 6 i.e. the second row
subplot(2,1,2)
plot(rev_acf, 'LineWidth', 2)
hold on
for i = 1:length(iter_to_plot)
    x = iter_to_plot(i);
    plot(x, rev_acf(x), 'ko', 'LineWidth', 3);
end
title(sprintf('Signals that actuate neuron %s', which_neur))
xlabel('Fraction nonzero frames')
ylabel('Autocorrelation')
xticks(iter_to_plot)
xticklabels(rev_nnz(iter_to_plot))
%---------------------------------------------
%% Plot 2: Signals as rank increases
%---------------------------------------------
all_figs{7} = figure('DefaultAxesFontSize', 18);
hold on
n_lines = length(registered_lines);
% max_plot_rank = max_rank;
max_plot_rank = 10;
% text_xy = zeros(n_lines, 2);
% text_names = cell(n_lines, 1);
acf_threshold = 0.5;
text_opt = {'FontSize',10};
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
    % Determine the colormap via cluster membership
    if ismember(i, real_clust)
        if contains(this_name, 'AVA')
            line_opt = {'Color', my_cmap_3d(4,:)};
        elseif contains(this_name, 'SMDD')
            line_opt = {'Color', my_cmap_3d(2,:)};
        elseif contains(this_name, 'SMDV')
            line_opt = {'Color', my_cmap_3d(1,:)};
        else
            line_opt = {};
        end
    elseif ismember(i, gray_clust)
        line_opt = {'Color', [0,0,0]+0.5};
    else
        line_opt = {'Color', 'k'};
    end
    % Actually plot
    if length(find(these_ranks_ind)) > 1
        plot(xy{these_ranks_ind,'which_rank'}, xy{these_ranks_ind,'this_y'},...
            'LineWidth',2, line_opt{:})
    else
        plot(xy{these_ranks_ind,'which_rank'}, xy{these_ranks_ind,'this_y'},...
            'o', line_opt{:})
    end
end
textfit(text_xy(:,1), text_xy(:,2), text_names, text_opt{:})
ylim([0, 1])
ylabel('Maximum autocorrelation')
xlabel('Rank')
title('Determination of rank truncation')
%% Save plots (one dataset)
if to_save
    for i = 1:length(all_figs)
        this_fig = all_figs{i};
        if ~isvalid(this_fig) || isempty(this_fig)
            continue
        end
        if i>=1 && i<7
%             figure(this_fig)
%             yticks([])
%             xticks([])
%             set(gca, 'box', 'off')
            sz = {'0.9\columnwidth', '0.2\paperheight'};
        elseif i>=5
            % Large title
            sz = {'0.9\columnwidth', '0.2\paperheight'};
        else
            sz = {'0.9\columnwidth', '0.1\paperheight'};
        end
        fname = sprintf('%ssupplement_fig_learn_%d', foldername, i);
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================

%% Setup, Plot, and Save: Other individuals
%% Dataset 2
dat_struct = importdata(all_filenames{2});
all_figs = cell(4, 1);
setup_control_learning_figures
plot_control_learning_figures

if to_save
    for i = 1:length(all_figs)
        this_fig = all_figs{i};
        if ~isvalid(this_fig) || isempty(this_fig)
            continue
        end
        if i>=1 && i<5
            figure(this_fig)
            yticks([])
            xticks([])
            set(gca, 'box', 'off')
%             sz = {'0.9\columnwidth', '0.025\paperheight'};
            sz = {'0.9\columnwidth', '0.12\paperheight'};
        elseif i>=7
            % Large title
            sz = {'0.9\columnwidth', '0.1\paperheight'};
        else
            sz = {'0.9\columnwidth', '0.1\paperheight'};
        end
        fname = sprintf('%ssupplement_fig_learn2_%d', foldername, i);
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
    %     zoom(1.2)
    %     colorbar off;
        saveas(this_fig, fname, 'png');
    end
end
%% Dataset 3
dat_struct = importdata(all_filenames{3});
all_figs = cell(4, 1);
setup_control_learning_figures
plot_control_learning_figures

if to_save
    for i = 1:length(all_figs)
        this_fig = all_figs{i};
        if ~isvalid(this_fig) || isempty(this_fig)
            continue
        end
        if i>=1 && i<5
            figure(this_fig)
            yticks([])
            xticks([])
            set(gca, 'box', 'off')
%             sz = {'0.9\columnwidth', '0.025\paperheight'};
            sz = {'0.9\columnwidth', '0.12\paperheight'};
        elseif i>=7
            % Large title
            sz = {'0.9\columnwidth', '0.1\paperheight'};
        else
            sz = {'0.9\columnwidth', '0.1\paperheight'};
        end
        fname = sprintf('%ssupplement_fig_learn3_%d', foldername, i);
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
    %     zoom(1.2)
    %     colorbar off;
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================



%% SECTION 2: Variable selection and encoding

%% Ventral Turn
%% Elimination path (Lasso)
all_figs = cell(7,1);
% Get the 'ideal' (time delayed) model
settings = settings_ideal;
settings.augment_data = 6;
my_model_time_delay = CElegansModel(filename_ideal, settings);
%---------------------------------------------
% Iteratively remove most important neurons
%---------------------------------------------
% max_iter = 5;
max_iter = 20;
which_fit = 4; %TODO: max LASSO/sparsity smarter

U2 = my_model_time_delay.control_signal(:,2:end);
X1 = my_model_time_delay.dat(:,1:end-1);
n = my_model_time_delay.original_sz(1);
num_ctr = size(U2,1);
disp('Calculating elimination path for each controller...')
all_intercepts_td = zeros(num_ctr, max_iter);
all_err = zeros(num_ctr, max_iter);
all_fp = zeros(num_ctr, max_iter);
all_fn = zeros(num_ctr, max_iter);
all_thresholds_3d = zeros(num_ctr, max_iter);
B_prime_lasso_td_3d = zeros(num_ctr, size(X1,1), max_iter);
elimination_pattern = false(size(B_prime_lasso_td_3d));
elimination_neurons = zeros(size(all_err));
num_spikes = zeros(num_ctr, 1);

ctr = my_model_time_delay.control_signal;

to_fit_all_models = false;
which_single_model = 4;

for i = 1:max_iter
    fprintf('Iteration %d...\n', i)
    % Remove the top neurons from the last round
    if i > 1
        [~, top_ind] = max(abs(B_prime_lasso_td_3d(:, :, i-1)), [], 2);
        % We'll get a single time slice of a neuron, but want to remove all
        % copies (cumulatively)
        top_ind = mod(top_ind,n) + n*(0:my_model_time_delay.augment_data-1);
        elimination_neurons(:,i) = top_ind(:,1);
        elimination_pattern(:,:,i) = elimination_pattern(:,:,i-1);
        for i4 = 1:size(top_ind,1)
            elimination_pattern(i4,top_ind(i4,:),i) = true;
        end
    end
    % Fit new Lasso models
    for i2 = 1:num_ctr
        if ~to_fit_all_models
            i2 = which_single_model;
        end
        this_X1 = X1;
        this_X1(elimination_pattern(i2,:,i),:) = 0;
        [all_fits, fit_info] = lasso(this_X1', U2(i2,:), 'NumLambda',5);
        B_prime_lasso_td_3d(i2, :, i) = all_fits(:,which_fit); % Which fit = determined by eye
        all_intercepts_td(i2, i) = fit_info.Intercept(which_fit);
        if ~to_fit_all_models
            break;
        end
    end
    % Get the reconstructions of the control signals
    ctr_tmp = B_prime_lasso_td_3d(:, :, i) * X1;
    ctr_reconstruct_td = [ctr(:,1), ctr_tmp + all_intercepts_td(:, i)];
    
    for i2 = 1:num_ctr
        if ~to_fit_all_models
            i2 = which_single_model;
        end
        % Find a threshold which is best for the all-neuron
        % reconstruction
        f = @(x) minimize_false_detection(ctr(i2,:), ...
            ctr_reconstruct_td(i2,:), x, 0.1);
        all_thresholds_3d(i2, i) = fminsearch(f, 1);
        % Old style flat threshold
        [all_fp(i2,i), all_fn(i2,i), num_spikes(i), ...
            ~, ~, true_pos, ~, ~, true_neg] = ...
            calc_false_detection(ctr(i2,:), ctr_reconstruct_td(i2,:),...
            all_thresholds_3d(i2, i), [], [], false, true, false);
        % "new" findpeaks on derivative version
%         [all_fp(i2,i), all_fn(i2,i), num_spikes(i)] = ...
%             calc_false_detection(ctr(i2,:), ctr_reconstruct_td(i2,:),...
%             all_thresholds_3d(i2, i), [], [], false, true);
        %   Matthew's correlation coefficient
        all_err(i2,i) = calc_mcc(...
            true_pos, true_neg, all_fp(i2,i), all_fn(i2,i));
        if ~to_fit_all_models
            break;
        end
    end
    
    % Also calculate another the error metric
    %   Correlation
%     this_corr = corrcoef([ctr_reconstruct_td' ctr']);
%     all_err(:,i) = diag(this_corr, num_ctr);
    %   L2 error
%     all_err(:,i) = vecnorm(ctr_reconstruct_td - ctr, 2, 2);
end

%---------------------------------------------
%% Plot the false positives/negatives
%---------------------------------------------
all_figs{1} = figure('DefaultAxesFontSize', 24);
which_ctr = 4;
max_plot_iter = 19;
plot(all_fp(which_ctr,1:max_plot_iter) / num_spikes(which_ctr), 'LineWidth',2)
hold on
plot(all_fn(which_ctr,1:max_plot_iter) / num_spikes(which_ctr), 'LineWidth',2)
legend({'False Positives', 'False Negatives'},'Location','northwest')

which_examples = [1 5 13];
for i2 = 1:length(which_examples)
    x = which_examples(i2);
%     scatter(x, all_fn(i, x), 'HandleVisibility','off', ...
%         'LineWidth', 3,'MarkerEdgeColor', 'k')
    plot(x, all_fn(which_ctr, x)/ num_spikes(which_ctr), ...
        'ok', 'HandleVisibility','off', ...
        'MarkerSize', 5,'LineWidth', 2)
end
xlim([1 max_plot_iter])

a = arrayfun(@(x)my_model_time_delay.get_names(x,true),...
    elimination_neurons(:,2:end), 'UniformOutput',false);
eliminated_names = cellfun(@(x)x{1},a,'UniformOutput',false);

set(gca, 'box', 'off')
xlim([1 max_plot_iter])
xticks(1:max_plot_iter)
xticklabels(['All neurons', eliminated_names(which_ctr,1:max_plot_iter)])
% xticklabels(['All neurons', eliminated_names(1:max_plot_iter)])
xtickangle(60)
ylabel('Fraction of events')
title('Elimination Path for Ventral Turn')

%---------------------------------------------
% Plot the data for an early iteration
which_iter = which_examples(1);
[all_figs{2}, all_figs{3}, this_caxis, this_cmap] = ...
    plot_signal_reconstruction_w_threshold(...
    my_model_time_delay, B_prime_lasso_td_3d, all_intercepts_td,...
    all_thresholds_3d, ...
    which_ctr, which_iter);

%---------------------------------------------
% Plot the data for a later iteration
which_iter = which_examples(2);
[all_figs{4}, all_figs{5}] = plot_signal_reconstruction_w_threshold(...
    my_model_time_delay, B_prime_lasso_td_3d, all_intercepts_td,...
    all_thresholds_3d, ...
    which_ctr, which_iter, this_caxis, this_cmap);

%---------------------------------------------
% Plot the data for a later iteration
which_iter = which_examples(3);
[all_figs{6}, all_figs{7}] = plot_signal_reconstruction_w_threshold(...
    my_model_time_delay, B_prime_lasso_td_3d, all_intercepts_td,...
    all_thresholds_3d, ...
    which_ctr, which_iter, this_caxis, this_cmap);
%---------------------------------------------
%% Save Ventral Turn figures
%---------------------------------------------
if to_save
%     save([dat_foldername 'elimination_path'], ...
%         'all_intercepts_td', 'all_err', 'all_fp', 'all_fn', ...
%         'all_thresholds_3d', 'elimination_neurons', 'num_spikes', ...
%         'B_prime_lasso_td_3d');
    
    for i = 1:length(all_figs)
        if isempty(all_figs{i}) || ~isvalid(all_figs{i})
            continue;
        end
        fname = sprintf('%ssupplement_figure_vt_%d', foldername, i);
        this_fig = all_figs{i};
        % Full screen
        set(this_fig, 'Position', get(0, 'Screensize'));
        if i==1 % Elimination path
            sz = {'0.9\columnwidth', '0.09\paperheight'};
        elseif mod(i,2)==0 % Variable selection
            sz = {'0.9\columnwidth', '0.08\paperheight'};
        elseif mod(i,2)==1 % Reconstructions
            sz = {'0.9\columnwidth', '0.05\paperheight'};
        end
        matlab2tikz('figurehandle',this_fig,'filename',...
            [fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================

%% Reversals (simple)
%% Elimination path (Lasso)
all_figs = cell(7,1);
% Get the 'ideal' (time delayed) model
settings = settings_ideal;
settings.augment_data = 6;
my_model_time_delay = CElegansModel(filename_ideal, settings);
my_model_time_delay.set_simple_labels();
my_model_time_delay.remove_all_control();
my_model_time_delay.build_model();
%---------------------------------------------
% Iteratively remove most important neurons
%---------------------------------------------
% max_iter = 5;
max_iter = 20;
which_fit = 4; %TODO: max LASSO/sparsity smarter

U2 = my_model_time_delay.control_signal(:,2:end);
X1 = my_model_time_delay.dat(:,1:end-1);
n = my_model_time_delay.original_sz(1);
num_ctr = size(U2,1);
disp('Calculating elimination path for each controller...')
all_intercepts_td = zeros(num_ctr, max_iter);
all_err = zeros(num_ctr, max_iter);
all_fp = zeros(num_ctr, max_iter);
all_fn = zeros(num_ctr, max_iter);
all_thresholds_3d = zeros(num_ctr, max_iter);
B_prime_lasso_td_3d = zeros(num_ctr, size(X1,1), max_iter);
elimination_pattern = false(size(B_prime_lasso_td_3d));
elimination_neurons = zeros(size(all_err));
num_spikes = zeros(num_ctr, 1);

ctr = my_model_time_delay.control_signal;

to_fit_all_models = false;
which_single_model = 4; % Note: this is the simplified indexing

for i = 1:max_iter
    fprintf('Iteration %d...\n', i)
    % Remove the top neurons from the last round
    if i > 1
        [~, top_ind] = max(abs(B_prime_lasso_td_3d(:, :, i-1)), [], 2);
        % We'll get a single time slice of a neuron, but want to remove all
        % copies (cumulatively)
        top_ind = mod(top_ind,n) + n*(0:my_model_time_delay.augment_data-1);
        elimination_neurons(:,i) = top_ind(:,1);
        elimination_pattern(:,:,i) = elimination_pattern(:,:,i-1);
        for i4 = 1:size(top_ind,1)
            elimination_pattern(i4,top_ind(i4,:),i) = true;
        end
    end
    % Fit new Lasso models
    for i2 = 1:num_ctr
        if ~to_fit_all_models
            i2 = which_single_model;
        end
        this_X1 = X1;
        this_X1(elimination_pattern(i2,:,i),:) = 0;
        [all_fits, fit_info] = lasso(this_X1', U2(i2,:), 'NumLambda',5);
        B_prime_lasso_td_3d(i2, :, i) = all_fits(:,which_fit); % Which fit = determined by eye
        all_intercepts_td(i2, i) = fit_info.Intercept(which_fit);
        if ~to_fit_all_models
            break;
        end
    end
    % Get the reconstructions of the control signals
    ctr_tmp = B_prime_lasso_td_3d(:, :, i) * X1;
    ctr_reconstruct_td = [ctr(:,1), ctr_tmp + all_intercepts_td(:, i)];
    
    for i2 = 1:num_ctr
        if ~to_fit_all_models
            i2 = which_single_model;
        end
        % Find a threshold which is best for the all-neuron
        % reconstruction
        f = @(x) minimize_false_detection(ctr(i2,:), ...
            ctr_reconstruct_td(i2,:), x, 0.1);
        all_thresholds_3d(i2, i) = fminsearch(f, 1);
        % Old style flat threshold
        [all_fp(i2,i), all_fn(i2,i), num_spikes(i), ...
            ~, ~, true_pos, ~, ~, true_neg] = ...
            calc_false_detection(ctr(i2,:), ctr_reconstruct_td(i2,:),...
            all_thresholds_3d(i2, i), [], [], false, true, false);
        % "new" findpeaks on derivative version
%         [all_fp(i2,i), all_fn(i2,i), num_spikes(i)] = ...
%             calc_false_detection(ctr(i2,:), ctr_reconstruct_td(i2,:),...
%             all_thresholds_3d(i2, i), [], [], false, true);
        %   Matthew's correlation coefficient
        all_err(i2,i) = calc_mcc(...
            true_pos, true_neg, all_fp(i2,i), all_fn(i2,i));
        if ~to_fit_all_models
            break;
        end
    end
    
    % Also calculate another the error metric
    %   Correlation
%     this_corr = corrcoef([ctr_reconstruct_td' ctr']);
%     all_err(:,i) = diag(this_corr, num_ctr);
    %   L2 error
%     all_err(:,i) = vecnorm(ctr_reconstruct_td - ctr, 2, 2);
end

%---------------------------------------------
%% Plot the false positives/negatives
%---------------------------------------------
all_figs{1} = figure('DefaultAxesFontSize', 24);
which_ctr = 4;
max_plot_iter = 19;
plot(all_fp(which_ctr,1:max_plot_iter) / num_spikes(which_ctr), 'LineWidth',2)
hold on
plot(all_fn(which_ctr,1:max_plot_iter) / num_spikes(which_ctr), 'LineWidth',2)
legend({'False Positives', 'False Negatives'},'Location','northwest')

which_examples = [1 5 13];
for i2 = 1:length(which_examples)
    x = which_examples(i2);
%     scatter(x, all_fn(i, x), 'HandleVisibility','off', ...
%         'LineWidth', 3,'MarkerEdgeColor', 'k')
    plot(x, all_fn(which_ctr, x)/ num_spikes(which_ctr), ...
        'ok', 'HandleVisibility','off', ...
        'MarkerSize', 5,'LineWidth', 2)
end
xlim([1 max_plot_iter])

a = arrayfun(@(x)my_model_time_delay.get_names(x,true),...
    elimination_neurons(:,2:end), 'UniformOutput',false);
eliminated_names = cellfun(@(x)x{1},a,'UniformOutput',false);

set(gca, 'box', 'off')
xlim([1 max_plot_iter])
xticks(1:max_plot_iter)
xticklabels(['All neurons', eliminated_names(which_ctr,1:max_plot_iter)])
% xticklabels(['All neurons', eliminated_names(1:max_plot_iter)])
xtickangle(60)
ylabel('Fraction of events')
title('Elimination Path for Reversal')

%---------------------------------------------
% Plot the data for an early iteration
which_iter = which_examples(1);
[all_figs{2}, all_figs{3}, this_caxis, this_cmap] = ...
    plot_signal_reconstruction_w_threshold(...
    my_model_time_delay, B_prime_lasso_td_3d, all_intercepts_td,...
    all_thresholds_3d, ...
    which_ctr, which_iter);

%---------------------------------------------
% Plot the data for a later iteration
which_iter = which_examples(2);
[all_figs{4}, all_figs{5}] = plot_signal_reconstruction_w_threshold(...
    my_model_time_delay, B_prime_lasso_td_3d, all_intercepts_td,...
    all_thresholds_3d, ...
    which_ctr, which_iter, this_caxis, this_cmap);

%---------------------------------------------
% Plot the data for a later iteration
which_iter = which_examples(3);
[all_figs{6}, all_figs{7}] = plot_signal_reconstruction_w_threshold(...
    my_model_time_delay, B_prime_lasso_td_3d, all_intercepts_td,...
    all_thresholds_3d, ...
    which_ctr, which_iter, this_caxis, this_cmap);
%---------------------------------------------
%% Save Reversal figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i}) || ~isvalid(all_figs{i})
            continue;
        end
        fname = sprintf('%ssupplement_figure_rev_%d', foldername, i);
        this_fig = all_figs{i};
        % Full screen
        set(this_fig, 'Position', get(0, 'Screensize'));
        if i==1 % Elimination path
            sz = {'0.9\columnwidth', '0.09\paperheight'};
        elseif mod(i,2)==0 % Variable selection
            sz = {'0.9\columnwidth', '0.08\paperheight'};
        elseif mod(i,2)==1 % Reconstructions
            sz = {'0.9\columnwidth', '0.05\paperheight'};
        end
        matlab2tikz('figurehandle',this_fig,'filename',...
            [fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================



%% Reconstructions: Build all models

settings_full = settings_ideal;

% settings_global_only = settings_ideal;
% settings_global_only.lambda_sparse = 0;

settings_no_control = settings_ideal;
settings_no_control.global_signal_mode = 'None';
% settings_no_control.lambda_sparse = 0;
settings_no_control.dmd_mode = 'tdmd';

%---------------------------------------------
% Settings for the colormaps
%---------------------------------------------
ind = my_cmap_dict_sort.values;
my_cmap_reconstructions = my_cmap_3d([ind{:}],:);

%---------------------------------------------
% Build all models
%---------------------------------------------
% n = length(all_filenames);
n = 4; % Don't actaully plot all of them
all_models_full = cell(n,1);
% all_models_global_only = cell(n,1);
all_models_no_control = cell(n,1);

for i = 1:n
    dat_struct = importdata(all_filenames{i});
    if i > num_type_1
        % A lot of the prelet data files have very bad initial frames
        dat_struct.traces = dat_struct.traces(100:end,:);
    end
    all_models_full{i} = CElegansModel(dat_struct, settings_full);
%     all_models_global_only{i} = ...
%         CElegansModel(dat_struct, settings_global_only);
    all_models_no_control{i} = ...
        CElegansModel(dat_struct, settings_no_control);
end
%==========================================================================
%% Reconstructions: Plotting
all_figs_data = cell(n,1);
all_figs_full = cell(n,1);
% all_figs_global_only = cell(n,1);
all_figs_no_control = cell(n,1);

%---------------------------------------------
% Create figures
%---------------------------------------------
for i = 1:n
    m = all_models_full{i};
    m.set_simple_labels();
    all_figs_full{i} = m.plot_colored_reconstruction(my_cmap_reconstructions);
    view(my_viewpoint)
    title('Supervised Control Signals')
    
%     m = all_models_global_only{i};
%     m.set_simple_labels();
%     all_figs_global_only{i} = m.plot_colored_reconstruction(false);
%     title('Only proportional control')
    
    m = all_models_no_control{i};
    m.set_simple_labels();
    all_figs_no_control{i} = m.plot_colored_reconstruction(my_cmap_reconstructions);
    title('No control')
    view(my_viewpoint)
    
    all_figs_data{i} = m.plot_colored_data(false, 'o', my_cmap_reconstructions);
    view(my_viewpoint)
    drawnow;
end

all_figs = [all_figs_data', all_figs_no_control', all_figs_full'];

%---------------------------------------------
%% Save figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = sprintf('%ssupplement_figure_reconstruction_%d', ...
            foldername, i);
        this_fig = all_figs{i};
%             ax = this_fig.Children(2);
%             ax.Clipping = 'Off';
%             prep_figure_no_axis(this_fig);
%             zoom(1.12);
        set(this_fig, 'Position', get(0, 'Screensize'));
%         prep_figure_tight_axes(this_fig);
        saveas(this_fig, fname, 'png');
    end
end

%==========================================================================





%% Scratch work
error('Scratch work')


%% Systematic 3d explorations of time-delay embedding and filtering
% filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);

settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','func_DMDc',...
    'global_signal_subset', {{'DT','VT','REV'}},...
    ...'autocorrelation_noise_threshold', 0.4,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

% Loop through different amounts of time-delay embedding
max_augment = 12;
max_filter = 6;
all_models = cell(max_augment+1, max_filter);

% Original model
all_models{1, 1} = CElegansModel(filename, settings);

n = all_models{1, 1}.dat_sz(1);
all_corr = zeros(max_augment+1, max_filter, n);
all_corr(1,1,:) = all_models{1,1}.calc_correlation_matrix();

for i = 0:max_augment
    settings.augment_data = i;
    for i2 = 1:max_filter
        if i == 0 && i2 == 1 
            continue
        end
        settings.filter_window_dat = i2;

        all_models{i+1, i2} = CElegansModel(filename, settings);
        % Note: we only want the correlations with the last datapoint
        tmp = all_models{i+1, i2}.calc_correlation_matrix(false, 'SVD');
        all_corr(i+1,i2,:) = tmp(1:n);
    end
end

% Plot the different correlation coefficients
figure;
boxPlot3D(permute(real(all_corr),[3 1 2]));
title('Non-delayed neurons in time-delay embedded models')
zlabel('Correlation coefficient')
xlabel('Delay embedding')
xticks(1:(max_augment+1))
xticklabels(0:max_augment)
ylabel('Filter window')
yticks(1:max_filter)
yticklabels(1:max_filter)

figure;
all_mins = min(real(all_corr),[],3)';
imagesc(all_mins)
my_cmap = cmap_white_zero(all_mins);
colormap(my_cmap)
colorbar
title('Minimum of correlation coefficient')
xlabel('Delay embedding')
xticks(1:(max_augment+1))
xticklabels(0:max_augment)
ylabel('Filter window')
yticks(1:max_filter)
yticklabels(1:max_filter)
%==========================================================================

%% Systematically analyze increasing sparsity
dat_struct = importdata(filename_base);
settings_sparse = settings;
settings_sparse.global_signal_mode = 'ID_binary_transitions';
settings_sparse.augment_data = 7;
settings_sparse.global_signal_subset = {'DT','VT','REV'};

min_sparse_factor = 0.0;
max_sparse_factor = 2.0;
sparse_step = 0.1;
sparse_vec = min_sparse_factor:sparse_step:max_sparse_factor;
sz = length(sparse_vec);
all_models_sparse = cell(sz, 1);
n = size(dat_struct.traces,2);
all_corr_sparse = zeros(sz, n*settings_sparse.augment_data);
all_nnz = zeros(sz,1);

for i = 1:sz
    ad_settings = struct('sparse_tol_factor', sparse_vec(i));
    settings_sparse.AdaptiveDmdc_settings = ad_settings;
%     all_models_sparse{i} = CElegansModel(filename, settings_sparse);
%     all_corr_sparse(i,:) = all_models_sparse{i}.calc_correlation_matrix();
    A = all_models_sparse{i}.AdaptiveDmdc_obj.A_original;
    A(abs(A)<1e-4) = 0;
    all_nnz(i) = nnz(A(1:n,1:n*settings_sparse.augment_data));
end

% Plot
figure
boxplot(real(all_corr_sparse)', round(all_nnz/(settings.augment_data*n^2),2));
xlabel('Percent of non-zero terms')
% xlabel('Number of non-zero terms')
title(sprintf('Models with delay embedding = %d', settings.augment_data))
ylabel('Correlation')
%==========================================================================