


%% Define folder to save in
to_save = false;

foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\figures\';

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
my_cmap_dict_sort = containers.Map(... % For if they are sorted
    {1, 2, 3, 4, 5},... %sorted: FWD, DT, VT, REV, NOSTATE
    {3, 2, 1, 4, 5}); %Want: VT, DT, FWD, REV, NOSTATE

close all
%==========================================================================

%% Define 'ideal' settings
filename_template = ...
    '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';
filename_ideal = sprintf(filename_template, 5);
dat_ideal = importdata(filename_ideal);
num_neurons = size(dat_ideal.traces,2);

settings_ideal = struct(...
    'to_subtract_mean',false,...
    'to_subtract_baselines',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'augment_data', 7,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','func_DMDc',...
    'global_signal_subset', {{'DT', 'VT', 'REV', 'FWD', 'SLOW'}},...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings_ideal.global_signal_mode = 'ID_binary_transitions';

%---------------------------------------------
% Build filename array (different data formats)
%---------------------------------------------
n = 15;
all_filenames = cell(n, 1);
foldername1 = '../../Zimmer_data/WildType_adult/';
filename1_template = 'simplewt%d/wbdataset.mat';
num_type_1 = 5;
foldername2 = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\npr1_1_PreLet\';
filename2_template = 'wbdataset.mat';

for i = 1:n
    if i <= num_type_1
        all_filenames{i} = sprintf([foldername1, filename1_template], i);
    else
        subfolder = dir(foldername2);
        all_filenames{i} = [foldername2, ...
            subfolder(i-num_type_1+2).name, '\', filename2_template];
    end
end
%==========================================================================
error('You probably do not want to run the entire file')




%% Figure 1: Intro to transition signals
all_figs = cell(5,1);
%---------------------------------------------
% First build a model with good parameters
%---------------------------------------------
% Use CElegans model to preprocess
settings = settings_ideal;
settings.global_signal_mode = 'None';
my_model_fig1 = CElegansModel(filename_ideal, settings);

%---------------------------------------------
% Second get a representative neuron and control kicks
%---------------------------------------------
neuron = 'AVAL'; % Important in reversals
% tspan = 300:550;
tspan = 100:1000; % decided by hand

% Plot
% all_figs{1} = my_model_fig1.plot_reconstruction_interactive(false, 'AVAL');
neuron_ind = my_model_fig1.name2ind(neuron);
all_figs{1} = figure('DefaultAxesFontSize',12);
plot(my_model_fig1.dat(neuron_ind,:), 'LineWidth', 3);
xlim([tspan(1) tspan(end)])
xlabel('')
xticks([])
yticks([])
ylabel('')
set(gca, 'box', 'off')
% title(sprintf('Reconstruction of %s', neuron))
title(sprintf('Data for neuron %s', neuron))
legend('off')

my_model_fig1.set_simple_labels();
ctr = my_model_fig1.control_signal;

all_figs{2} = figure('DefaultAxesFontSize', 14);
% plot(ctr(5,tspan))
plot(ctr(3,tspan)+ctr(4,tspan), ...
    'color', my_cmap_3d(4,:), 'LineWidth', 4)
xticks([])
xlim([0, length(tspan)])
ylim([0, 0.5])
yticks([])
set(gca, 'box', 'off')
title('Reversal')

all_figs{3} = figure('DefaultAxesFontSize', 14);
plot(ctr(1,tspan), ...
    'color', my_cmap_3d(2,:), 'LineWidth', 4)
xticks([])
xlim([0, length(tspan)])
ylim([0, 0.5])
yticks([])
set(gca, 'box', 'off')
title('Dorsal Turn')

all_figs{4} = figure('DefaultAxesFontSize', 14);
plot(ctr(2,tspan), ...
    'color', my_cmap_3d(1,:), 'LineWidth', 4)
xticks([])
xlim([0, length(tspan)])
ylim([0, 0.5])
yticks([])
set(gca, 'box', 'off')
title('Ventral Turn')

all_figs{5} = figure('DefaultAxesFontSize', 14);
imagesc(my_model_fig1.state_labels_ind(tspan))
title('State labels')
xlabel('Time')
yticks([])
set(gca, 'box', 'off')
% Make the colormap work as expected
idx = [3 2 1 4];
colormap(my_cmap_3d(idx,:))

%---------------------------------------------
% Save the figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = sprintf('%sfigure_1_%d', foldername, i);
        this_fig = all_figs{i};
        if i == 1
            sz = {'0.9\columnwidth', '0.1\paperheight'};
        else
            sz = {'0.9\columnwidth', '0.025\paperheight'};
        end
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
%         prep_figure_no_axis(this_fig)
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Figure 2??: Explanation of time-delay embedding

% Save figures
for i = 1:length(all_figs)
    this_fig = all_figs{i};
    if isempty(this_fig)
        continue
    end
    prep_figure_no_axis(this_fig)
    zoom(1.2)
    colorbar off;
    fname = sprintf('%sfigure_3_%d', foldername, i);
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Figure 2?: Sparse residual analysis (i.e. controller learning)
all_figs = cell(6,1);

dat_struct = dat_ideal;
f_smooth = @(x) smoothdata(x, 'gaussian', 3);
dat_struct.traces = f_smooth(dat_struct.traces);
warning('USING PREPROCESSED DATA')

% First get a baseline model as a preprocessor
settings_base = settings_ideal;
settings_base.augment_data = 0;
my_model_base = CElegansModel(dat_struct, settings_base);
n = my_model_base.dat_sz(1);
m = my_model_base.dat_sz(2);

% which_sparsity = 60; % Manually choose the sparsity for now
num_iter = 100;
max_rank = 25;
all_U1 = cell(max_rank, 1);
all_acf = cell(max_rank, 1);

% Build the sparse signals; plot vs. sparsity LATER
settings = struct('num_iter', num_iter);
for i = 1:max_rank
    settings.r_ctr = i;
    [U, A, B] = sparse_residual_analysis(my_model_base, settings);

    all_U1{i} = zeros(i, m-1);
    % Choose sparsity based on max acf
    all_acf{i} = zeros(i, num_iter);
    which_sparsity = zeros(num_iter, 1);
    for i2 = 1:i
        for i3 = 1:num_iter
            dat = U{i3}(i2,:)';
            all_acf{i}(i2, i3) = acf(dat, 1, false);
        end
        [~, which_sparsity] = max(all_acf{i}(i2, :));
        all_U1{i}(i2,:) = U{which_sparsity}(i2,:);
    end
end

% Build data to plot
X1 = my_model_base.dat(:,1:end-1);
X2 = my_model_base.dat(:,2:end);

all_names = cell(max_rank, 1);
registered_lines = {};
registered_names = {};
num_steps_to_plot = 3;
for i = 1:max_rank
    all_names{i} = cell(1, i);
    U = all_U1{i};
    AB = X2 / [X1; U];
    A = AB(:, 1:n);
    An = A^(num_steps_to_plot-2);
    A_nondiag = A - diag(diag(A));
    B = AB(:, (n+1):end);
    for i2 = 1:i
        x = abs(B(:, i2));
        [this_maxk, this_ind] = maxk(x, 2);
        if this_maxk(1)/this_maxk(2) > 2
            this_ind = this_ind(1);
        end
        
        all_names{i}{:,i2} = my_model_base.get_names(this_ind, true);
        if ischar(all_names{i}{:,i2})
            all_names{i}{:,i2} = {all_names{i}{:,i2}};
        end
        % Names for registered lines graphing
        
        % First feature: max acf
        signal_name = sort(all_names{i}{:,i2});
        found_registration = false;
        [this_y, which_sparsity_in_rank] = max(all_acf{i}(i2, :));
        
%         this_dat = [i; this_y; i2];
        which_rank = i;
        which_line_in_rank = i2;
        this_dat = table(which_rank, this_y, which_line_in_rank,...
            which_sparsity_in_rank);
        for i4 = 1:length(registered_names)
            % Attach them to a previous line if it exists
            if isequal(registered_names{i4}, signal_name)
                registered_lines{i4} = [registered_lines{i4}; ...
                    this_dat]; %#ok<SAGROW>
                found_registration = true;
                break
            end
        end
        if ~found_registration
            if isempty(registered_names)
                registered_names = {signal_name};
            else
                registered_names = [registered_names {signal_name}]; %#ok<AGROW>
            end
            registered_lines = [registered_lines ...
                {this_dat} ]; %#ok<AGROW>
        end
    end
end

%---------------------------------------------
%% Get names and points to cluster for ALL plots
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
%% PLOT1: vs. rank
%---------------------------------------------
all_figs{1} = figure('DefaultAxesFontSize', 18);
hold on
n_lines = length(registered_lines);
% max_plot_rank = max_rank;
max_plot_rank = 10;
text_xy = zeros(n_lines, 2);
text_names = cell(n_lines, 1);
acf_threshold = 0.5;
text_opt = {'FontSize',14};
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

%---------------------------------------------
%% PLOT2: vs. sparsity
%---------------------------------------------
all_figs{2} = figure('DefaultAxesFontSize', 18);
[~, sort_ind] = sort(final_xy(:,2), 'descend');
imagesc(all_acf{rank_to_plot}(sort_ind, :));colorbar
% imagesc(all_acf{rank_to_plot});colorbar
yticks(1:rank_to_plot)
yticklabels(final_names(sort_ind))
xlabel('Sparsity')
title('Determination of sparsity')

%---------------------------------------------
%% PLOT3: Visual connection with experimentalist signals
%---------------------------------------------
% Note: replot the "correct" signals
tspan = 100:1000; % decided by hand, SAME AS ABOVE FIGURE
ctr = my_model_base.control_signal;

%---------------------------------------------
% Reversal figure
%---------------------------------------------
all_figs{3} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(4,:), 'LineWidth', 2};

subplot(2,1,1)
plot(ctr(5,tspan)+ctr(6,tspan), opt{:})
title('Reversal')

subplot(2,1,2)
registration_ind = find(contains(text_names, 'AVA'), 1);
this_line = registered_lines{registration_ind};
[~, rank_ind] = max(this_line{:, 'this_y'}); % Max over ranks AND sparsity
rank_ind = this_line{rank_ind, 'which_rank'};
line_ind = this_line{rank_ind, 'which_line_in_rank'};
learned_u_REV = all_U1{rank_ind}(line_ind, :);
plot(learned_u_REV(tspan), opt{:})
xlim([0 length(tspan)])
% title('AVA')

prep_figure_no_box_no_zoom(all_figs{3})

%---------------------------------------------
% Dorsal turn figure
%---------------------------------------------
all_figs{4} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(2,:), 'LineWidth', 2};

subplot(2,1,1)
plot(ctr(3,tspan), opt{:})
title('Dorsal Turn')

subplot(2,1,2)
registration_ind = find(contains(text_names, 'SMDD'), 1);
this_line = registered_lines{registration_ind};
[~, rank_ind] = max(this_line{:, 'this_y'}); % Max over ranks AND sparsity
rank_ind = this_line{rank_ind, 'which_rank'};
line_ind = this_line{rank_ind, 'which_line_in_rank'};
learned_u_DT = all_U1{rank_ind}(line_ind, :);
plot(learned_u_DT(tspan), opt{:})
xlim([0 length(tspan)])
% title('SMDD')

prep_figure_no_box_no_zoom(all_figs{4})

%---------------------------------------------
% Ventral turn figure
%---------------------------------------------
all_figs{5} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(1,:), 'LineWidth', 2};

subplot(2,1,1)
plot(ctr(4,tspan), opt{:})
title('Ventral Turn')

subplot(2,1,2)
registration_ind = find(contains(text_names, 'SMDV'), 1);
this_line = registered_lines{registration_ind};
[~, rank_ind] = max(this_line{:, 'this_y'}); % Max over ranks AND sparsity
rank_ind = this_line{rank_ind, 'which_rank'};
line_ind = this_line{rank_ind, 'which_line_in_rank'};
learned_u_VT = all_U1{rank_ind}(line_ind, :);
plot(learned_u_VT(tspan), opt{:})
xlim([0 length(tspan)])
% title('SMDV')

prep_figure_no_box_no_zoom(all_figs{5})

%---------------------------------------------
%% PLOT4: Boxplot of correlation with experimentalist signals
%---------------------------------------------
% First, need to re-run the code for getting all of the signals for other
% datafiles
num_files = length(all_filenames);
all_models = cell(num_files, 1);
all_U2 = cell(num_files, 1);
all_acf2 = cell(num_files, 1);
% For the preprocessor model
settings_base = settings_ideal;
settings_base.augment_data = 0;
settings_base.dmd_mode = 'no_dynamics'; % But we want the controllers
% For the residual analysis
settings = struct('num_iter', 100, 'r_ctr', 15);

% Get the control signals and acf
for i = 1:num_files
    dat_struct = importdata(all_filenames{i});
    dat_struct.traces = f_smooth(dat_struct.traces);
    warning('USING PREPROCESSED DATA')
    
    if i > num_type_1 % Differently named states
        settings_base.global_signal_subset = ...
            {'Reversal', 'Dorsal turn', 'Ventral turn'};
    end
    % First get a baseline model as a preprocessor
    all_models{i} = CElegansModel(dat_struct, settings_base);
    n = all_models{i}.dat_sz(1);
    m = all_models{i}.dat_sz(2);
    
    % Build the sparse signals
    [U, A, B] = sparse_residual_analysis(all_models{i}, settings);
    all_U2{i} = zeros(i, m-1);
    % Choose sparsity based on max acf
    all_acf2{i} = zeros(i, num_iter);
    which_sparsity = zeros(num_iter, 1);
    for i2 = 1:settings.r_ctr
        for i3 = 1:settings.num_iter
            dat = U{i3}(i2,:)';
            all_acf{i}(i2, i3) = acf(dat, 1, false);
        end
        [~, which_sparsity] = max(all_acf{i}(i2, :));
        all_U2{i}(i2,:) = U{which_sparsity}(i2,:);
    end
end
%% Get the maximum correlation with the experimentalist signals for each
% model
corr_threshold = [0.1, 0.3]; % Signals need an offset; want a threshold after as well
max_offset = 20;
correlation_table_raw = table();
f_dist = @(x) 1 - squareform(pdist(x, 'correlation'));
for i = 1:num_files
    tm = all_models{i};
    learned_ctr = all_U2{i};
    learned_n = size(learned_ctr, 1);
    exp_ctr = tm.control_signal(:,1:end-1);
    exp_n = size(exp_ctr, 1);
    this_dat = [exp_ctr; learned_ctr];
    all_corr = f_dist(this_dat);
    all_corr = all_corr - diag(diag(all_corr));
    
    key_ind = contains(tm.state_labels_key, tm.global_signal_subset);
    key = tm.state_labels_key(key_ind);
    model_index = i;
    for i2 = 1:length(key)
        [max_corr, max_ind] = max(all_corr(i2,:));
        if (max_corr > corr_threshold(1)) && (max_ind>exp_n)
            % See if an offset gives a better correlation
            best_offset = 0;
            for i3 = 1:max_offset
                % Positive offset
                tmp = f_dist([this_dat(i2,1:end-i3);...
                    this_dat(max_ind,i3+1:end)]);
                this_corr = tmp(1,2); % off-diagonal term
                if this_corr > max_corr
                    max_corr = this_corr;
                    best_offset = i3;
                end
                % Negative offset
                tmp = f_dist([this_dat(i2,i3+1:end);...
                    this_dat(max_ind,1:end-i3)]);
                this_corr = tmp(1,2); % off-diagonal term
                if this_corr > max_corr
                    max_corr = this_corr;
                    best_offset = -i3;
                end
            end
            
            if max_corr > corr_threshold(2)
                experimental_signal_index = i2;
                experimental_signal_name = key(i2);
                learned_signal_index = max_ind - exp_n;
                maximum_correlation = max_corr;
                correlation_table_raw = [correlation_table_raw;
                    table(model_index, maximum_correlation, best_offset,...
                    experimental_signal_index, experimental_signal_name,...
                    learned_signal_index)];
            end
        end
    end
    
%     figure;imagesc(all_corr);colorbar
%     ind = tm.state_labels_ind(1:end-1);
%     key = tm.state_labels_key;
%     fig = figure;
%     subplot(2,1,1)
%     plot_colored(this_dat(1,:), ind, key, 'plot', [], fig);
%     subplot(2,1,2)
%     plot_colored(this_dat(22,:), ind, key, 'plot', [], fig);
end

% Consolidate differently named states; keep only the max
correlation_table = correlation_table_raw;
name_dict = containers.Map(...
    {'REVSUS','REV1','REV2','Reversal', 'DT', 'Dorsal turn',...
        'VT', 'Ventral turn'},...
    {'Reversal', 'Reversal', 'Reversal', 'Reversal', ...
        'Dorsal Turn', 'Dorsal Turn', ...
        'Ventral Turn', 'Ventral Turn'});
for i = 1:size(correlation_table_raw,1)
    correlation_table{i,'experimental_signal_name'} = ...
        {name_dict(correlation_table{i,'experimental_signal_name'}{1})};
end
for i = 1:num_files
    while true
        ind = find(correlation_table{:,'model_index'}==i);
        these_names = correlation_table{ind, 'experimental_signal_name'};
        [~, unique_ind] = unique(these_names, 'first');
        repeat_ind = 1:length(these_names);
        repeat_ind(unique_ind) = [];
        if isempty(repeat_ind)
            break
        end
        this_name = correlation_table{...
            ind(repeat_ind(1)), 'experimental_signal_name'};
        these_repeated_ind = find(strcmp(...
            correlation_table{ind, 'experimental_signal_name'}, this_name));
        [~, sort_ind] = sort(correlation_table{...
            ind(these_repeated_ind),'maximum_correlation'}, 'descend');
        to_remove_ind_original_basis = ind(...
            these_repeated_ind(sort_ind(2:end)));
        correlation_table(to_remove_ind_original_basis, :) = [];
    end
end

% Actually plot (boxplot)
all_names = unique(name_dict.values);
all_figs{6} = figure('DefaultAxesFontSize', 14);
h = boxplot(correlation_table{:,'maximum_correlation'}, ...
    correlation_table{:, 'experimental_signal_name'})
% set(h,{'linew'},{2})
xtickangle(30)
ylabel('Correlation')
ylim([0 1])
title('Signals across 15 individuals')
%% Save figures
if to_save
    for i = 1:length(all_figs)
        this_fig = all_figs{i};
        if ~isvalid(this_fig) || isempty(this_fig)
            continue
        end
        if i>=3 && i<6
            figure(this_fig)
            yticks([])
            xticks([])
            set(gca, 'box', 'off')
%             sz = {'0.9\columnwidth', '0.025\paperheight'};
            sz = {'0.9\columnwidth', '0.1\paperheight'};
        elseif i>=6
            % Large title
            sz = {'0.9\columnwidth', '0.07\paperheight'};
        else
            sz = {'0.9\columnwidth', '0.1\paperheight'};
        end
        fname = sprintf('%sfigure_2_new_%d', foldername, i);
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
    %     zoom(1.2)
    %     colorbar off;
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Figure ?: Hold-out cross-validation
all_figs = cell(1,1);
filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';

% First the baseline settings
ad_settings = struct(...
    'hold_out_fraction',0.2,...
    'cross_val_window_size_percent', 0.8);
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'AdaptiveDmdc_settings',ad_settings);
settings.global_signal_mode = 'ID_binary_and_grad';

num_worms = 5;
err_train = zeros(200, num_worms);
err_test = zeros(200,num_worms);
for i=1:num_worms
    filename = sprintf(filename_template,i);
    my_model_crossval = CElegansModel(filename, settings);

    % Use crossvalidation functions
    err_train(:,i) = my_model_crossval.AdaptiveDmdc_obj.calc_baseline_error();
    err_test(:,i) = my_model_crossval.AdaptiveDmdc_obj.calc_test_error();
end

all_figs{1} = figure('DefaultAxesFontSize',14);
boxplot(err_test,'colors',[1 0 0])
hold on
boxplot(err_train)
ylim([0, 1.1*max(max([err_test;err_train]))])
ylabel('L2 error')
xlabel('Worm ID number')
title('Training and Test Data Reconstruction')

% Save figures
if to_save
    fname = sprintf('%sfigure_3b_%d', foldername, 1);
    this_fig = all_figs{1};
    matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex']);
    prep_figure_no_axis(this_fig)
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Figure 3a-c: Reconstructions (multiple methods)
all_figs = cell(12,1);
tspan = [500, 1500];

ind = my_cmap_dict_sort.values;
my_cmap_figure3 = my_cmap_3d([ind{:}],:);

%---------------------------------------------
% Get time-delayed model (best)
%---------------------------------------------
my_model_fig3_a = CElegansModel(filename_ideal, settings_ideal);

% Original data; same for all models
my_model_fig3_a.set_simple_labels();
new_labels_key = my_model_fig3_a.state_labels_key;
all_figs{1} = my_model_fig3_a.plot_colored_data(false, 'plot', my_cmap_figure3);
view(my_viewpoint)

% 3d pca plot
all_figs{2} = my_model_fig3_a.plot_colored_reconstruction(my_cmap_figure3);
view(my_viewpoint)

% Reconstruct some individual neurons
neur_labels = {'AVAL', 'SMDDL'};
neur_id = [my_model_fig3_a.name2ind(neur_labels{1}), ...
    my_model_fig3_a.name2ind(neur_labels{2})];
fig_dict = containers.Map(...
    {neur_id(1), neur_id(2)},...
    {8, 9});
for i = 1:length(neur_id)
    this_n = neur_id(i);
    all_figs{fig_dict(this_n)} = ...
        my_model_fig3_a.plot_reconstruction_interactive(false,this_n);
    title(neur_labels{i})
    xlim(tspan);
    set(gca, 'box', 'off')
    ylabel('')
    xlabel('')
end

%---------------------------------------------
% Compare with no time delay
%---------------------------------------------
settings = settings_ideal;
settings.augment_data = 0;
my_model_fig3_b = CElegansModel(filename_ideal, settings);

my_model_fig3_b.set_simple_labels();
all_figs{3} = my_model_fig3_b.plot_colored_reconstruction(my_cmap_figure3);
view(my_viewpoint)

% Get individual neurons (same as above)
fig_dict = containers.Map(...
    {neur_id(1), neur_id(2)}, {10, 11});
for i = 1:length(neur_id)
    this_n = neur_id(i);
    all_figs{fig_dict(this_n)} = ...
        my_model_fig3_b.plot_reconstruction_interactive(false,this_n);
    title(neur_labels{i})
    xlim(tspan);
    set(gca, 'box', 'off')
    ylabel('')
    xlabel('')
end

%---------------------------------------------
% Simplest comparison: no control at all!
%---------------------------------------------
settings = settings_ideal;
settings.augment_data = 0;
settings.global_signal_mode = 'None';
settings.dmd_mode = 'tdmd';
my_model_fig3_c = CElegansModel(filename_ideal, settings);

my_model_fig3_c.set_simple_labels();
all_figs{4} = my_model_fig3_c.plot_colored_reconstruction(my_cmap_figure3);
view(my_viewpoint)
% Now make the colormap match the bar graphs
for i=1:length(new_labels_key)
    all_figs{4}.Children(2).Children(i).CData = ...
        my_cmap_3d(my_cmap_dict(i),:);
end

%---------------------------------------------
% Save figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i}) || ~isvalid(all_figs{i})
            continue;
        end
        fname = sprintf('%sfigure_4_%d', foldername, i);
        this_fig = all_figs{i};
        if i <= 4 % 3d PCA plots
            ax = this_fig.Children(2);
            ax.Clipping = 'Off';
            prep_figure_no_axis(this_fig)
            zoom(1.14)
            if i==1
                zoom(1.05)
            end
        end
        if i >= 5 % Histograms and Single neurons
            if i >=8 %Single neurons
                prep_figure_no_axis(this_fig)
            end
            sz = {'0.9\columnwidth', '0.14\paperheight'}
            matlab2tikz('figurehandle',this_fig,'filename',...
                [fname '_raw.tex'], ...
                'width', sz{1}, 'height', sz{2});
        end
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================
%% Figure 3s: CDF
all_figs = cell(1);
% Use models from above to get correlation data
td_correlations = real(my_model_fig3_a.calc_correlation_matrix());
one_correlations = real(my_model_fig3_b.calc_correlation_matrix());
nc_correlations = real(my_model_fig3_c.calc_correlation_matrix());

n = length(one_correlations);
td_correlations = td_correlations(1:n);

% Build cumulative distribution vectors
num_pts = 1000;
x = linspace(0, 1, num_pts);
td_cdf = zeros(size(td_correlations));
one_cdf = td_cdf;
nc_cdf = td_cdf;
for i = 1:num_pts
    td_cdf(i) = length(find(td_correlations >= x(i)))/n;
    one_cdf(i) = length(find(one_correlations >= x(i)))/n;
    nc_cdf(i) = length(find(nc_correlations >= x(i)))/n;
end

% Plot CDF
all_figs{1} = figure('DefaultAxesFontSize', 16);
plot(x, td_cdf, 'LineWidth', 3)
hold on
plot(x, one_cdf, 'LineWidth', 3)
plot(x, nc_cdf, 'LineWidth', 3)
ylabel('Fraction of Neurons')
xlabel('Correlation Coefficient (cumulative)')
set(gca, 'box', 'off')
legend({'Time Delay', 'Single Step', 'No Control'})
legend boxoff

% Also plot some circles for the individual neuron reconstructions
% Use variable from above
to_plot_circles = false;
if to_plot_circles
    f_close = @(val) find(x>val,1); % Find closest value (in the x vector)
    for i = 1:length(neur_id)
        neur = neur_id(i);
        td_x = td_correlations(neur);
    %     text(td_x, td_cdf(f_close(td_x)), neur_labels{i})
        plot(td_x, td_cdf(f_close(td_x)), 'ko', 'LineWidth', 5)
        one_x = one_correlations(neur);
    %     text(one_x, one_cdf(f_close(one_x)), neur_labels{i})
        plot(one_x, one_cdf(f_close(one_x)), 'ko', 'LineWidth', 5)
    end
end

%---------------------------------------------
% Save figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = sprintf('%sfigure_4s_%d', foldername, i);
        this_fig = all_figs{i};
        sz = {'0.9\columnwidth', '0.1\paperheight'}
        matlab2tikz('figurehandle',this_fig,'filename',...
            [fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================
%% Figure 3d: time-delay embeddings 
all_figs = cell(1);
max_augment = 12;
all_models = cell(max_augment+1,1);

% Original model
settings = settings_ideal;
settings.augment_data = 0;
all_models{1} = CElegansModel(filename_ideal, settings);

n = all_models{1}.dat_sz(1);
all_corr = zeros(max_augment + 1, n);
all_corr(1,:) = all_models{1}.calc_correlation_matrix(false, 'SVD');

for i = 1:max_augment
    settings.augment_data = i;
    
    all_models{i+1} = CElegansModel(filename_ideal, settings);
    % Note: we only want the correlations with the last datapoint
    tmp = all_models{i+1}.calc_correlation_matrix(false, 'SVD');
    all_corr(i+1,:) = tmp(1:n);
end

% Plot the different correlation coefficients
all_figs{1} = figure('DefaultAxesFontSize', 14);
boxplot(real(all_corr)');
title('Non-delayed neurons in time-delay embedded models')
ylabel('Correlation coefficient')
xlabel('Delay embedding')
xticklabels(0:max_augment)

%---------------------------------------------
% Save the figures
%---------------------------------------------
if to_save
    fname =  [foldername 'figure_4_12'];
    this_fig = all_figs{1};
    matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex']);
    sz = {'0.9\columnwidth', '0.05\paperheight'}
    matlab2tikz('figurehandle',this_fig,'filename',...
        [fname '_raw.tex'], ...
        'width', sz{1}, 'height', sz{2});
end

%==========================================================================


%% Figure 4a-d version 2: Variable selection
all_figs = cell(4,1);
% Get the 'ideal' (time delayed) model
my_model_time_delay = CElegansModel(filename_ideal, settings_ideal);

%---------------------------------------------
% LASSO from a direct fit to data
%---------------------------------------------
U2 = my_model_time_delay.control_signal(:,2:end);
X1 = my_model_time_delay.dat(:,1:end-1);
disp('Fitting lasso models...')
all_intercepts_td = zeros(size(U2,1),1);
B_prime_lasso_td = zeros(size(U2,1), size(X1,1));
which_fit = 8;
for i = 1:size(U2,1)
    [all_fits, fit_info] = lasso(X1', U2(i,:), 'NumLambda',10);
    all_intercepts_td(i) = fit_info.Intercept(which_fit);
    B_prime_lasso_td(i,:) = all_fits(:,which_fit); % Which fit = determined by eye
end

%---------------------------------------------
%% Plot the unrolled matrix of one control signal (Dorsal Turn)
%---------------------------------------------
% which_ctr = 1;
which_ctr = 3;
names = my_model_time_delay.get_names([], true);
% ind = contains(my_model_time_delay.state_labels_key, {'REV', 'DT', 'VT'});
% Unroll one of the controllers
unroll_sz = [my_model_time_delay.original_sz(1), settings_ideal.augment_data];
dat_unroll1 = reshape(B_prime_lasso_td(which_ctr,:), unroll_sz);

all_figs{1} = figure('DefaultAxesFontSize', 14);
[ordered_dat, ordered_ind] = top_ind_then_sort(dat_unroll1, [], 1e-2);
imagesc(ordered_dat)
colormap(cmap_white_zero(ordered_dat));
% colorbar
% title(sprintf('All predictors for control signal %d', which_ctr))
% title('Predictors for Dorsal Turn (all neurons)')
title('Predictors for Dorsal Turn control signal')
yticks(1:unroll_sz(1))
yticklabels(names(ordered_ind))
xlabel('Number of delay frames')

%---------------------------------------------
% Plot reconstructions of some control signals
%---------------------------------------------
% tspan = 100:1000;
% Get the reconstructions
X1 = my_model_time_delay.dat(:,1:end-1);
ctr_reconstruct = [my_model_time_delay.control_signal(which_ctr,1)...
    B_prime_lasso_td(which_ctr,:) * X1];
ctr_reconstruct_td = ctr_reconstruct + all_intercepts_td(which_ctr);
ctr = my_model_time_delay.control_signal(which_ctr,:);

% Get the threshold
f = @(x) minimize_false_detection(ctr, ...
    ctr_reconstruct_td, x, 0.1);
this_threshold = fminsearch(f, 1.5);

% Plot
[~, ~, ~, ~, ~, all_figs{2}] = ...
    calc_false_detection(ctr, ctr_reconstruct_td,...
            this_threshold, [], [], true);
% title('Sparse Reconstruction (all neurons)')
title('Sparse Reconstruction')
set(gca, 'box', 'off')
legend off
yticklabels('')
ylabel('Arbitrary units')
% xticklabels('')
xlabel('Time (s)')
% title(sprintf('Sparse reconstruction of control signal %d',which_ctr))

%---------------------------------------------
% Save figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i}) || is_invalid_gui_obj_handle(all_figs{i})
            continue;
        end
        fname = sprintf('%sfigure_5a_%d', foldername, i);
        this_fig = all_figs{i};
%         if i >= 2
%             prep_figure_no_axis(this_fig)
%         end
        sz = {'0.9\columnwidth', '0.12\paperheight'};
        matlab2tikz('figurehandle',this_fig,'filename',...
            [fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================
%% Figure 4e: Elimination path (Lasso)
all_figs = cell(3,1);
%---------------------------------------------
% Iteratively remove most important neurons
%---------------------------------------------
% max_iter = 5;
max_iter = 20;
which_fit = 4;

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

for i = 1:max_iter
    fprintf('Iteration %d...\n', i)
    % Remove the top neurons from the last round
    if i > 1
        [~, top_ind] = max(abs(B_prime_lasso_td_3d(:, :, i-1)), [], 2);
        % We'll get a single time slice of a neuron, but want to remove all
        % copies (cumulatively)
        top_ind = mod(top_ind,n) + n*(0:settings_ideal.augment_data-1);
        elimination_neurons(:,i) = top_ind(:,1);
        elimination_pattern(:,:,i) = elimination_pattern(:,:,i-1);
        for i4 = 1:size(top_ind,1)
            elimination_pattern(i4,top_ind(i4,:),i) = true;
        end
    end
    % Fit new Lasso models
    for i2 = 1:size(U2,1)
        this_X1 = X1;
        this_X1(elimination_pattern(i2,:,i),:) = 0;
        [all_fits, fit_info] = lasso(this_X1', U2(i2,:), 'NumLambda',5);
        B_prime_lasso_td_3d(i2, :, i) = all_fits(:,which_fit); % Which fit = determined by eye
        all_intercepts_td(i2, i) = fit_info.Intercept(which_fit);
    end
    % Get the reconstructions of the control signals
    ctr_tmp = B_prime_lasso_td_3d(:, :, i) * X1;
    ctr_reconstruct_td = [ctr(:,1), ctr_tmp + all_intercepts_td(:, i)];
    
    % Get the error metric
%     this_corr = corrcoef([ctr_reconstruct_td' ctr']);
%     all_err(:,i) = diag(this_corr, num_ctr);
    for i2 = 1:num_ctr
        % Find a threshold which is best for the all-neuron
        % reconstruction
        f = @(x) minimize_false_detection(ctr(i2,:), ...
            ctr_reconstruct_td(i2,:), x, 0.1);
        all_thresholds_3d(i2, i) = fminsearch(f, 1);
        [all_fp(i2,i), all_fn(i2,i), num_spikes(i)] = ...
            calc_false_detection(ctr(i2,:), ctr_reconstruct_td(i2,:),...
            all_thresholds_3d(i2, i));
    end
end

%---------------------------------------------
% Plot the false positives/negatives
%---------------------------------------------
all_figs{1} = figure('DefaultAxesFontSize', 16);
i = 1; % Dorsal Turn
plot(all_fp(i,:) / num_spikes(i), 'LineWidth',2)
hold on
plot(all_fn(i,:) / num_spikes(i), 'LineWidth',2)
legend({'False Positives', 'False Negatives'},'Location','northwest')

a = arrayfun(@(x)my_model_time_delay.get_names(x,true),...
    elimination_neurons(:,2:end), 'UniformOutput',false);
eliminated_names = cellfun(@(x)x{1},a,'UniformOutput',false);

set(gca, 'box', 'off')
xlim([1 max_iter])
xticks(1:max_iter)
xticklabels(['All neurons', eliminated_names(i,:)])
xtickangle(60)
% xlabel('Eliminated neuron (cumulative)')
ylabel('Percentage of transitions')
title('Elimination Path')
% disp(eliminated_names)

%---------------------------------------------
% Plot the unrolled matrix of one control signal
%---------------------------------------------
which_iter = 5;
which_ctr = 1;
names = my_model_time_delay.get_names([], true);
% ind = contains(my_model_time_delay.state_labels_key, {'REV', 'DT', 'VT'});
% Unroll one of the controllers
unroll_sz = [my_model_time_delay.original_sz(1), settings_ideal.augment_data];
dat_unroll1 = reshape(B_prime_lasso_td_3d(which_ctr,:,which_iter), unroll_sz);

all_figs{2} = figure('DefaultAxesFontSize', 16);
[ordered_dat, ordered_ind] = top_ind_then_sort(dat_unroll1, [], 1e-2);
imagesc(ordered_dat)
colormap(cmap_white_zero(ordered_dat));
% colorbar
title(sprintf('Predictors with %d neurons eliminated', which_iter))
% title('Predictors for Dorsal Turn control signal')
yticks(1:unroll_sz(1))
yticklabels(names(ordered_ind))
xlabel('Number of delay frames')

%---------------------------------------------
% Plot a reconstruction
%---------------------------------------------
% Get the reconstructions
X1 = my_model_time_delay.dat(:,1:end-1);
ctr_reconstruct = [my_model_time_delay.control_signal(which_ctr,1)...
    B_prime_lasso_td_3d(which_ctr,:,which_iter) * X1];
ctr_reconstruct_td = ctr_reconstruct + ...
    all_intercepts_td(which_ctr, which_iter);

% Plot
ctr = my_model_time_delay.control_signal(which_ctr,:);
[~, ~, ~, ~, ~, all_figs{3}] = ...
    calc_false_detection(ctr, ctr_reconstruct_td,...
            all_thresholds_3d(which_ctr, which_iter), [], [], true);
% title(sprintf('Reconstruction of control signal %d for iteration %d', ...
%     which_ctr, which_iter))
title('Sparse Reconstruction')
set(gca, 'box', 'off')
legend off
yticklabels('')
ylabel('Arbitrary units')
% xticklabels('')
xlabel('Time (s)')

%---------------------------------------------
% Save figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i}) || ~isvalid(all_figs{i})
            continue;
        end
        fname = sprintf('%sfigure_5e2_%d', foldername, i);
        this_fig = all_figs{i};
        sz = {'0.9\columnwidth', '0.12\paperheight'};
        matlab2tikz('figurehandle',this_fig,'filename',...
            [fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Figure 6: Clustering on dynamic features
all_figs = cell(4,1);
rng(13);

%---------------------------------------------
% Get feature set from models
%---------------------------------------------
my_features = {'Correlation','L2_distance','dtw',...
    'FN_norm','FP_norm','TP_norm','FN','FP','TP'};
sz = 5;
all_co_occurrence = cell(sz,1);
all_names = cell(sz,1);
all_models = cell(sz,1);
all_dat = cell(sz, 1);
for i = 1:sz
    filename = sprintf(filename_template, i);
    [all_co_occurrence{i}, all_names{i}, all_models{i}, all_dat{i}] = ...
        zimmer_dynamic_clustering(filename, my_features, false);
end

disp('Finished building all co-occurrence matrices')

%---------------------------------------------
% Build the global name list
%---------------------------------------------
sz = 5;
max_length = 5; % Only keep unambiguously identified neurons
total_names = {};
for i = 1:sz
    this_names = all_names{i};
    to_keep_ind = false(size(this_names));
    for i2 = 1:length(this_names)
        x = this_names{i2};
        to_keep_ind(i2) = ( length(x) <=max_length && ...
            ~isempty(regexp(x,'\D(?#any non-digit)', 'once')));
    end
    
    total_names = union(total_names, this_names(to_keep_ind));
end

%---------------------------------------------
% Build the total co-occurrence matrix
%---------------------------------------------
total_co_occurrence_3d = nan(length(total_names),length(total_names),sz);
num_identifications = zeros(size(total_names));

for i = 1:sz
    [this_names, this_ind] = sort(all_names{i});
    this_dat = all_co_occurrence{i}(this_ind,this_ind);
    [~, ~, ib] = unique(this_names, 'stable');
    repeat_names = find(hist(ib, unique(ib))>1);
    if ~isempty(repeat_names)
        this_names(strcmp(this_names{repeat_names}, this_names)) = [];
    end
    
    to_keep_names = intersect(this_names, total_names);
    ind_in_individual = cellfun(@(x) any(strcmp(x, to_keep_names)),...
        this_names);
    ind_in_total = cellfun(@(x) any(strcmp(x, to_keep_names)),...
        total_names);
    num_identifications = num_identifications + ind_in_total;
    
    total_co_occurrence_3d(ind_in_total, ind_in_total, i) = ...
        this_dat(ind_in_individual, ind_in_individual);
end

% Remove rarely identified neurons
min_identifications = 3;
to_keep_ind = (num_identifications >= min_identifications);
total_co_occurrence_3d = total_co_occurrence_3d(to_keep_ind, to_keep_ind, :);

%---------------------------------------------
% Normalize according to how many times the neuron appeared
%---------------------------------------------
tmp = total_co_occurrence_3d;
tmp = tmp - min(min(tmp));
sz = length(find(to_keep_ind));
I = logical(eye(sz));
for i3 = 1:size(tmp,3)
    for i = 1:sz
        for i2 = 1:sz
            if i==i2
                continue;
            end
            tmp(i,i2,i3) = tmp(i,i2,i3) / min([tmp(i,i,i3), tmp(i2,i2,i3)]);
        end
    end
    tmp(:,:, i3) = tmp(:,:, i3) - diag(diag(tmp(:,:, i3))) + I;
end
total_co_occurrence_std = std(tmp, 0, 3, 'omitnan');
tmp(isnan(tmp)) = 0;
total_co_occurrence = mean(tmp, 3);
total_names = total_names(to_keep_ind);

disp('Finished building TOTAL co-occurrence matrix')

%---------------------------------------------
% Hierarchical clustering on the co-occurence matrix and plot
%---------------------------------------------
rng(4);
% Get 'best' number of clusters
E = evalclusters(co_occurrence,...
    'linkage', 'silhouette', 'KList', 1:settings.max_clusters);
all_figs{4} = figure('DefaultAxesFontSize',16); % Note: for the supplement
plot(E)
k = E.OptimalK;

% Heatmap
idx = E.OptimalY;
[all_figs{1}, c, all_ind] = cluster_and_imagesc(...
    co_occurrence, idx, these_names, []);
title(sprintf('Number of clusters: %d', k))

% Dendrogram
tree = linkage(co_occurrence,'Ward');
all_figs{2} = figure('DefaultAxesFontSize', 16);
cutoff = median([tree(end-k+1,3) tree(end-k+2,3)]);
[H, T, outperm] = dendrogram(tree, 15, ...
    'Orientation','left','ColorThreshold',cutoff);
tree_names = cell(length(outperm),1);
for i = 1:length(outperm)
    tree_names{outperm==i} = strjoin(these_names(T==i), ';');
end
yticklabels(tree_names)
title(sprintf('Dendrogram with %d clusters', k))

% Silhouette diagram
all_figs{3} = figure('DefaultAxesFontSize',16);
[all_scores_table, S] = silhouette_with_names(...
    total_co_occurrence, c, total_names, [all_ind{:}]);

%---------------------------------------------
% Save
%---------------------------------------------
to_save = false;
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i}) || ~isvalid(all_figs{i})
            continue;
        end
        fname = sprintf('%sfigure_cluster_%d', foldername, i);
        this_fig = all_figs{i};
        sz = {'0.9\columnwidth', '0.25\paperheight'};
        matlab2tikz('figurehandle',this_fig,'filename',...
            [fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end

%==========================================================================


