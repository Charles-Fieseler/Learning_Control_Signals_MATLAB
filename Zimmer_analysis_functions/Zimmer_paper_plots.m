


%% Define folder to save in
to_save = false;

foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\figures\';

my_viewpoint = [0, 90];
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
%---------------------------------------------
% Build filename array (different data formats)
%---------------------------------------------
[all_filenames, num_type_1] = get_Zimmer_filenames();

filename_ideal = all_filenames{5};
dat_ideal = importdata(filename_ideal);
num_neurons = size(dat_ideal.traces,2);

settings_ideal = define_ideal_settings();

dat_foldername = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_analysis_git\dat\';
%==========================================================================
error('You probably do not want to run the entire file')




%% Figure 1: Intro to transition signals
all_figs = cell(5,1);
%---------------------------------------------
% First build a model with good parameters
%---------------------------------------------
% Use CElegans model to preprocess
settings = settings_ideal;
my_model_fig1 = CElegansModel(filename_ideal, settings);

%---------------------------------------------
% Second get a representative neuron and control kicks
%---------------------------------------------
neuron = 'AVAL'; % Important in reversals
% tspan = 300:550;
tspan = 100:1000; % decided by hand

% Plot
% all_figs{1} = my_model_fig1.plot_reconstruction_interactive(false, 'AVAL');
% neuron_ind = my_model_fig1.name2ind(neuron);
% all_figs{1} = figure('DefaultAxesFontSize',12);
% plot(my_model_fig1.dat(neuron_ind,:), 'LineWidth', 3);
% xlim([tspan(1) tspan(end)])
% xlabel('')
% xticks([])
% yticks([])
% ylabel('Calcium amplitude')
% set(gca, 'box', 'off')
% % title(sprintf('Reconstruction of %s', neuron))
% % title(sprintf('Data for neuron %s', neuron))
% title('Trace of a Reversal-active neuron')
% legend('off')

% Make the colormap work as expected
idx = [3 2 1 4 5];
this_cmap = my_cmap_3d(idx,:);
my_model_fig1.set_simple_labels();

% all_figs{1} = my_model_fig1.plot_colored_neuron(neuron, this_cmap);
all_figs{1} = figure('DefaultAxesFontSize', 14);
subplot(3,1,1);
my_model_fig1.plot_colored_neuron(neuron, this_cmap, all_figs{1});
xlim([tspan(1), tspan(end)])
xlabel('')
xticks([])
yticks([])
ylabel('Calcium amplitude')
set(gca, 'box', 'off')
title('A Reversal-active Neuron')
legend('off')

% Get the control signals from the simple labeling
my_model_fig1.remove_all_control();
my_model_fig1.calc_all_control_signals();
ctr = my_model_fig1.control_signal;

opt = {'LineWidth', 4};

% Reversal
% all_figs{2} = figure('DefaultAxesFontSize', 14);
subplot(6,1,3);
plot(ctr(4,tspan), 'color', my_cmap_3d(4,:), opt{:})
xticks([])
xlim([0, length(tspan)])
ylim([0, 0.5])
yticks([])
set(gca, 'box', 'off')
title('Onset of Reversal')

% Forward
% all_figs{5} = figure('DefaultAxesFontSize', 14);
subplot(6,1,4);
plot(ctr(1,tspan), 'color', my_cmap_3d(3,:), opt{:})
xticks([])
xlim([0, length(tspan)])
ylim([0, 0.5])
yticks([])
set(gca, 'box', 'off')
title('Onset of Forward State')

% Dorsal
% all_figs{3} = figure('DefaultAxesFontSize', 14);
subplot(6,1,5);
plot(ctr(2,tspan), 'color', my_cmap_3d(2,:), opt{:})
xticks([])
xlim([0, length(tspan)])
ylim([0, 0.5])
yticks([])
set(gca, 'box', 'off')
title('Onset of Dorsal Turn')

% Ventral
% all_figs{4} = figure('DefaultAxesFontSize', 14);
subplot(6,1,6);
plot(ctr(3,tspan), 'color', my_cmap_3d(1,:), opt{:})
xticks([])
xlim([0, length(tspan)])
ylim([0, 0.5])
yticks([])
set(gca, 'box', 'off')
title('Onset of Ventral Turn')
xlabel('Time')


% States
% all_figs{5} = figure('DefaultAxesFontSize', 14);
% imagesc(my_model_fig1.state_labels_ind(tspan))
% title('State labels')
% xlabel('Time')
% yticks([])
% set(gca, 'box', 'off')
% % Make the colormap work as expected
% idx = [3 2 1 4];
% colormap(my_cmap_3d(idx,:))

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
%         if i == 1
%             sz = {'0.9\columnwidth', '0.1\paperheight'};
%         else
%             sz = {'0.9\columnwidth', '0.025\paperheight'};
%         end
        sz = {'0.9\columnwidth', '0.3\paperheight'};
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
all_figs = cell(7,1);

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

% Loop through sparsity and rank to get a set of control signals
num_iter = 100;
max_rank = 10;
% max_rank = 25; % For the supplement
[all_U1, all_acf, all_nnz] = ...
    sparse_residual_analysis_max_acf(my_model_base, num_iter, 1:max_rank);
% Register the lines across rank
[registered_lines, registered_names] = ...
    sparse_residual_line_registration(all_U1, all_acf, my_model_base);

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
all_figs{2} = figure('DefaultAxesFontSize', 16);
[~, sort_ind] = sort(final_xy(:,2), 'descend');
imagesc(all_acf{rank_to_plot}(sort_ind, :)); colorbar

this_sparsity = all_nnz{rank_to_plot} ./ numel(all_U1{rank_to_plot});
tick_ind = round(linspace(1, 0.8*length(this_sparsity), 5));
xticks(tick_ind)
xticklabels(round(this_sparsity(tick_ind),2))
% imagesc(all_acf{rank_to_plot});colorbar
yticks(1:rank_to_plot)
yticklabels(final_names(sort_ind))
xlabel('Fraction of nonzero entries')
title('Determination of sparsity')

%---------------------------------------------
%% PLOT3: Visual connection with expert signals
%---------------------------------------------
% Note: replot the "correct" signals
tspan = 100:1000; % decided by hand, SAME AS ABOVE FIGURE
% ctr = my_model_base.control_signal;
my_model_simple = my_model_base;
my_model_simple.set_simple_labels();
my_model_simple.remove_all_control();
my_model_simple.calc_all_control_signals();
ctr = my_model_simple.control_signal;

%---------------------------------------------
% Reversal figure
%---------------------------------------------
all_figs{3} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(4,:), 'LineWidth', 2};

subplot(2,1,1)
% plot(ctr(5,tspan)+ctr(6,tspan), opt{:})
plot(ctr(4,tspan), opt{:})
title('Reversal')
ylabel('Expert')

subplot(2,1,2)
registration_ind = find(contains(text_names, 'AVA'), 1);
this_line = registered_lines{registration_ind};
[~, rank_ind] = max(this_line{:, 'this_y'}); % Max over ranks AND sparsity
line_ind = this_line{rank_ind, 'which_line_in_rank'};
rank_ind = this_line{rank_ind, 'which_rank'};
learned_u_REV = all_U1{rank_ind}(line_ind, :);
plot(learned_u_REV(tspan), opt{:})
xlim([0 length(tspan)])
ylabel('Learned')

prep_figure_no_box_no_zoom(all_figs{3})

%---------------------------------------------
% Dorsal turn figure
%---------------------------------------------
all_figs{4} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(2,:), 'LineWidth', 2};

subplot(2,1,1)
% plot(ctr(3,tspan), opt{:})
plot(ctr(2,tspan), opt{:})
title('Dorsal Turn')

subplot(2,1,2)
registration_ind = find(contains(text_names, 'SMDD'), 1);
this_line = registered_lines{registration_ind};
[~, rank_ind] = max(this_line{:, 'this_y'}); % Max over ranks AND sparsity
line_ind = this_line{rank_ind, 'which_line_in_rank'};
rank_ind = this_line{rank_ind, 'which_rank'};
learned_u_DT = all_U1{rank_ind}(line_ind, :);
plot(learned_u_DT(tspan), opt{:})
xlim([0 length(tspan)])

prep_figure_no_box_no_zoom(all_figs{4})

%---------------------------------------------
% Ventral turn figure
%---------------------------------------------
all_figs{5} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(1,:), 'LineWidth', 2};

subplot(2,1,1)
% plot(ctr(4,tspan), opt{:})
plot(ctr(3,tspan), opt{:})
title('Ventral Turn')

subplot(2,1,2)
registration_ind = find(contains(text_names, 'SMDV'), 1);
this_line = registered_lines{registration_ind};
[~, rank_ind] = max(this_line{:, 'this_y'}); % Max over ranks AND sparsity
line_ind = this_line{rank_ind, 'which_line_in_rank'};
rank_ind = this_line{rank_ind, 'which_rank'};
learned_u_VT = all_U1{rank_ind}(line_ind, :);
plot(learned_u_VT(tspan), opt{:})
xlim([0 length(tspan)])

prep_figure_no_box_no_zoom(all_figs{5})

%---------------------------------------------
% Forward figure
%---------------------------------------------
all_figs{6} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(3,:), 'LineWidth', 2};

subplot(2,1,1)
plot(ctr(1,tspan), opt{:})
title('Forward')

subplot(2,1,2)
registration_ind = find(contains(text_names, {'RIB','AVB'}), 1);
this_line = registered_lines{registration_ind};
[~, rank_ind] = max(this_line{:, 'this_y'}); % Max over ranks AND sparsity
line_ind = this_line{rank_ind, 'which_line_in_rank'};
rank_ind = this_line{rank_ind, 'which_rank'};
learned_u_FWD = all_U1{rank_ind}(line_ind, :);
plot(learned_u_FWD(tspan), opt{:})
xlim([0 length(tspan)])

prep_figure_no_box_no_zoom(all_figs{6})

%---------------------------------------------
%% PLOT4: LONG RUNTIME; produce data for boxplots
% Boxplot of correlation with expert signals
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
f_smooth = @(x) smoothdata(x, 'gaussian', 3);
% For the residual analysis
settings = struct('num_iter', 100, 'r_ctr', 15);

% Get the control signals and acf
for i = 1:num_files
    dat_struct = importdata(all_filenames{i});
    dat_struct.traces = f_smooth(dat_struct.traces);
    warning('USING PREPROCESSED DATA')
    
%     if i > num_type_1 % Differently named states
%         settings_base.global_signal_subset = ...
%             {'Reversal', 'Dorsal turn', 'Ventral turn'};
%     end
    % First get a baseline model as a preprocessor
    all_models{i} = CElegansModel(dat_struct, settings_base);
%     n = all_models{i}.dat_sz(1);
%     m = all_models{i}.dat_sz(2);
    
    % Build the sparse signals
%     [U, A, B] = sparse_residual_analysis(all_models{i}, settings);
%     all_U2{i} = zeros(i, m-1);
%     % Choose sparsity based on max acf
%     all_acf2{i} = zeros(i, num_iter);
%     which_sparsity = zeros(num_iter, 1);
%     for i2 = 1:settings.r_ctr
%         for i3 = 1:settings.num_iter
%             dat = U{i3}(i2,:)';
%             all_acf2{i}(i2, i3) = acf(dat, 1, false);
%         end
%         [~, which_sparsity] = max(all_acf2{i}(i2, :));
%         all_U2{i}(i2,:) = U{which_sparsity}(i2,:);
%     end
    [all_U, all_acf] = sparse_residual_analysis_max_acf(all_models{i}, ...
        settings.num_iter, settings.r_ctr);
    all_U2{i} = all_U{1};
    all_acf2{i} = all_acf{1};
end
%% Actually plot
% Get the maximum correlation with the experimentalist signals for each
% model
correlation_table_raw = ...
    connect_learned_and_expert_signals(all_U2, all_models);

% Consolidate differently named states; keep only the max
correlation_table = correlation_table_raw;
% name_dict = containers.Map(...
%     {'REVSUS','REV1','REV2','Reversal', 'DT', 'Dorsal turn',...
%         'VT', 'Ventral turn'},...
%     {'Reversal', 'Reversal', 'Reversal', 'Reversal', ...
%         'Dorsal Turn', 'Dorsal Turn', ...
%         'Ventral Turn', 'Ventral Turn'});
% TODO: refactor using model.set_simple_states()
name_dict = containers.Map(...
    {'REVSUS','REV1','REV2','Reversal', 'DT', 'Dorsal turn', 'Dorsal Turn',...
        'VT', 'Ventral turn', 'Ventral Turn', 'FWD', 'SLOW', 'Forward'},...
    {'Reversal', 'Reversal', 'Reversal', 'Reversal', ...
        'Dorsal Turn', 'Dorsal Turn', 'Dorsal Turn', ...
        'Ventral Turn', 'Ventral Turn', 'Ventral Turn', ...
        'Forward', 'Forward', 'Forward'});
to_keep_ind = true(size(correlation_table_raw,1),1);
for i = 1:size(correlation_table_raw,1)
    this_signal = correlation_table{i,'experimental_signal_name'}{1};
    try
        correlation_table{i,'experimental_signal_name'} = ...
            {name_dict(this_signal)};
    catch
        to_keep_ind(i) = false;
        fprintf('Failed to add signal %s; probably not an error\n', ...
            this_signal)
    end
end
correlation_table = correlation_table(to_keep_ind,:);
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
all_figs{7} = figure('DefaultAxesFontSize', 14);
h = boxplot(correlation_table{:,'maximum_correlation'}, ...
    correlation_table{:, 'experimental_signal_name'},...
    'GroupOrder', {'Reversal', 'Dorsal Turn', 'Ventral Turn', 'Forward'});
% set(h,{'linew'},{2})
% xtickangle(30)
xticklabels({'Rev', 'DT', 'VT', 'Fwd'})
ylabel('Correlation')
ylim([0 1])
title('Comparison across 15 datasets')
%% Save figures and data
if to_save
    writetable(correlation_table, ...
        [dat_foldername 'correlation_table']);
    
    for i = 1:length(all_figs)
        this_fig = all_figs{i};
        if ~isvalid(this_fig) || isempty(this_fig)
            continue
        end
        if i>=3 && i<7
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
% First: Learned signals model
%---------------------------------------------
% TODO
settings = settings_ideal;
settings.augment_data = 0;
settings.global_signal_mode = 'None';
% Depends on if the full code for figure 2 was run
if exist('all_U2', 'var')
    which_signals = 1:5; % TODO
    this_ctr = [smoothdata(5*all_U2{5}(which_signals,:),2,'movmean',2),...
        zeros(length(which_signals),1)];
elseif exist('all_U1', 'var')
    which_signals = 1:5; % TODO
    this_ctr = [smoothdata(5*all_U1{5}(which_signals,:),2,'movmean',2),...
        zeros(length(which_signals),1)];
end
settings.custom_control_signal = this_ctr; % TO CHECK
my_model_fig3_a = CElegansModel(filename_ideal, settings);

% 3d pca plot
my_model_fig3_a.set_simple_labels();
all_figs{2} = my_model_fig3_a.plot_colored_reconstruction(my_cmap_figure3);
view(my_viewpoint)
legend off
xticks([])
xlabel('')
yticks([])
ylabel('')
title('Unsupervised')

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
%     title(neur_labels{i})
    title('')
    xlim(tspan);
    set(gca, 'box', 'off')
    if i == 1
        ylabel('Calcium amplitude')
    else
        ylabel('')
    end
    xlabel('Time')
    xticks([])
    yticks([])
    legend off
end

%---------------------------------------------
% Second: experimentalist signals model
%---------------------------------------------
settings = settings_ideal;
settings.augment_data = 0;
my_model_fig3_b = CElegansModel(filename_ideal, settings);

% 3d pca plot
my_model_fig3_b.set_simple_labels();
all_figs{3} = my_model_fig3_b.plot_colored_reconstruction(my_cmap_figure3);
view(my_viewpoint)
legend off
xticks([])
xlabel('')
yticks([])
ylabel('')
title('Supervised')

% Reconstruct some individual neurons
neur_labels = {'AVAL', 'SMDDL'};
neur_id = [my_model_fig3_b.name2ind(neur_labels{1}), ...
    my_model_fig3_b.name2ind(neur_labels{2})];
fig_dict = containers.Map(...
    {neur_id(1), neur_id(2)},...
    {10, 11});
for i = 1:length(neur_id)
    this_n = neur_id(i);
    all_figs{fig_dict(this_n)} = ...
        my_model_fig3_b.plot_reconstruction_interactive(false,this_n);
    title(neur_labels{i})
%     title('')
    xlim(tspan);
    set(gca, 'box', 'off')
    if i == 1
        ylabel('Calcium amplitude')
    else
        ylabel('')
    end
    xlabel('')
    xticks([])
    yticks([])
    legend off
end

%---------------------------------------------
% Simplest comparison: no control at all
%---------------------------------------------
settings = settings_ideal;
settings.augment_data = 0;
settings.global_signal_mode = 'None';
settings.dmd_mode = 'tdmd';
my_model_fig3_c = CElegansModel(filename_ideal, settings);

% Original data; same for all models
my_model_fig3_c.set_simple_labels();
new_labels_key = my_model_fig3_c.state_labels_key;
all_figs{1} = my_model_fig3_c.plot_colored_data(false, 'plot', my_cmap_figure3);
view(my_viewpoint)
legend off
xticks([])
xlabel('1st PCA mode')
yticks([])
ylabel('2nd PCA mode')
title('Data')

all_figs{4} = my_model_fig3_c.plot_colored_reconstruction(my_cmap_figure3);
view(my_viewpoint)
legend off
xticks([])
xlabel('')
yticks([])
ylabel('')
title('No control')
% Now make the colormap match the bar graphs
% for i=1:length(new_labels_key)
%     all_figs{4}.Children(2).Children(i).CData = ...
%         my_cmap_3d(my_cmap_dict(i),:);
% end

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
%             ax = this_fig.Children(2);
%             ax.Clipping = 'Off';
%             prep_figure_no_axis(this_fig)
            prep_figure_tight_axes(this_fig);
            warning('PAUSE: DO THE ZOOM MANUALLY FOR PCA PLOTS')
%             pause
%             zoom(1.14)
%             if i==1
%                 zoom(1.05)
%             end
            sz = {'0.9\columnwidth', '0.14\paperheight'}
            matlab2tikz('figurehandle',this_fig,'filename',...
                [fname '_raw.tex'], ...
                'width', sz{1}, 'height', sz{2});
        end
        if i >= 5 % Histograms and Single neurons
            if i >=8 %Single neurons
                prep_figure_tight_axes(this_fig);
%                 prep_figure_no_axis(this_fig)
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

%% Figure 3d: Improvement for unsupervised control signals
% First, need to re-run the code for getting all of the signals
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
    [all_U, all_acf] = sparse_residual_analysis_max_acf(all_models{i}, ...
        settings.num_iter, settings.r_ctr);
    all_U2{i} = all_U{1};
    all_acf2{i} = all_acf{1};
end
%% Register the lines, and take the one that contains the proper neuron name
% with the highest acf to be comparable to the experimental one
all_lines = cell(num_files, 1);
all_names = cell(num_files, 1);
all_learned_signals = cell(num_files, 1);
dict_learned_to_experimental = {...
    {{'AVA'}, {'REV'}},...
    {{'AVB', 'RIB'}, {'FWD'}},...
    {{'SMDV'}, {'VT'}},...
    {{'SMDD'}, {'DT'}} };
for i = 1:num_files
    [all_lines{i}, all_names{i}] = sparse_residual_line_registration(...
        all_U2(i), all_acf2(i), all_models{i});
    this_learned_signal = struct();
    for i2 = 1:length(dict_learned_to_experimental)
        % Get name overlap, then the max acf if more than one
        exp_name = dict_learned_to_experimental{i2}{1};
        this_line_ind = find( cellfun(...
            @(x)any(contains(x,exp_name)), all_names{i}) );
        this_signal_name = dict_learned_to_experimental{i2}{2}{:};
        if ~isempty(this_line_ind)
            this_dat = vertcat(all_lines{i}{this_line_ind});
            [~, sub_ind] = max(this_dat{:,'this_y'});
            this_line_ind = this_line_ind(sub_ind);
            % Save it
            this_line = all_lines{i}(this_line_ind);
            U_ind = this_line{1}{1,'which_line_in_rank'};
            warning('NEED TO CHECK THIS OFFSET')
            this_learned_signal.(this_signal_name) = [0 all_U2{i}(U_ind, :)];
        else
            this_learned_signal.(this_signal_name) = [];
        end
    end
    all_learned_signals{i} = this_learned_signal;
end
%% With the control signals, now build the actual analysis models
all_filenames = get_Zimmer_filenames();
settings_base = struct(...
    'add_constant_signal',false,...
    'augment_data', 0,...
    'filter_window_dat', 1,...
    ...'autocorrelation_noise_threshold', 0.3,...
    'dmd_mode','func_DMDc',...
    'global_signal_mode', 'None',...
    'to_add_stimulus_signal', false);
settings_vec = cell(length(all_filenames), 1);
all_settings_loop = { {},...
    {'REV'}, {'FWD'}, {'REV', 'DT', 'VT'},...
    {'FWD', 'DT', 'VT'}, {'REV', 'DT', 'VT', 'FWD'} };
all_settings_loop_names = {'No_control', 'Forward', 'Reversal',...
    'Rev_and_turns', 'Fwd_and_turns', 'All'};

all_models_table = [];
for iInput = 1:length(all_settings_loop)
    fprintf('Analyzing models with control type: %s\n',...
        all_settings_loop_names{iInput});
    % Set up the vector of settings structs for each filename type
    these_settings = all_settings_loop{iInput};
    for i = 1:length(all_filenames)
        settings_vec{i} = settings_base;
        this_ctr = [];
        for i2 = 1:length(these_settings)
            this_ctr = [this_ctr; ...
                all_learned_signals{i}.(these_settings{i2})];
        end
        settings_vec{i}.custom_control_signal = this_ctr;
    end
    % Then get the actual models
    [obj_vec] = initialize_multiple_models(...
        all_filenames, @CElegansModel, settings_vec);
    
    if isempty(all_models_table)
        all_models_table = cell2table(obj_vec);
        all_models_table.Properties.VariableNames = all_settings_loop_names(1);
    else
        all_models_table = [all_models_table cell2table(obj_vec)];
        all_models_table.Properties.VariableNames(end) = ...
            all_settings_loop_names(iInput);
    end
end

[final_dat] = calculate_correlations_and_differences(all_models_table,...
    all_settings_loop_names);
%% Finally plot
all_figs = cell(2,1);
% Do not include the "other" group of neurons
color_order = [4 3 2 1];
cmap = my_cmap_3d(color_order, :);
opt = {'Color', 'k', 'LineStyle', '--', 'HandleVisibility','off'};

% all_figs{1} = boxplot_on_table(final_dat(1:3,1:end-1), [], cmap, 'rows');
% title('Correlation with Data')
% ylim([-1 1])
% line([-0.5 15], [0 0], opt{:});
% legend off
% 
% all_figs{2} = boxplot_on_table(final_dat(4:end,1:end-1), [], cmap, 'rows');
% title('Improvement via additional signals')
% ylim([-1 1])
% yticklabels('')
% line([-0.5 15], [0 0], opt{:});
% legend off

% New simpler version
all_figs{1} = boxplot_on_table(final_dat(1:2,1:end-1), [], cmap, 'rows');
title('Baselines')
ylim([-1 1])
line([-0.5 15], [0 0], opt{:});
legend off
ylabel('Correlation')

all_figs{2} = boxplot_on_table(final_dat(4:6,1:end-1), [], cmap, 'rows');
title('Improvement via additional signals')
ylim([-1 1])
yticklabels('')
line([-0.5 15], [0 0], opt{:});
legend off
%% Save data and figures
if to_save
    save([dat_foldername 'control_signals_learned_boxplot'], ...
        'final_dat');
    
    for i = 1:length(all_figs)
        this_fig = all_figs{i};
        if ~isvalid(this_fig) || isempty(this_fig)
            continue
        end
        figure(this_fig)
        sz = {'0.9\columnwidth', '0.1\paperheight'};
            
        fname = sprintf('%sfigure_3a_new_%d', foldername, i);
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================

%% Figure 3e: Improvement for experimentalists control signals
% First build models
all_filenames = get_Zimmer_filenames();
settings_base = struct(...
    'add_constant_signal',false,...
    'augment_data', 0,...
    'filter_window_dat', 1,...
    ...'autocorrelation_noise_threshold', 0.3,...
    'dmd_mode','func_DMDc',...
    'global_signal_mode', 'ID_binary_transitions',...
    'to_add_stimulus_signal', false);

settings_vec = cell(length(all_filenames), 1);
all_settings_loop = { {{}, {}},...
    {{'REV'}, {'Reversal'}},...
    {{'FWD', 'SLOW'}, {'Forward'}},...
    {{'REV', 'DT', 'VT'}, ...
        {'Reversal', 'Dorsal turn', 'Ventral turn'}},...
    {{'FWD', 'SLOW', 'DT', 'VT'}, ...
        {'Forward', 'Dorsal turn', 'Ventral turn'}},...
    {{'REV', 'DT', 'VT', 'FWD', 'SLOW'}, ...
        {'Reversal', 'Dorsal turn', 'Ventral turn', 'Forward'}} };
all_settings_loop_names = {'No_control', 'Forward', 'Reversal',...
    'Rev_and_turns', 'Fwd_and_turns', 'All'};

all_models_table = [];
for iInput = 1:length(all_settings_loop)
    fprintf('Analyzing models with control type: %s\n',...
        all_settings_loop_names{iInput});
    % Set up the vector of settings structs for each filename type
    these_settings = all_settings_loop{iInput};
    settings_type1 = these_settings{1};
    settings_type2 = these_settings{2};
    for i = 1:length(all_filenames)
        settings_vec{i} = settings_base;
        if i <= num_type_1
            settings_vec{i}.global_signal_subset = settings_type1;
        else
            settings_vec{i}.global_signal_subset = settings_type2;
        end
    end
    
    % Then get the actual models
    [obj_vec] = initialize_multiple_models(...
        all_filenames, @CElegansModel, settings_vec);
    
    if isempty(all_models_table)
        all_models_table = cell2table(obj_vec);
        all_models_table.Properties.VariableNames = all_settings_loop_names(1);
    else
        all_models_table = [all_models_table cell2table(obj_vec)];
        all_models_table.Properties.VariableNames(end) = ...
            all_settings_loop_names(iInput);
    end
end
% Calculate the correlations for each model and the clusters
sz = size(all_models_table);
all_dat_table = cell(sz);
baseline_dat_table = cell(sz(1),1);
dat_func = @calc_correlation_matrix;
baseline_func = @calc_linear_correlation;
for i = 1:sz(1)
    fprintf('Analyzing filename: %d\n', i);
    for i2 = 1:sz(2)
        all_dat_table{i,i2} = dat_func(all_models_table{i,i2});
    end
    baseline_dat_table{i} = baseline_func(all_models_table{i,1});
end
all_dat_table = cell2table([baseline_dat_table all_dat_table]);
all_dat_table.Properties.VariableNames = ...
    ['Linear_correlation' all_settings_loop_names];

% Build comparable clusters
%   Note: Each row of the model table has the same neurons and dataset
interpretable_clusters = { ...
    {'AVA', 'RIM', 'AIB', 'RME', 'VA', 'DA',},...
    {'AVB', 'RIB', 'RME'},...
    {'SMDD'},...
    {'SMDV', 'RIV'} ,...
    {'ASK', 'OLQ', 'URY', 'AVF', 'RIS', 'IL'}};
interpretable_clusters_names = {...
    'Reversal', 'Forward', 'Dorsal_turn', 'Ventral_turn', 'Other'};
num_clusters = length(interpretable_clusters);
all_ind_table = cell(sz(1),num_clusters);
for i = 1:sz(1)
    this_model = all_models_table{i,1};
    for i2 = 1:num_clusters
        all_ind_table{i, i2} = ...
            this_model.name2ind(interpretable_clusters{i2});
    end
end
all_ind_table = cell2table(all_ind_table);
all_ind_table.Properties.VariableNames = interpretable_clusters_names;
% Now calculate the data differences by cluster
all_differences = {...
    'Linear_correlation',...
    'No_control',...
    'Rev_and_turns',...
    {'Reversal', 'Linear_correlation'},...
    {'Rev_and_turns', 'Reversal'},...
    {'All', 'Rev_and_turns'} };
all_differences_names = {...
    'Linear', 'No_Control', 'Full',...
    'Reversal', 'Turn', 'Forward'};
num_differences = length(all_differences);
clust_dat_cell = cell(num_differences, 1);
sz = size(all_ind_table);
for iDiff = 1:num_differences
    this_dat = cell(sz);
    if iscell(all_differences{iDiff})
        n1 = all_differences{iDiff}{1};
        n2 = all_differences{iDiff}{2};
    else
        n1 = all_differences{iDiff};
        n2 = [];
    end
    for iFile = 1:sz(1)
        dat1 = all_dat_table{iFile, n1}{1};
        if isempty(n2)
            dat2 = zeros(size(dat1));
        else
            dat2 = all_dat_table{iFile, n2}{1};
        end
        for iClust = 1:sz(2)
            ind = all_ind_table{iFile, iClust}{1};
            this_dat{iFile, iClust} = dat1(ind) - dat2(ind);
        end
    end
    this_dat = cell2table(this_dat);
    this_dat.Properties.VariableNames = interpretable_clusters_names;
    clust_dat_cell{iDiff} = this_dat;
end

% Then combine across individuals
final_dat_cell = cell(num_differences, num_clusters);
for i = 1:num_differences
    for i2 = 1:num_clusters
        final_dat_cell{i, i2} = real(vertcat(clust_dat_cell{i}{:,i2}{:}));
    end
end
final_dat = cell2table(final_dat_cell);
final_dat.Properties.VariableNames = interpretable_clusters_names;
final_dat.Properties.RowNames = all_differences_names;
%% Finally, plot each difference by cluster
all_figs = cell(2,1);
% Do not include the "other" group of neurons
color_order = [4 3 2 1];
cmap = my_cmap_3d(color_order, :);
opt = {'Color', 'k', 'LineStyle', '--', 'HandleVisibility','off'};

% all_figs{1} = boxplot_on_table(final_dat(1:3,1:end-1), [], cmap, 'rows');
% title('Correlation with Data')
% ylim([-1 1])
% line([-0.5 15], [0 0], opt{:});
% legend off
% 

% New: Only plot the comparison between these full models
learned_dat_name = [dat_foldername 'control_signals_learned_boxplot.mat'];
assert(logical(exist(learned_dat_name, 'file')),...
    'Must have collected data from the unsupervised models and saved the data');
learned_dat = importdata(learned_dat_name);
learned_dat.Properties.RowNames{3} = 'Learned';
comparison_table = [final_dat(3, 1:end-1); learned_dat(3, 1:end-1)];
comparison_table.Properties.RowNames{1} = 'Expert';

all_figs{1} = boxplot_on_table(comparison_table, [], cmap, 'rows');
title('Full models')
ylim([-1 1])
yticklabels('')
line([-0.5 15], [0 0], opt{:});
legend off

all_figs{2} = boxplot_on_table(final_dat(4:end,1:end-1), [], cmap, 'rows');
title('Improvement via additional signals')
ylim([-1 1])
yticklabels('')
line([-0.5 15], [0 0], opt{:});
legend off

%% Save data and plots
if to_save
%     save([dat_foldername 'control_signals_expert_boxplot'], ...
%         'final_dat', 'all_models_table', 'all_ind_table', ...
%         'baseline_dat_table');
    save([dat_foldername 'control_signals_expert_boxplot'], ...
        'final_dat', 'comparison_table');
    
    for i = 1:length(all_figs)
        this_fig = all_figs{i};
        if ~isvalid(this_fig) || isempty(this_fig)
            continue
        end
        figure(this_fig)
        sz = {'0.9\columnwidth', '0.1\paperheight'};
            
        fname = sprintf('%sfigure_3_new_%d', foldername, i);
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Figure 4: Elimination path (Lasso)
all_figs = cell(5,1);
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
which_single_model = 3;

for i = 1:max_iter
    fprintf('Iteration %d...\n', i)
    % Remove the top neurons from the last round
    if i > 1
        [~, top_ind] = max(abs(B_prime_lasso_td_3d(:, :, i-1)), [], 2);
        % We'll get a single time slice of a neuron, but want to remove all
        % copies (cumulatively)
        top_ind = mod(top_ind,n) + n*(0:settings.augment_data-1);
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
all_figs{1} = figure('DefaultAxesFontSize', 16);
i = 3; % Dorsal Turn
max_plot_iter = 15;
plot(all_fp(i,1:max_plot_iter) / num_spikes(i), 'LineWidth',2)
hold on
plot(all_fn(i,1:max_plot_iter) / num_spikes(i), 'LineWidth',2)
legend({'False Positives', 'False Negatives'},'Location','northwest')

which_examples = [1 5];
for i2 = 1:length(which_examples)
    x = which_examples(i2);
%     scatter(x, all_fn(i, x), 'HandleVisibility','off', ...
%         'LineWidth', 3,'MarkerEdgeColor', 'k')
    plot(x, all_fn(i, x), 'ok', 'HandleVisibility','off', ...
        'MarkerSize', 5,'LineWidth', 2)
end
xlim([1 max_plot_iter])

% Correlation version of error
% plot(all_err(i,:), 'LineWidth',2)
% legend({'False Positives', 'False Negatives', 'Correlation'},'Location','northwest')

a = arrayfun(@(x)my_model_time_delay.get_names(x,true),...
    elimination_neurons(:,2:end), 'UniformOutput',false);
eliminated_names = cellfun(@(x)x{1},a,'UniformOutput',false);

set(gca, 'box', 'off')
xlim([1 max_plot_iter])
xticks(1:max_plot_iter)
xticklabels(['All neurons', eliminated_names(i,1:max_plot_iter)])
% xticklabels(['All neurons', eliminated_names(1:max_plot_iter)])
xtickangle(60)
% xlabel('Eliminated neuron (cumulative)')
ylabel('Fraction of events')
title('Elimination Path for Dorsal Turn')
% disp(eliminated_names)

%---------------------------------------------
%% Plot the data for an early iteration

% Plot the unrolled matrix of one control signal
%---------------------------------------------
% Settings for both iteration plots
which_ctr = 3;
num_neurons_to_plot = 10;
opt = {'DefaultAxesFontSize', 10};
tspan = [0 1000];

% First plot
which_iter = which_examples(1);
names = my_model_time_delay.get_names([], true);
% ind = contains(my_model_time_delay.state_labels_key, {'REV', 'DT', 'VT'});
% Unroll one of the controllers
unroll_sz = [my_model_time_delay.original_sz(1), settings.augment_data];
dat_unroll1 = reshape(B_prime_lasso_td_3d(which_ctr,:,which_iter), unroll_sz);
fig4_normalization = max(max(abs(dat_unroll1))); % To use later as well

all_figs{2} = figure(opt{:});
[ordered_dat, ordered_ind] = top_ind_then_sort(dat_unroll1, num_neurons_to_plot);
imagesc(ordered_dat./fig4_normalization)
fig4_cmap = cmap_white_zero(ordered_dat);
fig4_caxis = caxis;
colormap(fig4_cmap); % i.e. equal to the first plot
colorbar
title('Predictors (All Neurons)')
yticks(1:unroll_sz(1))
yticklabels(names(ordered_ind))
xlabel('Delay frames')

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
[~, ~, ~, ~, ~, ~, all_figs{3}] = ...
    calc_false_detection(ctr, ctr_reconstruct_td,...
            all_thresholds_3d(which_ctr, which_iter), [], [], true, true, false);
% title(sprintf('%d Neurons Eliminated', which_iter-1))
title('Reconstruction (All Neurons)')
set(gca, 'box', 'off')
set(gca, 'FontSize', opt{2})
legend off
yticklabels('')
xlabel('Time')
xlim(tspan)

%---------------------------------------------
%% Plot the data for a later iteration

% Plot the unrolled matrix of one control signal
%---------------------------------------------
which_iter = which_examples(2);
names = my_model_time_delay.get_names([], true);
% ind = contains(my_model_time_delay.state_labels_key, {'REV', 'DT', 'VT'});
% Unroll one of the controllers
unroll_sz = [my_model_time_delay.original_sz(1), settings.augment_data];
dat_unroll1 = reshape(B_prime_lasso_td_3d(which_ctr,:,which_iter), unroll_sz);

all_figs{4} = figure(opt{:});
[ordered_dat, ordered_ind] = top_ind_then_sort(dat_unroll1, num_neurons_to_plot);
imagesc(ordered_dat./fig4_normalization)
% colormap(cmap_white_zero(ordered_dat));
colormap(fig4_cmap); % i.e. equal to the first plot
caxis(fig4_caxis)
title(sprintf('(%d Neurons Eliminated)', which_iter-1))
% title('Predictors for Dorsal Turn control signal')
yticks(1:unroll_sz(1))
yticklabels(names(ordered_ind))
xlabel('Delay frames')

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
[~, ~, ~, ~, ~, ~, all_figs{5}] = ...
    calc_false_detection(ctr, ctr_reconstruct_td,...
            all_thresholds_3d(which_ctr, which_iter), [], [], true, true, false);
% title(sprintf('Reconstruction of control signal %d for iteration %d', ...
%     which_ctr, which_iter))
title(sprintf('(%d Neurons Eliminated)', which_iter-1))
set(gca, 'box', 'off')
set(gca, 'FontSize', opt{2})
legend off
yticklabels('')
% ylabel('Arbitrary units')
% xticklabels('')
xlabel('Time')
xlim(tspan)

%---------------------------------------------
%% Save figures and data
%---------------------------------------------
if to_save
    save([dat_foldername 'elimination_path'], ...
        'all_intercepts_td', 'all_err', 'all_fp', 'all_fn', ...
        'all_thresholds_3d', 'elimination_neurons', 'num_spikes');
    
    for i = 1:length(all_figs)
        if isempty(all_figs{i}) || ~isvalid(all_figs{i})
            continue;
        end
        fname = sprintf('%sfigure_5_%d', foldername, i);
        this_fig = all_figs{i};
        if i==1 % Elimination path
            sz = {'0.9\columnwidth', '0.09\paperheight'};
        elseif i==2 || i==4 % Variable selection
            sz = {'0.9\columnwidth', '0.08\paperheight'};
        elseif i==3 || i==5 % Reconstructions
            sz = {'0.9\columnwidth', '0.05\paperheight'};
        end
        matlab2tikz('figurehandle',this_fig,'filename',...
            [fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================

%% Figure 4d: Plot elimination path for all datasets
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

%---------------------------------------------
% Get elimination paths
%---------------------------------------------
for i = 1:n
    fprintf('Processing file %d\n', i)
    
    dat_struct = importdata(all_filenames{i});
    dat_struct.traces = dat_struct.traces ...
        ./ vecnorm(dat_struct.traces, 2, 1);
    warning('Test for normalized data')
    
    settings_model = settings_ideal;
    settings_model.augment_data = 6;
    settings_model.dmd_mode = 'no_dynamics';
    settings_model.global_signal_mode = 'ID_binary_transitions';
    settings_model.to_add_stimulus_signal = false;
    if i > num_type_1 % Differently named states
        settings_model.global_signal_subset = ...
            {'Reversal', 'Dorsal turn', 'Ventral turn'};
    end
    my_model_time_delay = CElegansModel(dat_struct, settings_model);
    
    s = struct('to_calculate_false_detection', true, ...
        'which_error_metric', 'mcc', 'max_iter', 3);
    [B_prime_lasso_td_3d, all_intercepts_td, elimination_neurons, ...
        eliminated_names, which_ctr, ...
        all_err, all_fp, all_fn, num_spikes] = ...
        sparse_encoding_analysis(my_model_time_delay, s);
    
    sz = size(B_prime_lasso_td_3d);
    B_prime_lasso_td_3d = {reshape(...
        B_prime_lasso_td_3d(which_ctr,:,:), sz(2:3))};
    all_intercepts_td = all_intercepts_td(which_ctr,:);
    elimination_neurons = elimination_neurons(which_ctr,:);
    eliminated_names = eliminated_names(which_ctr,:);
    all_mcc = {all_err};
    false_positives = {all_fp};
    false_negatives = {all_fn};
    number_of_spikes = {num_spikes};
    tmp_table = table(eliminated_names, elimination_neurons,...
            all_intercepts_td, B_prime_lasso_td_3d, which_ctr, ...
            all_mcc, false_positives, false_negatives, number_of_spikes);
    if i == 1
        elimination_table = tmp_table;
    else
        elimination_table = [elimination_table; tmp_table];
    end
end

%---------------------------------------------
%% Get histogram data for each neuron
%---------------------------------------------
all_names = elimination_table{:, 'eliminated_names'};
max_name_length = 5;
to_combine_LR = true;
unique_names = unique(all_names);
unique_names = unique_names( cellfun(@isempty, ...
    regexp(unique_names, '^\d{1,3}')) ); % Get rid of numbers
for i = 1:length(unique_names) % Only keep first 5 letters
    this_name = unique_names{i};
    if length(this_name) > max_name_length
        unique_names{i} = unique_names{i}(1:max_name_length);
    end
    if to_combine_LR && length(this_name) > 3 ...
            && endsWith(this_name, {'L', 'R'})
        unique_names{i} = unique_names{i}(1:end-1);
    end
end
unique_names = unique(unique_names);

neuron_counts = zeros(length(unique_names), s.max_iter);
registered_dat = table(neuron_counts, 'RowNames', unique_names); 

for i = 1:length(unique_names)
    this_name = unique_names{i};
    for i2 = 1:s.max_iter-1
        this_iter_names = all_names(:,i2);
        f = @(x) startsWith(x, this_name); % Because we shortened names
        these_counts = sum( cellfun(f, this_iter_names) );
        
        registered_dat{this_name,1}(i2) = these_counts;
    end
end
all_counts = sum(registered_dat{:,1}, 2);
x = min([10, s.max_iter]);
all_counts_early_enough = sum(registered_dat{:,1}(:,1:x), 2);
registered_dat = [registered_dat ...
    table(all_counts) table(all_counts_early_enough)];

%---------------------------------------------
%% Plot histogram data for each neuron
%---------------------------------------------
all_figs = cell(1,1);

max_rank_to_plot = 15;

all_figs{1} = figure('DefaultAxesFontSize', 16);
threshold_dat = registered_dat{:, 'all_counts'};
% [~, to_plot_ind] = maxk(threshold_dat, 4);
% to_plot_ind = threshold_dat > quantile(threshold_dat, 0.75);
sorted_thresholds = sort(unique(threshold_dat), 'descend');
to_plot_ind = threshold_dat > sorted_thresholds(5);
bar(registered_dat{to_plot_ind, 'neuron_counts'}', 'stacked')
legend(unique_names(to_plot_ind))
xlabel('Elimination Iteration')
ylabel('Occurences across individuals')
%==========================================================================
%% Make a table for error and removal information
%---------------------------------------------

% Initialize
max_removals_in_table = 3;
all_mcc = elimination_table{:, 'all_mcc'};
count_threshold = 2;
neuron_ind = find(...
    registered_dat{:, 'all_counts_early_enough'}>count_threshold);
neuron_names = registered_dat.Properties.RowNames(neuron_ind);
which_signal = 1;
Rank = (1:max_removals_in_table)';
table_to_export = table(Rank);

% Add column for error
err_col = {};
for iRank = 1:max_removals_in_table
    err = vertcat(cellfun(@(x) x(which_signal,iRank), all_mcc));
    err_col = [err_col; ...
        {sprintf('%.2f (%.2f)', mean(err), std(err))}];
end
table_to_export = [table_to_export, ...
    table(err_col, 'VariableNames', {'Error'})];

% Add rows for neuron occurences
neuron_table = table();
ind = 1:max_removals_in_table;
for iNeur = 1:length(neuron_names)
    cumulative_counts = cumsum(...
        registered_dat{:, 'neuron_counts'}(neuron_ind(iNeur), ind), 2)';
    
    neuron_table = [neuron_table ...
        table(cumulative_counts, 'VariableNames', neuron_names(iNeur))];
end
table_to_export = [table_to_export, neuron_table];
%==========================================================================
%% Make a table for error and removal information (old)
%---------------------------------------------

max_removals_in_table = 3;
all_mcc = elimination_table{:, 'all_mcc'};
neuron_names = registered_dat.Properties.RowNames;
which_signal = 1;
table_to_export = table();
for iIter = 1:max_removals_in_table
    [counts, neur_ind] = max(registered_dat{:, 'neuron_counts'}(:, iIter));
    neuron_str = neuron_names{neur_ind};
    combined_neuron_str = sprintf('%s (%d)', neuron_str, counts);
    this_err = cellfun(@(x)x(which_signal, iIter), all_mcc);
    combined_err_str = sprintf('%.2f (%.2f)',...
        mean(this_err), std(this_err));
    tmp_table = table({combined_neuron_str; combined_err_str});
    tmp_table.Properties.VariableNames = {'Dorsal_Turn'};
    table_to_export = [table_to_export; tmp_table];
end
%==========================================================================

%% Figure 4a-b version 2: Variable selection (old)
% Figure 4a-b version 2: Variable selection
all_figs = cell(4,1);
% Get the 'ideal' (time delayed) model
settings = settings_ideal;
settings.augment_data = 6;
my_model_time_delay = CElegansModel(filename_ideal, settings);

%---------------------------------------------
% LASSO from a direct fit to data
%---------------------------------------------
U2 = my_model_time_delay.control_signal(:,2:end);
X1 = my_model_time_delay.dat(:,1:end-1);
disp('Fitting lasso models...')
all_intercepts_td1 = zeros(size(U2,1),1);
B_prime_lasso_td1 = zeros(size(U2,1), size(X1,1));
which_fit = 8;
for i = 1:size(U2,1)
    [all_fits, fit_info] = lasso(X1', U2(i,:), 'NumLambda',10);
    all_intercepts_td1(i) = fit_info.Intercept(which_fit);
    B_prime_lasso_td1(i,:) = all_fits(:,which_fit); % Which fit = determined by eye
end

%---------------------------------------------
% Plot the unrolled matrix of one control signal (Dorsal Turn)
%---------------------------------------------
% which_ctr = 1;
which_ctr = 3;
names = my_model_time_delay.get_names([], true);
% ind = contains(my_model_time_delay.state_labels_key, {'REV', 'DT', 'VT'});
% Unroll one of the controllers
unroll_sz = [my_model_time_delay.original_sz(1), settings.augment_data];
dat_unroll1 = reshape(B_prime_lasso_td1(which_ctr,:), unroll_sz);
fig4_normalization = max(max(abs(dat_unroll1))); % To use later as well
dat_unroll1 = dat_unroll1./fig4_normalization;

all_figs{1} = figure('DefaultAxesFontSize', 10);
% [ordered_dat, ordered_ind] = top_ind_then_sort(dat_unroll1, [], 1e-2);
[ordered_dat, ordered_ind] = top_ind_then_sort(dat_unroll1, 10);
imagesc(ordered_dat)
fig4_cmap = cmap_white_zero(ordered_dat);
fig4_caxis = caxis;
colormap(fig4_cmap);
% colorbar
% title(sprintf('All predictors for control signal %d', which_ctr))
% title('Predictors for Dorsal Turn (all neurons)')
% title('Predictors for Dorsal Turn control signal')
title('Top 10 Selected Neurons')
yticks(1:unroll_sz(1))
yticklabels(names(ordered_ind))
xlabel('Number of delay frames')
colorbar();

%---------------------------------------------
% Plot reconstructions of some control signals
%---------------------------------------------
% Get the reconstructions
X1 = my_model_time_delay.dat(:,1:end-1);
ctr_reconstruct = [my_model_time_delay.control_signal(which_ctr,1)...
    B_prime_lasso_td1(which_ctr,:) * X1];
ctr_reconstruct_td = ctr_reconstruct + all_intercepts_td1(which_ctr);
ctr = my_model_time_delay.control_signal(which_ctr,:);

% tspan = 100:1000;
% ctr_reconstruct_td = ctr_reconstruct_td(tspan);
% ctr = ctr(tspan);

% Get the threshold
f = @(x) minimize_false_detection(ctr, ...
    ctr_reconstruct_td, x, 0.1);
this_threshold = fminsearch(f, 1.5);

% Plot
[~, ~, ~, ~, ~, ~, all_figs{2}] = ...
    calc_false_detection(ctr, ctr_reconstruct_td,...
            this_threshold, [], [], true, true, false);
% title('Sparse Reconstruction (all neurons)')
title('Signal Reconstruction')
set(gca, 'box', 'off')
set(gca,'FontSize',10)
l = get(gca, 'Legend');
l.NumColumns = 2;
legend off
yticklabels('')
ylabel('Arbitrary units')
% xticklabels('')
xlabel('Time')
% title(sprintf('Sparse reconstruction of control signal %d',which_ctr))

%---------------------------------------------
% Save figures: part a
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
        if i==2
            sz = {'0.9\columnwidth', '0.07\paperheight'};
        else
            sz = {'0.9\columnwidth', '0.12\paperheight'};
        end
        matlab2tikz('figurehandle',this_fig,'filename',...
            [fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================
