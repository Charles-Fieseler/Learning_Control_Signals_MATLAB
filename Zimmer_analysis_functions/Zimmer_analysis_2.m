

%% Trying to get AIC to learn control signals, i.e. figure 3
%% SETUP
all_figs = cell(7,1);

dat_struct = dat_ideal;
% f_smooth = @(x) smoothdata(x, 'gaussian', 3);
% dat_struct.traces = f_smooth(dat_struct.traces);
% warning('USING PREPROCESSED DATA')

% First get a baseline model as a preprocessor
settings_base = settings_ideal;
settings_base.augment_data = 0;
settings_base.autocorrelation_noise_threshold = 0.5;
my_model_base = CElegansModel(dat_struct, settings_base);
n = my_model_base.dat_sz(1);
m = my_model_base.dat_sz(2);

% Loop through sparsity and rank to get a set of control signals
num_iter = 100;
max_rank = 5;
% max_rank = 25; % For the supplement
% [all_U1, all_acf, all_nnz] = ...
%     sparse_residual_analysis_max_over_iter(my_model_base, num_iter, 1:max_rank);
% Try aic instead
[all_U1, all_acf, all_nnz] = ...
    sparse_residual_analysis_max_over_iter(my_model_base,...
    num_iter, 1:max_rank, 'aic');
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
% Define text_names
%---------------------------------------------
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
%     if length(find(these_ranks_ind)) > 1
%         plot(xy{these_ranks_ind,'which_rank'}, xy{these_ranks_ind,'this_y'},...
%             'LineWidth',2, line_opt{:})
%     else
%         plot(xy{these_ranks_ind,'which_rank'}, xy{these_ranks_ind,'this_y'},...
%             'o', line_opt{:})
%     end
end
% textfit(text_xy(:,1), text_xy(:,2), text_names, text_opt{:})
% ylim([0, 1])
% ylabel('Maximum autocorrelation')
% xlabel('Rank')
% title('Determination of rank truncation')

%% New: plot the aic for each rank
figure;
hold on
for i = 1:length(all_acf)
    plot(50:length(all_acf{1}), all_acf{i}(1,50:end));
%     plot(all_acf{i}(1,:));
end
legend()
title('AIC values for each rank')
%---------------------------------------------

%% For new plot 2: get full path of signals
disp('Getting full path of for a single rank...')
rank_to_plot = 5;
settings = struct('r_ctr', rank_to_plot, 'num_iter', num_iter);    

[all_U_path, all_A, all_B, ~] = ...
    sparse_residual_analysis(my_model_base, settings);
% [all_U_path, all_A, all_B, ~] = ...
%     sparse_residual_analysis(all_models{2}, settings); % ONLY WORKS IF THE LOWER SECTION IS RUN
%% Calculate AIC and cross-validation, and plot
disp('Calculating AIC for a single rank...')
% num_steps = 10:10:50;
% num_steps = [1, 5, 10, 20];
% num_steps = [20, 50, 100];
num_steps = 200;
all_aic = zeros(length(all_U_path), length(num_steps));
all_crossval = zeros(size(all_aic));
all_nnz_path = zeros(size(all_aic));
do_aicc = false;
% formula_mode = 'stanford';
formula_mode = 'stanford2';
% formula_mode = 'standard';
% formula_mode = 'one_step_stanford'; % test for different dof formula
to_calc_crossval = true;
to_calc_aic = false;
X = my_model_base.dat;
% X = all_models{2}.dat;
for i = 1:length(all_U_path)-1
    U = all_U_path{i};
    A = all_A{i};
    B = all_B{i};
    for i2 = 1:length(num_steps)
        if to_calc_aic
            all_aic(i, i2) = ...
                -aic_2step_dmdc(X, U, A, B, num_steps(i2), do_aicc, formula_mode);
        end
        if to_calc_crossval
            all_crossval(i, i2) = dmdc_cross_val(X, U, 5, num_steps(i2));
        end
        all_nnz_path(i) = nnz(all_U_path{i});
    end
end
all_aic(end, :) = all_aic(end-1, :); % TODO
if to_calc_crossval
    all_crossval(end, :) = all_crossval(end-1, :); % TODO
end

figure;
ind = 10:size(all_aic,1);
subplot(2,1,1)
plot(ind, all_aic(ind, :))
title(sprintf('%s AIC values (error at %d steps)', ...
    formula_mode, num_steps))
subplot(2,1,2)
plot(ind, all_crossval(ind, :));
title('Cross-validation errors')

% MAX OF NEGATIVE 
[~, aic_min_ind] = max(all_aic);
fprintf('Minimum for aic: %d\n', aic_min_ind)

%% New plot 2: path of REV
%---------------------------------------------
% rank_to_plot = 4;
tspan = 1:3000; % decided by hand, SAME AS ABOVE FIGURE
all_figs{2} = figure('DefaultAxesFontSize', 16);

% Plot 5 example traces, with acf titles
% registration_ind = find(contains(text_names, 'AVA'), 1);
registration_ind = find(contains(text_names, 'RIM'), 1);
this_line = registered_lines{registration_ind};
rank_to_plot_table = find(this_line{:, 'which_rank'} == rank_to_plot);
if isempty(rank_to_plot_table)
    error('Signal not found in path matrix; rerun sparse_residual_analysis')
end
rev_ind = this_line{rank_to_plot_table, 'which_line_in_rank'};
% rev_ind = 10;

rev_aic = all_acf{rank_to_plot}(rev_ind, :);
rev_sparsity = all_nnz{rank_to_plot} ./ numel(all_U1{rank_to_plot});

opt = {'color', my_cmap_3d(4,:), 'LineWidth', 2}; % Match paper colors

% Plot 1: initialization
this_iter = 1;
subplot(1,5,1)
plot(all_U_path{this_iter}(rev_ind, tspan), opt{:})
title(sprintf('%d; acf=%.2f',...
    this_iter, rev_aic(this_iter)))
xlim([0 length(tspan)])

% Plot 2: Some sparsity
this_iter = 25;
subplot(1,5,2)
plot(all_U_path{this_iter}(rev_ind, tspan), opt{:})
title(sprintf('%d; acf=%.2f',...
    this_iter, rev_aic(this_iter)))
xlim([0 length(tspan)])

% Plot 3: Some more sparsity
this_iter = 50;
subplot(1,5,3)
plot(all_U_path{this_iter}(rev_ind, tspan), opt{:})
title(sprintf('%d; acf=%.2f',...
    this_iter, rev_aic(this_iter)))
xlim([0 length(tspan)])

% Plot 4: Even more sparsity (max acf?)
this_iter = aic_min_ind;
% this_iter = 75;
subplot(1,5,4)
plot(all_U_path{this_iter}(rev_ind, tspan), opt{:})
title(sprintf('%d; acf=%.2f=max',...
    this_iter, rev_aic(this_iter)))
xlim([0 length(tspan)])

% Plot 5: Too much sparsity (lower acf)
this_iter = 100;
subplot(1,5,5)
plot(all_U_path{this_iter}(rev_ind, tspan), opt{:})
title(sprintf('%d; acf=%.2f',...
    this_iter, rev_aic(this_iter)))
xlim([0 length(tspan)])

% Note: use rank_to_plot instead of rank_ind for now
% learned_u_REV = all_U1{rank_to_plot}(rev_ind, :);
% plot(learned_u_REV(tspan), opt{:})
% xlim([0 length(tspan)])
% 
%---------------------------------------------

%% New plot: falloff time if the signal is not included

% First, set up the control signals
which_signals = 1:5; % TODO
this_ctr = [smoothdata(5*all_U1{5}(which_signals,:),2,'movmean',2),...
    zeros(length(which_signals),1)];
% this_ctr = [all_U1{5}(which_signals,:), zeros(length(which_signals), 1)];

% Second, set up two alternate control signals
settings1 = settings_ideal;
settings1.augment_data = 0;
settings1.global_signal_mode = 'None';
settings1.custom_control_signal = this_ctr;
settings1.AdaptiveDmdc_settings = struct('truncation_rank', -1);
my_model_learned1 = CElegansModel(dat_struct, settings1);

% TODO: for now, just a no-control model
settings2.augment_data = 0;
settings2.global_signal_mode = 'None';
settings2.dmd_mode = 'tdmd';
my_model_learned2 = CElegansModel(dat_struct, settings2);

%---------------------------------------------
%% Calculate the falloff times
%---------------------------------------------
% First get indices for each of the control signal blocks
i = 2;
[all_starts, all_ends] = calc_contiguous_blocks(...
    logical(this_ctr(i, :)), 1, 3);

max_step = 50;
ind_to_plot = 45;
% ind_to_plot = 0;
tol = [];
func = @(m, x0, t) m.AdaptiveDmdc_obj.calc_reconstruction_control(x0, t);
dat = my_model_learned1.dat;
all_falloff_times = zeros(length(all_starts)-1, 2);
% for i2 = 1:length(all_starts)-1
for i2 = 2:2
    this_len = all_ends(i2)-all_starts(i2);
    opt = {func, max_step + this_len, tol, ind_to_plot};
    ind = all_starts(i2):all_ends(i2)+max_step;
%     this_dat = dat(:, ind);
    
    if ind_to_plot > 0
        figure; subplot(4,1,1)
    end
    [all_falloff_times(i2,1), err_vec1] = calc_falloff_time(...
        my_model_learned1, ind, opt{:});
    if ind_to_plot > 0
        title('Learned signals')
        subplot(4,1,2)
        plot(this_ctr(i, ind)')
        title('Control signal')
    
        subplot(4,1,3)
    end
    
    [all_falloff_times(i2,2), err_vec2] = calc_falloff_time(...
        my_model_learned2, ind,  opt{:});
    if ind_to_plot > 0
        title('No signals')

        subplot(4,1,4)
        plot(dat(ind_to_plot, :))
        hold on
        plot(ind(1), 0, 'o')
        plot(ind(end), 0, '*')

        pause
    end
end

figure;
histogram(all_falloff_times(:,1))
hold on
histogram(all_falloff_times(:,2))
legend('With control signals', 'No control signals')
%% Calculate a "real-ness score" for each control signal element
which_iter = 1;
% this_ctr = [all_U_path{which_iter}, zeros(length(which_signals),1)];
this_ctr = all_U_path{which_iter};
this_ctr = [this_ctr, zeros(size(this_ctr,1),1)];

settings1 = settings_ideal;
settings1.augment_data = 0;
settings1.global_signal_mode = 'None';
settings1.autocorrelation_noise_threshold = 0.7;
settings1.AdaptiveDmdc_settings = struct('truncation_rank', -1);

num_iter = 99;
all_U_test = cell(num_iter,1);
all_err_path = cell(num_iter, 1);
all_total_err = zeros(num_iter, 1);
all_corr = zeros(num_iter, 1);
all_improvement_path = cell(num_iter, 1);

new_update_rule = false;

for i = 1:num_iter
    all_U_test{i} = this_ctr;
    settings1.custom_control_signal = this_ctr;
    my_model_full = CElegansModel(dat_struct, settings1);

    % Calculate whether the control signal is real
    
    [all_err, out] = calc_control_realness(my_model_full);
    all_improvement = abs(all_err(:, 2)) - abs(all_err(:, 1));
    
    % Update control signal
    if new_update_rule
        if i==1
            thresh = 0;
        else
            thresh = quantile(all_improvement(all_improvement>0), 0.1);
        end
        this_ctr(:, all_improvement < thresh) = 0;
    else
        this_ctr = all_U_path{i+1};
        this_ctr = [this_ctr, zeros(size(this_ctr,1),1)]; %#ok<AGROW>
    end
    
    % Save for later analysis
    all_improvement_path{i} = all_improvement;
    all_err_path{i} = all_err;
    all_total_err(i) = ...
        my_model_full.AdaptiveDmdc_obj.calc_reconstruction_error();
    all_corr(i) = mean(my_model_full.calc_correlation_matrix());
end

figure;
% plot(all_err(:, 1))
% hold on
% plot(all_err(:, 2))
% legend('Errors for ctr', 'Errors for noise')
% plot(all_err(:, 1) - all_err(:, 2))
plot(all_improvement)
title('Improvement via addition on control signals')

figure;
plot(all_total_err)
title('Total reconstruction errors')

figure;
% plot(cellfun(@(x)log(sum(x(:,1))./sum(x(:,2))), all_err_path))
% plot(cellfun(@(x)sum(x.^2), all_improvement_path))
plot(cellfun(@sum, all_improvement_path))
title('Total improvements')

%% Calculate a model for control signals on flat portion of AIC
which_models = [50, 60, 70, 80, 90];
% which_models = 40:2:80;
all_models = cell(length(which_models),1);
all_err = zeros(size(all_models));

f_smooth = @(U) [smoothdata(5*U,2,'movmean',2),...
    zeros(length(which_signals),1)];
% f_smooth = @(U) [U, zeros(length(which_signals),1)];

% Second, set up two alternate control signals
settings1 = settings_ideal;
settings1.augment_data = 0;
settings1.global_signal_mode = 'None';
settings1.dmd_mode = 'sparse_fast';
settings1.autocorrelation_noise_threshold = 0.5;
settings1.AdaptiveDmdc_settings = struct('truncation_rank', -1);
to_plot = true;
for i = 1:length(which_models)
    settings1.custom_control_signal = f_smooth(all_U_path{which_models(i)});
    all_models{i} = CElegansModel(dat_struct, settings1);
    if to_plot
%         all_models{i}.plot_reconstruction_interactive(true, 45);
        all_models{i}.plot_colored_reconstruction();
        title(sprintf('Model for signal %d', which_models(i)))
        drawnow
    end
    all_err(i) = all_models{i}.AdaptiveDmdc_obj.calc_reconstruction_error();
end

figure
plot(which_models, all_err)
title('Full model reconstruction error')
xlabel('Path index for sparsity')

%% PLOT3: Visual connection with expert signals
%---------------------------------------------
% Note: replot the "correct" signals
tspan = 1:3000; % decided by hand, SAME AS ABOVE FIGURE
% ctr = my_model_base.control_signal;
my_model_simple = my_model_base;
% my_model_simple = all_models{2}; % ONLY WORKS IF THE LOWER SECTION IS RUN

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
%% PLOT4: Long-ish runtime; produce data for boxplots
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
% settings = struct('num_iter', 100, 'r_ctr', 15);
settings = struct('num_iter', 100, 'r_ctr', 5);

% Get the control signals and acf
for i = 1:num_files
    dat_struct = importdata(all_filenames{i});
%     dat_struct.traces = f_smooth(dat_struct.traces);
%     warning('USING PREPROCESSED DATA')
    % First get a baseline model as a preprocessor
    all_models{i} = CElegansModel(dat_struct, settings_base);
    all_models{i}.set_simple_labels();
    all_models{i}.remove_all_control();
    all_models{i}.calc_all_control_signals();
    [all_U, all_acf_tmp] = sparse_residual_analysis_max_over_iter(all_models{i}, ...
        settings.num_iter, settings.r_ctr, 'aic');
    all_U2{i} = all_U{1};
    all_acf2{i} = all_acf_tmp{1};
end
%% Test plot: compare learned and expert controllers

file_num = 1;
learn_ind = 3;
exp_ind = 2;
calc_false_detection(all_U2{file_num}(learn_ind,:),...
    all_models{file_num}.control_signal(exp_ind,:), [], [], [], true);
title(sprintf('learned index: %d; expert index: %d', ...
    learn_ind, exp_ind))
%% Actually plot
% Get the maximum correlation with the experimentalist signals for each
% model
f_dist = @(dat, recon) calc_f1_score(dat, recon);

correlation_table_raw = ...
    connect_learned_and_expert_signals(all_U2, all_models, f_dist);

% Consolidate differently named states; keep only the max
correlation_table = correlation_table_raw;
name_dict = containers.Map(...
    {'Simple Reverse', 'DT', 'Dorsal turn', 'Dorsal Turn',...
        'VT', 'Ventral turn', 'Ventral Turn', 'Simple Forward'},...
    {'Reversal', ...
        'Dorsal Turn', 'Dorsal Turn', 'Dorsal Turn', ...
        'Ventral Turn', 'Ventral Turn', 'Ventral Turn', ...
        'Forward'});
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
    'GroupOrder', ...
    {'Reversal', 'Dorsal Turn', 'Ventral Turn', 'Forward'});
% set(h,{'linew'},{2})
% xtickangle(30)
xticklabels({'Rev', 'DT', 'VT', 'Fwd'})
ylabel('Correlation')
ylim([0 1])
title('Comparison across 15 datasets')
%==========================================================================



%% Try postprocessing algorithm of control signals

[all_filenames, num_type_1] = get_Zimmer_filenames();

% Preprocess to get rid of some 'noise neurons'
filename_ideal = all_filenames{5};
settings_base = define_ideal_settings();
settings_base.augment_data = 0;
settings_base.autocorrelation_noise_threshold = 0.6;
my_model_base = CElegansModel(filename_ideal, settings_base);

this_dat = my_model_base.dat;

% Settings for control signals, then solve
i = 1;
fprintf('Getting control signals for dataset %d...\n', i)
sra_settings.r_ctr = 4;
sra_settings.verbose = false;
sra_settings.only_positive_U = true;
sra_settings.num_iter = 70;
[all_U, all_A, all_B] = sparse_residual_analysis(this_dat, sra_settings);

fprintf('Postprocessing control signals for dataset %d...\n', i)
post_settings = struct('verbose', 2, ...
    'safe_ind_thresh', 1.4, ...
    'improvement_threshold', 1.5,...
    'multiple_removal_thresh', 3,...
    'start_mode', 'acf',...
    'aic_mode', 'stanford');
[U_best, all_improvement, new_U] = postprocess_control_signals(this_dat, all_U, all_A, all_B,...
    post_settings);

%% Make a model with this controller
settings_learned = settings_base;
settings_learned.dmd_mode = 'naive';
settings_learned.global_signal_mode = 'None';
settings_learned.custom_control_signal = [U_best zeros(size(U_best,1),1)];
my_model_processed = CElegansModel(filename_ideal, settings_learned);

settings_learned.custom_control_signal = [new_U{1} zeros(size(U_best,1),1)];
my_model_initial = CElegansModel(filename_ideal, settings_learned);

% Plot some example neurons in both models
neur = 'AVAL';
my_model_processed.plot_reconstruction_interactive(true, neur);
title('Postprocessed signal')
my_model_initial.plot_reconstruction_interactive(true, neur);
title('Unprocessed signal')
%---------------------------------------------
%% Super simple post-processing: remove isolated spikes
%---------------------------------------------
U0 = new_U{1};
for i = 1:size(U0,1)
    [all_starts, all_ends] = ...
        calc_contiguous_blocks(logical(U0(i,:)), 1, 1);
    blocks_to_remove = (all_ends - all_starts) < 1;
    for i2 = 1:length(all_starts)
        if blocks_to_remove(i2)
            U0(i, all_starts(i2):all_ends(i2)) = 0;
        end
    end
end

settings_learned.custom_control_signal = [U0 zeros(size(U_best,1),1)];
my_model_isolated = CElegansModel(filename_ideal, settings_learned);

my_model_isolated.plot_reconstruction_interactive(true, 1);

%==========================================================================



%% Going back to ACF for figure 3, but with post-processing
%% SETUP
all_figs = cell(7,1);

dat_struct = dat_ideal;
% f_smooth = @(x) smoothdata(x, 'gaussian', 3);
% dat_struct.traces = f_smooth(dat_struct.traces);
% warning('USING PREPROCESSED DATA')

% First get a baseline model as a preprocessor
settings_base = settings_ideal;
settings_base.augment_data = 0;
settings_base.autocorrelation_noise_threshold = 0.5;
my_model_base = CElegansModel(dat_struct, settings_base);
n = my_model_base.dat_sz(1);
m = my_model_base.dat_sz(2);

% Loop through sparsity and rank to get a set of control signals
num_iter = 100;
max_rank = 5;
% max_rank = 25; % For the supplement
% [all_U1, all_acf, all_nnz] = ...
%     sparse_residual_analysis_max_over_iter(my_model_base, num_iter, 1:max_rank);
% Try aic instead
[all_U1, all_acf, all_nnz] = ...
    sparse_residual_analysis_max_over_iter(my_model_base,...
    num_iter, 1:max_rank, 'acf');
for i = 1:length(all_U1)
    all_U1{i} = remove_isolated_spikes(all_U1{i});
end
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
% Define text_names
%---------------------------------------------
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
%     if length(find(these_ranks_ind)) > 1
%         plot(xy{these_ranks_ind,'which_rank'}, xy{these_ranks_ind,'this_y'},...
%             'LineWidth',2, line_opt{:})
%     else
%         plot(xy{these_ranks_ind,'which_rank'}, xy{these_ranks_ind,'this_y'},...
%             'o', line_opt{:})
%     end
end
% textfit(text_xy(:,1), text_xy(:,2), text_names, text_opt{:})
% ylim([0, 1])
% ylabel('Maximum autocorrelation')
% xlabel('Rank')
% title('Determination of rank truncation')

%% New: plot the aic for each rank
fig = figure;
hold on
start_ind = 10;
for i = 1:length(all_acf)
    plot(start_ind:length(all_acf{1}), all_acf{i}(1,start_ind:end));
end
legend()
title('AIC values for each rank (Note: only first signal)')

% Different plot with all lines
[fig] = plot_cell_array(...
    cellfun(@transpose,all_acf, 'UniformOutput', false), 3);
title('acf over all ranks for all signals, for all sparsities')
%---------------------------------------------

%% For new plot 2: get full path of signals
disp('Getting full path of for a single rank...')
rank_to_plot = 5;
settings = struct('r_ctr', rank_to_plot, 'num_iter', num_iter);    

[all_U_path, all_A, all_B, ~] = ...
    sparse_residual_analysis(my_model_base, settings);
for i = 1:length(all_U_path)
    all_U_path{i} = remove_isolated_spikes(all_U_path{i});
end
% [all_U_path, all_A, all_B, ~] = ...
%     sparse_residual_analysis(all_models{2}, settings); % ONLY WORKS IF THE LOWER SECTION IS RUN

%% New plot 2: path of REV
%---------------------------------------------
% rank_to_plot = 4;
tspan = 100:1000; % decided by hand, SAME AS ABOVE FIGURE
% tspan = 1:3000; % decided by hand, SAME AS ABOVE FIGURE
all_figs{2} = figure('DefaultAxesFontSize', 16);

% Plot 5 example traces, with acf titles
registration_ind = find(contains(text_names, 'AVA'), 1);
% registration_ind = find(contains(text_names, 'SMDDL'), 1);
this_line = registered_lines{registration_ind};
rank_to_plot_table = find(this_line{:, 'which_rank'} == rank_to_plot);
if isempty(rank_to_plot_table)
    error('Signal not found in path matrix; rerun sparse_residual_analysis')
end
rev_ind = this_line{rank_to_plot_table, 'which_line_in_rank'};
% rev_ind = 10;

rev_aic = all_acf{rank_to_plot}(rev_ind, :);
rev_sparsity = all_nnz{rank_to_plot} ./ numel(all_U1{rank_to_plot});

opt = {'color', my_cmap_3d(4,:), 'LineWidth', 2}; % Match paper colors

% Plot 1: initialization
this_iter = 1;
subplot(1,5,1)
plot(all_U_path{this_iter}(rev_ind, tspan), opt{:})
title(sprintf('%d; acf=%.2f',...
    this_iter, rev_aic(this_iter)))
xlim([0 length(tspan)])

% Plot 2: Some sparsity
this_iter = 20;
subplot(1,5,2)
plot(all_U_path{this_iter}(rev_ind, tspan), opt{:})
title(sprintf('%d; acf=%.2f',...
    this_iter, rev_aic(this_iter)))
xlim([0 length(tspan)])

% Plot 3: Some more sparsity
this_iter = 40;
subplot(1,5,3)
plot(all_U_path{this_iter}(rev_ind, tspan), opt{:})
title(sprintf('%d; acf=%.2f',...
    this_iter, rev_aic(this_iter)))
xlim([0 length(tspan)])

% Plot 4: Even more sparsity (max acf?)
try
    this_iter = aic_min_ind;
catch
    try
        [~, this_iter] = max(all_acf{rank_to_plot}(rev_ind, :));
    catch
        this_iter = 75;
    end
end
U_best = all_U_path{this_iter};
subplot(1,5,4)
plot(all_U_path{this_iter}(rev_ind, tspan), opt{:})
title(sprintf('%d; acf=%.2f=max',...
    this_iter, rev_aic(this_iter)))
xlim([0 length(tspan)])

% Plot 5: Too much sparsity (lower acf)
this_iter = 80;
subplot(1,5,5)
plot(all_U_path{this_iter}(rev_ind, tspan), opt{:})
title(sprintf('%d; acf=%.2f',...
    this_iter, rev_aic(this_iter)))
xlim([0 length(tspan)])

% Note: use rank_to_plot instead of rank_ind for now
% learned_u_REV = all_U1{rank_to_plot}(rev_ind, :);
% plot(learned_u_REV(tspan), opt{:})
% xlim([0 length(tspan)])
% 
%---------------------------------------------
%% Eyeball test: build a model with the best signal

f_smooth = @(x) smoothdata(5*x,2,'movmean',2);
% f_smooth = @(x) x;

settings = settings_ideal;
settings.augment_data = 0;
settings.dmd_mode = 'naive';
settings.global_signal_mode = 'None';
settings.custom_control_signal = [f_smooth(U_best)...
    zeros(size(U_best, 1), 1)];
my_model_learned = CElegansModel(filename_ideal, settings);

my_model_learned.plot_reconstruction_interactive(true, 'AVAL');

%% PLOT3: Visual connection with expert signals
%---------------------------------------------
% Note: replot the "correct" signals
tspan = 100:1000; % decided by hand, SAME AS ABOVE FIGURE
% tspan = 1:3000; % decided by hand, SAME AS ABOVE FIGURE
% ctr = my_model_base.control_signal;
my_model_simple = my_model_base;
% my_model_simple = all_models{2}; % ONLY WORKS IF THE LOWER SECTION IS RUN

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
%% PLOT4: Long-ish runtime; produce data for boxplots
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
% settings = struct('num_iter', 100, 'r_ctr', 15);
settings = struct('num_iter', 100, 'r_ctr', 5);

% Get the control signals and acf
for i = 1:num_files
    dat_struct = importdata(all_filenames{i});
%     dat_struct.traces = f_smooth(dat_struct.traces);
%     warning('USING PREPROCESSED DATA')
    % First get a baseline model as a preprocessor
    all_models{i} = CElegansModel(dat_struct, settings_base);
    all_models{i}.set_simple_labels();
    all_models{i}.remove_all_control();
    all_models{i}.calc_all_control_signals();
    [all_U, all_acf_tmp] = sparse_residual_analysis_max_over_iter(all_models{i}, ...
        settings.num_iter, settings.r_ctr, 'acf');
    all_U2{i} = all_U{1};
    all_acf2{i} = all_acf_tmp{1};
end
all_U2 = cellfun(@remove_isolated_spikes, all_U2, 'UniformOutput', false);
%% Test plot: compare learned and expert controllers

file_num = 1;
learn_ind = 1;
exp_ind = 4;
exp_ctr = all_models{file_num}.control_signal(exp_ind,:);
learn_ctr = [all_U2{file_num}(learn_ind,:) 0];
[exp_ctr, learn_ctr] = alignsignals(exp_ctr, learn_ctr, 20, 'truncate');
this_corr = corrcoef(exp_ctr, learn_ctr);
calc_false_detection(exp_ctr, learn_ctr, [], [], [], true);
title(sprintf('learned index: %d; expert index: %d; correlation: %.2f', ...
    learn_ind, exp_ind, this_corr(2,1)))
%% Actually plot
% Get the maximum correlation with the experimentalist signals for each
% model
% f_dist = @(dat, recon) calc_f1_score(dat, recon);
f_dist = [];

correlation_table_raw = ...
    connect_learned_and_expert_signals(all_U2, all_models, f_dist);

% Consolidate differently named states; keep only the max
correlation_table = correlation_table_raw;
name_dict = containers.Map(...
    {'Simple Reverse', 'DT', 'Dorsal turn', 'Dorsal Turn',...
        'VT', 'Ventral turn', 'Ventral Turn', 'Simple Forward'},...
    {'Reversal', ...
        'Dorsal Turn', 'Dorsal Turn', 'Dorsal Turn', ...
        'Ventral Turn', 'Ventral Turn', 'Ventral Turn', ...
        'Forward'});
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
    'GroupOrder', ...
    {'Reversal', 'Dorsal Turn', 'Ventral Turn', 'Forward'});
% set(h,{'linew'},{2})
% xtickangle(30)
xticklabels({'Rev', 'DT', 'VT', 'Fwd'})
ylabel('Correlation')
ylim([0 1])
title('Comparison across 15 datasets')
%==========================================================================

