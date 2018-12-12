


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

close all
%==========================================================================

%% Define 'ideal' settings
filename_ideal = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_ideal = importdata(filename_ideal);
num_neurons = size(dat_ideal.traces,2);

settings_ideal = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'augment_data', 9,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','func_DMDc',...
    'global_signal_subset', {{'DT','VT','REV'}},...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings_ideal.global_signal_mode = 'ID_binary_transitions';

%==========================================================================




%% Figure 1: Intro to transition signals
all_figs = cell(5,1);
%---------------------------------------------
% First build a model with good parameters
%---------------------------------------------
% Use CElegans model to preprocess
my_model_fig1 = CElegansModel(filename_ideal, settings_ideal);

%---------------------------------------------
% Second get a representative neuron and control kicks
%---------------------------------------------
%   tspans decided by hand
neuron = 'AVAL';
% tspan = 300:550;
tspan = 100:1000;

% Plot
all_figs{1} = my_model_fig1.plot_reconstruction_interactive(false, 'AVAL');
xlim([tspan(1) tspan(end)])
xlabel('')
xticks([])
yticks([])
ylabel('')
set(gca, 'box', 'off')
title(sprintf('Reconstruction of %s', neuron))
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
            sz = {'0.9\columnwidth', '0.1\paperheight'}
        else
            sz = {'0.9\columnwidth', '0.025\paperheight'}
        end
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
%         prep_figure_no_axis(this_fig)
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Figure 2: Explanation of time-delay embedding

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

%---------------------------------------------
% Get time-delayed model (best)
%---------------------------------------------
my_model_fig3_a = CElegansModel(filename_ideal, settings_ideal);

% Original data; same for all models
my_model_fig3_a.set_simple_labels();
new_labels_key = my_model_fig3_a.state_labels_key;
all_figs{1} = my_model_fig3_a.plot_colored_data(false, 'o');
for i=1:length(new_labels_key)
    all_figs{1}.Children(2).Children(i).CData = ...
        my_cmap_3d(my_cmap_dict(i),:);
end
view(my_viewpoint)

% 3d pca plot
all_figs{2} = my_model_fig3_a.plot_colored_reconstruction();
view(my_viewpoint)
for i=1:length(new_labels_key)
    all_figs{2}.Children(2).Children(i).CData = ...
        my_cmap_3d(my_cmap_dict(i),:);
end

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

% Correlation histogram
all_figs{5} = my_model_fig3_a.plot_correlation_histogram();
xlim([-0.2, 1])
ylim([0 15])
title('')
legend off
set(gca, 'box', 'off')

%---------------------------------------------
% Compare with no time delay
%---------------------------------------------
settings = settings_ideal;
settings.augment_data = 0;
my_model_fig3_b = CElegansModel(filename_ideal, settings);

my_model_fig3_b.set_simple_labels();
all_figs{3} = my_model_fig3_b.plot_colored_reconstruction();
view(my_viewpoint)
% Now make the colormap match the bar graphs
for i=1:length(new_labels_key)
    all_figs{3}.Children(2).Children(i).CData = ...
        my_cmap_3d(my_cmap_dict(i),:);
end

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

% Correlation histogram
all_figs{6} = my_model_fig3_b.plot_correlation_histogram();
xlim([-0.2, 1])
ylim([0 15])
title('')
legend off
set(gca, 'box', 'off')

%---------------------------------------------
% Simplest comparison: no control at all!
%---------------------------------------------
settings = settings_ideal;
settings.augment_data = 0;
settings.global_signal_mode = 'None';
settings.dmd_mode = 'tdmd';
my_model_fig3_c = CElegansModel(filename_ideal, settings);

my_model_fig3_c.set_simple_labels();
all_figs{4} = my_model_fig3_c.plot_colored_reconstruction();
view(my_viewpoint)
% Now make the colormap match the bar graphs
for i=1:length(new_labels_key)
    all_figs{4}.Children(2).Children(i).CData = ...
        my_cmap_3d(my_cmap_dict(i),:);
end

% Correlation histogram
all_figs{7} = my_model_fig3_c.plot_correlation_histogram();
xlim([-0.2, 1])
ylim([0 15])
title('')
legend off
set(gca, 'box', 'off')

%---------------------------------------------
% Save figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = sprintf('%sfigure_4_%d', foldername, i);
        this_fig = all_figs{i};
        if i <= 4 % 3d PCA plots
            ax = this_fig.Children(2);
            ax.Clipping = 'Off';
            prep_figure_no_axis(this_fig)
            zoom(1.17)
        end
        if i >= 5 % Histograms and Single neurons
            if i >=8 %Single neurons
                prep_figure_no_axis(this_fig)
            end
            sz = {'0.9\columnwidth', '0.1\paperheight'}
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
legend({'Time Delay', 'Single Step', 'No Control'})

% Also plot some circles for the individual neuron reconstructions
% Use variable from above
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
% Plot the unrolled matrix of one control signal
%---------------------------------------------
which_ctr = 1;
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
title('Predictors for Dorsal Turn control signal')
yticks(1:unroll_sz(1))
yticklabels(names(ordered_ind))
xlabel('Number of delay frames')

%---------------------------------------------
% Plot reconstructions of some control signals
%---------------------------------------------
tspan = 100:1000;
% Get the reconstructions
X1 = my_model_time_delay.dat(:,1:end-1);
ctr_reconstruct = [my_model_time_delay.control_signal(which_ctr,1)...
    B_prime_lasso_td(which_ctr,:) * X1];
ctr_reconstruct_td = ctr_reconstruct + all_intercepts_td(which_ctr);

% Plot
all_figs{2} = figure('DefaultAxesFontSize', 14);
ctr = my_model_time_delay.control_signal(which_ctr,:);
plot(ctr(tspan))
hold on
plot(ctr_reconstruct_td(tspan), 'Linewidth',2)
xlim([0 length(tspan)])
% title(sprintf('Sparse reconstruction of control signal %d',which_ctr))
title('Sparse Reconstruction')
legend({'Data','Time-delay'})

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
        if i >= 2
            prep_figure_no_axis(this_fig)
        end
        sz = {'0.9\columnwidth', '0.1\paperheight'};
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
max_iter = 5;
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
all_thresholds = zeros(num_ctr, max_iter);
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
        all_thresholds(i2, i) = fminsearch(f, 1);
        [all_fp(i2,i), all_fn(i2,i), num_spikes(i)] = ...
            calc_false_detection(ctr(i2,:), ctr_reconstruct_td(i2,:),...
            all_thresholds(i2, i));
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
legend({'False Positives', 'False Negatives'})

a = arrayfun(@(x)my_model_time_delay.get_names(x,true),...
    elimination_neurons(:,2:end), 'UniformOutput',false);
eliminated_names = cellfun(@(x)x{1},a,'UniformOutput',false);

set(gca, 'box', 'off')
xlim([1 max_iter])
xticks(1:max_iter)
xticklabels(['All neurons', eliminated_names(i,:)])
xtickangle(60)
xlabel('Eliminated neuron (cumulative)')
ylabel('Percentage of total spikes')
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
% title(sprintf('All predictors for control signal %d', which_ctr))
title('All predictors for Dorsal Turn control signal')
yticks(1:unroll_sz(1))
yticklabels(names(ordered_ind))
xlabel('Number of delay frames')

%---------------------------------------------
% Plot a reconstruction
%---------------------------------------------
tspan = 100:1000;
% Get the reconstructions
X1 = my_model_time_delay.dat(:,1:end-1);
ctr_reconstruct = [my_model_time_delay.control_signal(which_ctr,1)...
    B_prime_lasso_td_3d(which_ctr,:,which_iter) * X1];
ctr_reconstruct_td = ctr_reconstruct + ...
    all_intercepts_td(which_ctr, which_iter);

% Plot
ctr = my_model_time_delay.control_signal(which_ctr,:);
% all_figs{3} = figure('DefaultAxesFontSize', 16);
% ctr = my_model_time_delay.control_signal(which_ctr,:);
% plot(ctr(tspan))
% hold on
% plot(ctr_reconstruct_td(tspan), 'Linewidth',2)
% title(sprintf('Reconstruction of control signal %d for iteration %d', ...
%     which_ctr, which_iter))
% legend({'Data','Time-delay'})

% Test
[~, ~, ~, ~, ~, all_figs{3}] = ...
    calc_false_detection(ctr, ctr_reconstruct_td,...
            all_thresholds(which_ctr, which_iter), [], [], true);
title(sprintf('Reconstruction of control signal %d for iteration %d', ...
    which_ctr, which_iter))

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
        sz = {'0.9\columnwidth', '0.1\paperheight'};
        matlab2tikz('figurehandle',this_fig,'filename',...
            [fname '_raw.tex'], ...
            'width', sz{1}, 'height', sz{2});
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================

