

%% Supplemental Figure 1: Outlier detection
all_figs = cell(3,1);

filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_fig2 = importdata(filename);

id_struct = struct(...
    'ID',{dat_fig2.ID},'ID2',{dat_fig2.ID2},'ID3',{dat_fig2.ID3});
settings = struct(...
    'to_plot_cutoff',true,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false,...
    'to_plot_A_matrix_svd',false,...
    'to_plot_data',false,...
    'id_struct',id_struct);
ad_obj_fig2 = AdaptiveDmdc(dat_fig2.traces.',settings);
all_figs{1} = gcf;

% Plot individual neurons
neurons = [2,46];
for i = 1:length(neurons)
    this_neuron = neurons(i);
    this_name = ad_obj_fig2.get_names(this_neuron);
    outlier_window = 2000;
    filter_window = 10;
    all_figs{i+1} = ad_obj_fig2.plot_data_and_filter(...
        ad_obj_fig2.neuron_errors(this_neuron ,:)',...
                        filter_window, outlier_window);
    title(sprintf('Residual for neuron %d (name=%s)',...
        this_neuron , this_name))
    xlabel('Time')
    ylabel('Error')
    xlim([0,3000])
end

% Save figures
for i = 1:length(all_figs)
    fname = sprintf('%sfigure_5_%d', foldername, i);
    this_fig = all_figs{i};
    set(this_fig, 'Position', get(0, 'Screensize'));
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Supplementary Figure 2: Neuron classifications
all_figs = cell(5,1);

filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'filter_window_dat',3,...
    'use_deriv',true,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary';
% settings.global_signal_mode = 'ID_binary_and_grad';

%---------------------------------------------
% Calculate 5 worms and get roles
%---------------------------------------------
all_models = cell(5,1);
all_roles_dynamics = cell(5,2);
all_roles_centroid = cell(5,2);
all_roles_global = cell(5,2);
for i=1:5
    filename = sprintf(filename_template,i);
    if i==4
        settings.lambda_sparse = 0.035; % Decided by looking at pareto front
    else
        settings.lambda_sparse = 0.05;
    end
    all_models{i} = CElegansModel(filename,settings);
end
max_err_percent = 0.2;
for i=1:5
    % Use the dynamic fixed point
    [all_roles_dynamics{i,1}, all_roles_dynamics{i,2}] = ...
        all_models{i}.calc_neuron_roles_in_transition(true, [], true);
    % Just use centroid of a behavior
    [all_roles_centroid{i,1}, all_roles_centroid{i,2}] = ...
        all_models{i}.calc_neuron_roles_in_transition(true, [], false);
    % Global mode actuation
    [all_roles_global{i,1}, all_roles_global{i,2}] = ...
        all_models{i}.calc_neuron_roles_in_global_modes(true, [], max_err_percent);
end

%---------------------------------------------
% Data preprocessing
%---------------------------------------------
[ combined_dat_dynamic, all_labels_dynamic ] =...
    combine_different_trials( all_roles_dynamics );
[ combined_dat_centroid, all_labels_centroid ] =...
    combine_different_trials( all_roles_centroid );
[ combined_dat_global, all_labels_global ] =...
    combine_different_trials( all_roles_global );

%---------------------------------------------
% Bar graph of transition kicks (dynamics, no worm 4)
%---------------------------------------------
possible_roles = unique(combined_dat_dynamic);
possible_roles(cellfun(@isempty,possible_roles)) = [];
num_neurons = size(combined_dat_dynamic,1);
these_worms = [1, 2, 3, 5];
role_counts = zeros(num_neurons,length(possible_roles));

for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_dynamic(:,these_worms), possible_roles{i}),2);
end
% Long names mean it was an ambiguous identification
name_lengths = cellfun(@(x) length(x)<6, all_labels_dynamic)';
this_ind = find((sum(role_counts,2)>3).*name_lengths);
all_figs{4} = figure('DefaultAxesFontSize',14);
b = bar(role_counts(this_ind,:), 'stacked');
for i=1:length(b)
    % Skip the first color (dark blue)
    b(i).FaceColor = my_cmap_3d(i,:);
end
legend(possible_roles)
yticks(1:max(role_counts))
ylabel('Number of times identified')
xticks(1:length(this_ind));
xticklabels(all_labels_dynamic(this_ind))
xtickangle(90)
yticks(1:length(possible_roles))
title('Neuron roles using similarity to attractors (no 4th worm)')

%---------------------------------------------
% Bar graph of transition kicks (dynamics, WITH worm 4)
%---------------------------------------------
possible_roles = unique(combined_dat_dynamic);
possible_roles(cellfun(@isempty,possible_roles)) = [];
num_neurons = size(combined_dat_dynamic,1);
these_worms = 1:5; %DIFFERENT
role_counts = zeros(num_neurons,length(possible_roles));

for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_dynamic(:,these_worms), possible_roles{i}),2);
end
% Long names mean it was an ambiguous identification
name_lengths = cellfun(@(x) length(x)<6, all_labels_dynamic)';
this_ind = find((sum(role_counts,2)>3).*name_lengths);
all_figs{1} = figure('DefaultAxesFontSize',14);
b = bar(role_counts(this_ind,:), 'stacked');
for i=1:length(b)
    % Skip the first color (dark blue)
    b(i).FaceColor = my_cmap_3d(i,:);
end
legend(possible_roles)
yticks(1:max(role_counts))
ylabel('Number of times identified')
xticks(1:length(this_ind));
xticklabels(all_labels_dynamic(this_ind))
xtickangle(90)
yticks(1:length(possible_roles))
title('Neuron roles using similarity to attractors')

%---------------------------------------------
% Bar graph of transition kicks (centroids, no worm 4)
%---------------------------------------------
these_worms = [1, 2, 3, 5];
role_counts = zeros(num_neurons,length(possible_roles));
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_centroid(:,these_worms), possible_roles{i}),2);
end
% Long names mean it was an ambiguous identification
name_lengths = cellfun(@(x) length(x)<6, all_labels_centroid)';
this_ind = find((sum(role_counts,2)>3).*name_lengths);
all_figs{5} = figure('DefaultAxesFontSize',14);
b = bar(role_counts(this_ind,:), 'stacked');
for i=1:length(b)
    % Skip the first color (dark blue)
    b(i).FaceColor = my_cmap_3d(i,:);
end
legend(possible_roles)
yticks(1:max(role_counts))
ylabel('Number of times identified')
xticks(1:length(this_ind));
xticklabels(all_labels_centroid(this_ind))
xtickangle(90)
yticks(1:length(possible_roles))
title('Neuron roles using similarity to centroids (no 4th worm)')

%---------------------------------------------
% Bar graph of transition kicks (centroids, WITH worm 4)
%---------------------------------------------
these_worms = 1:5;
role_counts = zeros(num_neurons,length(possible_roles));
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_centroid(:,these_worms), possible_roles{i}),2);
end
% Long names mean it was an ambiguous identification
name_lengths = cellfun(@(x) length(x)<6, all_labels_centroid)';
this_ind = find((sum(role_counts,2)>3).*name_lengths);
all_figs{2} = figure('DefaultAxesFontSize',14);
b = bar(role_counts(this_ind,:), 'stacked');
for i=1:length(b)
    % Skip the first color (dark blue)
    b(i).FaceColor = my_cmap_3d(i,:);
end
legend(possible_roles)
yticks(1:max(role_counts))
ylabel('Number of times identified')
xticks(1:length(this_ind));
xticklabels(all_labels_centroid(this_ind))
xtickangle(90)
yticks(1:length(possible_roles))
title('Neuron roles using similarity to centroids')

%---------------------------------------------
% Roles for global neurons
%---------------------------------------------
% First make the field names the same
d = containers.Map(...
    {'group 1', 'group 2', 'other', '', 'both', 'high error'},...
    {'simple REVSUS', 'simple FWD', 'other', '', 'both', 'z_error'});
combined_dat_global = cellfun(@(x) d(x), combined_dat_global,...
    'UniformOutput', false);
possible_roles = unique(combined_dat_global);
possible_roles(cellfun(@isempty,possible_roles)) = [];
role_counts = zeros(size(combined_dat_global,1),length(possible_roles));
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_global, possible_roles{i}),2);
end
% Long names mean it was an ambiguous identification
name_lengths = cellfun(@(x) length(x)<6, all_labels_global)';
this_ind = find((sum(role_counts,2)>3).*name_lengths);
all_figs{3} = figure('DefaultAxesFontSize',14);
b = bar(role_counts(this_ind,:), 'stacked');
for i=1:length(b)
    b(i).FaceColor = my_cmap_3d(i,:);
end
legend([possible_roles(1:end-1); {'high error'}])
yticks(1:max(max((role_counts))))
ylabel('Number of times identified')
xticks(1:length(this_ind));
xticklabels(all_labels_global(this_ind))
xtickangle(90)
title('Neuron roles using global mode activation')

%---------------------------------------------
% Save figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        fname = sprintf('%sfigure_s2_%d', foldername, i);
        this_fig = all_figs{i};
        if isempty(this_fig)
            continue
        end
        set(this_fig, 'Position', get(0, 'Screensize'));
        saveas(this_fig, fname, 'png');
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_w_error.tex']);
    end
end
%==========================================================================


%% Supplementary Figure 3: sparse lambda errors for all worms (ID signal)
warning('Will clear all variables; Press enter if this is okay')
pause
clear all


num_figures = 1;
all_figs = cell(num_figures,1);

filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';

% Overall settings... determines how long this will take!
which_worms = [1, 3, 4];
num_runs = 100;

% Settings for each object
model_settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_save_raw_data',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'add_constant_signal',false);
% global_signal_modes = {{'ID','ID_binary'}};
% global_signal_modes = {{'ID','ID_simple'}};
% global_signal_modes = {{'ID_binary', 'ID_binary_and_grad'}};
global_signal_modes = {{'ID_binary_and_grad'}};
iterate_setting = 'global_signal_mode';
% lambda_vec = linspace(0.0185, 0.08, 400);
lambda_vec = linspace(0.03, 0.08, num_runs);
% lambda_vec = linspace(0.02, 0.1, 20);
settings = struct(...
    'base_settings', model_settings,...
    'iterate_settings',struct(iterate_setting,global_signal_modes),...
    'x_vector', lambda_vec,...
    'x_fieldname', 'lambda_sparse',...
    'fields_to_plot',{{{'AdaptiveDmdc_obj','calc_reconstruction_error'},...
                        {'S_sparse_nnz'}}});

% this_pareto_obj = cell(num_figures,1);
use_baseline = false;
to_plot_global_only = false;
% this_scale_factor = 1e-9;
default_lines = cell(1, length(which_worms));
this_scale_factor = 2e-10;
if use_baseline
    baseline_func_persistence = ...
        @(x) x.AdaptiveDmdc_obj.calc_reconstruction_error([],true);
    y_func_global = ...
        @(x,~) x.run_with_only_global_control(...
        @(x2)x2.AdaptiveDmdc_obj.calc_reconstruction_error() );
end
for i = 1:length(which_worms)
    clear this_pareto_obj
    settings.file_or_dat = sprintf(filename_template, which_worms(i));
    this_pareto_obj = ParetoFrontObj('CElegansModel', settings);
    
    for i2=1:length(global_signal_modes{1})
        this_global_mode = global_signal_modes{1}{i2};
        % Combine error data and number of nonzeros
        this_pareto_obj.save_combined_y_val(...
            sprintf('%s_AdaptiveDmdc_obj_calc_reconstruction_error',this_global_mode),...
            sprintf('%s_S_sparse_nnz',this_global_mode),...
            this_scale_factor);
        
    end
    % Calculate a persistence baseline and combine with nnz
    this_global_mode = global_signal_modes{1}{1};
    if use_baseline
        this_pareto_obj.save_baseline(this_global_mode, baseline_func_persistence);
        this_pareto_obj.save_combined_y_val(...
            sprintf('baseline__%s',this_global_mode),...
            sprintf('%s_S_sparse_nnz',this_global_mode),...
            this_scale_factor);
    end
    
    % Calculate the error with only the global control signal, and combine
    if to_plot_global_only
        y_global_str = sprintf('global_only_%s',this_global_mode);
        this_pareto_obj.save_y_vals(...
            iterate_setting, [], y_func_global);
        this_pareto_obj.save_combined_y_val(...
            sprintf('%s_custom_func', this_global_mode),...
            sprintf('%s_S_sparse_nnz',this_global_mode),...
            this_scale_factor);
    end
    
    if isempty(all_figs{1})
        all_figs{1} = this_pareto_obj.plot_pareto_front('combine');
        leg_cell = ...
            {sprintf('Behavioral ID (worm %d)',which_worms(i)),...
            sprintf('Default value (worm %d)',which_worms(i))};
    else
        this_pareto_obj.plot_pareto_front('combine', true, all_figs{1});
        leg_cell = ...
            [leg_cell ...
            {sprintf('Behavioral ID (worm %d)',which_worms(i)),...
            sprintf('Default value (worm %d)',which_worms(i))}]; %#ok<AGROW>
    end
    xlabel('\lambda')
    ylabel('Weighted Error')
    
    % Plot a vertical line for the default value
    hold on
%     lambda_default = 0.043;
    [min_error, lambda_default] = min(...
        this_pareto_obj.y_struct.combine__ID_binary__ID_binary_);
    lambda_default = this_pareto_obj.x_vector(lambda_default);
    hax = all_figs{1}.Children(2);
    default_lines{i} = line([lambda_default lambda_default],get(hax,'YLim'),...
        'Color',[0 0 0], 'LineWidth',2, 'LineStyle','-.');
    hLegend = findobj(all_figs{1}, 'Type', 'Legend');
    legend(leg_cell);
    fprintf('The minimum error is %.3f at \lambda=%.3f\n',...
        min_error, lambda_default)
%     legend({'Full Behavioral ID', 'Simplified ID', 'Default value'});
end

% Set all the lines to the same height
hax = all_figs{1}.Children(2);
for i = 1:length(default_lines)
    default_lines{i}.YData =  get(hax,'YLim');
end
%---------------------------------------------
% Save figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        fname = sprintf('%sfigure_s3_%d', foldername, i);
        this_fig = all_figs{i};
        set(this_fig, 'Position', get(0, 'Screensize'));
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_raw.tex']);
        this_fig = prep_figure_no_axis();
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Supplementary Figure 4: global lambda errors for all worms
num_figures = 1;
all_figs = cell(num_figures,1);

filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';

to_plot_global_only = false;

model_settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'global_signal_mode', 'RPCA',...
    'max_rank_global', 0);
% sparse_lambda_vals = {{0.05}};
% global_lambda_vec = linspace(0.0018,0.01,3);
sparse_lambda_vals = {{0.03, 0.043, 0.056}};
% global_lambda_vec = linspace(0.0018, 0.01, 200);
global_lambda_vec = linspace(0.0018,0.01,10);
settings = struct(...
    'base_settings', model_settings,...
    'iterate_settings',struct('lambda_sparse',sparse_lambda_vals),...
    'x_vector', global_lambda_vec,...
    'x_fieldname', 'lambda_global',...
    'fields_to_plot',{{{'AdaptiveDmdc_obj','calc_reconstruction_error'},...
                        {'L_global_rank'}}});

this_pareto_obj = cell(num_figures,1);
this_scale_factor = 1e-5;
baseline_func_persistence = ...
    @(x) x.AdaptiveDmdc_obj.calc_reconstruction_error([],true);
y_func_global = ...
    @(x,~) x.run_with_only_global_control(...
    @(x2)x2.AdaptiveDmdc_obj.calc_reconstruction_error() );
for i=1:num_figures
    settings.file_or_dat = sprintf(filename_template, i);
    this_pareto_obj{i} = ParetoFrontObj('CElegansModel', settings);
    
    for i2=1:length(sparse_lambda_vals{1})
        this_sparse = sparse_lambda_vals{1}{i2};
        this_mode_str = this_pareto_obj{i}.make_valid_name(this_sparse);
        this_pareto_obj{i}.save_combined_y_val(...
            sprintf('%s_AdaptiveDmdc_obj_calc_reconstruction_error',this_mode_str),...
            sprintf('%s_L_global_rank',this_mode_str),...
            this_scale_factor);
    end
    
    % Calculate a persistence baseline and combine with rank
    this_mode_str = this_pareto_obj{i}.make_valid_name(sparse_lambda_vals{1}{1});
    this_pareto_obj{i}.save_baseline(this_mode_str, baseline_func_persistence);
    this_pareto_obj{i}.save_combined_y_val(...
        sprintf('baseline__%s',this_mode_str),...
        sprintf('%s_L_global_rank',this_mode_str),...
        this_scale_factor);
    
    % Calculate the error with only the global control signal, and combine
    if to_plot_global_only
        y_global_str = sprintf('global_only_%s',this_mode_str);
        this_pareto_obj{i}.save_y_vals(...
            iterate_setting, [], y_func_global);
        this_pareto_obj{i}.save_combined_y_val(...
            sprintf('%s_custom_func', y_global_str),...
            sprintf('%s_S_sparse_nnz',y_global_str),...
            this_scale_factor);
    end
        
    all_figs{i} = this_pareto_obj{i}.plot_pareto_front('combine');
end

%---------------------------------------------
% Save figures
%---------------------------------------------
for i = 1:length(all_figs)
    fname = sprintf('%sfigure_s4_%d', foldername, i);
    this_fig = all_figs{i};
    set(this_fig, 'Position', get(0, 'Screensize'));
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Figure 4e: Elimination path (Lasso)

% all_figs{5} = figure('DefaultAxesFontSize', 14);
% plot(all_err(which_ctr,:), 'LineWidth',2)
% xticks(1:max_iter)
% xticklabels

% Make sure it's working
% figure;
% a = B_prime_lasso_td(:,:,1);
% imagesc(a)
% colormap(cmap_white_zero(a))
% title('Original encoding')
% figure;
% a = B_prime_lasso_td(:,:,end);
% imagesc(a)
% colormap(cmap_white_zero(a))
% title(sprintf('Encoding after %d removals', max_iter))

% figure;
% imagesc(all_err)
% colorbar
%==========================================================================


%% Figure 2: rank and sparsity of signals


num_clusters = 3;
[idx, centroids] = kmeans(final_xy(2,:)', num_clusters, 'Replicates', 10);
[~, tmp] = max(final_xy(2,:));
real_clust_idx = idx(tmp);
[~, tmp] = min(final_xy(2,:));
noise_clust_idx = idx(tmp);

real_clust = final_xy(1, idx==real_clust_idx);
gray_clust = final_xy(1, logical(...
    (idx~=real_clust_idx).*(idx~=noise_clust_idx)) );
noise_clust = final_xy(1, idx==noise_clust_idx);
%==========================================================================


