


%% Define folder to save in
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\figures\';
%==========================================================================


%% Define colormap
num_colors = 16;
my_colormap = brewermap(num_colors,'OrRd');
my_colormap(1,:) = ones(1,3);
colormap(my_colormap)
my_colormap(end,:) = zeros(1,3);
set(0, 'DefaultFigureColormap', my_colormap)
% caxis([-0.5,1.0])
close all

%==========================================================================


%% Figure 1: Intro to control
% Maybe have a trace of some controller neurons?
to_plot_figure_1 = false;
if to_plot_figure_1
    filename = 'C:\cygwin64\home\charl\GitWormSim\Model\simdata_original.csv';
    WormView(filename,struct('pauseAt',7.16,'startAtPause',true,'quitAtPause',true))
    error('Need to zoom by hand here')
    fig = prep_figure_no_axis();
    fname = sprintf('%sfigure_1_%d', foldername, 1);
    saveas(fig, fname, 'png');
end
%==========================================================================


%% Figure 2: Robust PCA
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
all_figs = cell(9,1);

% Calculate double RPCA model
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
my_model_fig3 = CElegansModel(filename, settings);

% Plot decompositions
all_figs{1} = figure('DefaultAxesFontSize',12);
imagesc(my_model_fig3.dat_without_control);
ylabel('Neurons')
xlabel('Time')
colorbar
title('Original data')
caxis(all_figs{1}.CurrentAxes, [-0.0, 1.0])
% Large lambda (very sparse)
title2 = 'Large \lambda low-rank component';
title1 = 'Large \lambda sparse component';
dat2 = my_model_fig3.L_sparse;
dat1 = my_model_fig3.S_sparse;
plot_mode = '2_figures';
[ all_figs{2}, all_figs{3} ] = plot_2imagesc_colorbar( ...
    dat1, dat2, plot_mode, title1, title2 );
caxis(all_figs{2}.CurrentAxes, [-0.0, 1.0])
caxis(all_figs{3}.CurrentAxes, [-0.0, 1.0])
% Also plot a shorter time period
ind = 1000:1500;
dat2 = my_model_fig3.L_sparse(:,ind);
dat1 = my_model_fig3.S_sparse(:,ind);
[ all_figs{6}, all_figs{7} ] = plot_2imagesc_colorbar( ...
    dat1, dat2, plot_mode, title1, title2 );
caxis(all_figs{6}.CurrentAxes, [-0.0, 1.0])
caxis(all_figs{7}.CurrentAxes, [-0.0, 1.0])

% Small lambda (low-rank)
title2 = 'Small \lambda low-rank component';
title1 = 'Small \lambda sparse component';
dat2 = my_model_fig3.L_global_raw;
dat1 = my_model_fig3.S_global;
plot_mode = '2_figures';
[ all_figs{4}, all_figs{5} ] = plot_2imagesc_colorbar( ...
    dat1, dat2, plot_mode, title1, title2 );
caxis(all_figs{4}.CurrentAxes, [-0.0, 1.0])
caxis(all_figs{5}.CurrentAxes, [-0.0, 1.0])
% Also plot a shorter time period
ind = 1000:1500;
dat2 = my_model_fig3.L_global_raw(:,ind);
dat1 = my_model_fig3.S_global(:,ind);
[ all_figs{8}, all_figs{9} ] = plot_2imagesc_colorbar( ...
    dat1, dat2, plot_mode, title1, title2 );
caxis(all_figs{8}.CurrentAxes, [-0.0, 1.0])
caxis(all_figs{9}.CurrentAxes, [-0.0, 1.0])

% Save figures
for i = 1:length(all_figs)
    this_fig = all_figs{i};
    prep_figure_no_axis(this_fig)
    colorbar off;
    fname = sprintf('%sfigure_3_%d', foldername, i);
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Figure 3: Reconstructions (multiple methods)
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
all_figs = cell(10,1);

%---------------------------------------------
% Get neuron removal model
%---------------------------------------------
dat_struct = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat_struct.traces,3).';

% For comparison, we can look at the 3d diagram produced by the
% reconstructed data. The first step is to get the AdaptiveDmdc object;
% see AdaptiveDmdc_documentation for a more thorough explanation:
id_struct = struct(...
    'ID', {dat_struct.ID},...
    'ID2', {dat_struct.ID2},...
    'ID3', {dat_struct.ID3});
cutoff_multiplier = 3.0;
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',true,...
    'to_plot_nothing',true,...
    'id_struct',id_struct,...
    'cutoff_multiplier', cutoff_multiplier);
ad_obj_fig4 = AdaptiveDmdc(this_dat, settings);

% Use this object to reconstruct the data. First, plot it in comparison to
% the original data:
approx_data = ad_obj_fig4.plot_reconstruction(true, false).';

% Now use robust PCA and visualize this using the same algorithm as above
lambda = 0.05;
[L_reconstruct, S_reconstruct] = RobustPCA(approx_data, lambda);
% Plot the 2nd low-rank component
filter_window = 10;
L_filter2 = my_filter(L_reconstruct,filter_window)';
[u,s,v,proj3d] = plotSVD(L_filter2(:,filter_window:end),...
    struct('PCA3d',false,'sigma',false));
all_figs{1} = plot_colored(proj3d,...
    dat_struct.SevenStates(filter_window:end),dat_struct.SevenStatesKey,'o');
title('Dynamics of the low-rank component (reconstructed)')

% Now a single neuron reconstruction
neur_id = [38, 59];
fig_dict = containers.Map({neur_id(1), neur_id(2)}, {2, 3});
for i = neur_id
    [~, all_figs{fig_dict(i)}] = ...
        ad_obj_fig4.plot_reconstruction(true,false,true,i);
end

%---------------------------------------------
% Get double RPCA model
%---------------------------------------------
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true,...
    'lambda_sparse', 0.04);
my_model_fig4_b = CElegansModel(filename, settings);

% Use original control; 3d pca plot
% NOTE: reconstruction is not really that impressive here!
my_model_fig4_b.add_partial_original_control_signal();
my_model_fig4_b.plot_reconstruction_user_control();
all_figs{4} = my_model_fig4_b.plot_colored_user_control([],false);

% Reconstruct some individual neurons
neur_id = [38, 59];
fig_dict = containers.Map({neur_id(1), neur_id(2)}, {5, 6});
for i = neur_id
    [~, all_figs{fig_dict(i)}] = ...
        my_model_fig4_b.AdaptiveDmdc_obj.plot_reconstruction(true,false,true,i);
end

%---------------------------------------------
% Get experimentalist labels model
%---------------------------------------------
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
settings.global_signal_mode = 'ID_and_offset';
my_model_fig4_c = CElegansModel(filename, settings);

% Use original control; 3d pca plot
% NOTE: reconstruction is not really that impressive here!
my_model_fig4_c.add_partial_original_control_signal();
my_model_fig4_c.plot_reconstruction_user_control();
all_figs{7} = my_model_fig4_c.plot_colored_user_control([],false);

% Also original data; same for all models
all_figs{10} = my_model_fig4_c.plot_colored_data(false, 'o');

% Reconstruct some individual neurons
neur_id = [38, 59];
fig_dict = containers.Map({neur_id(1), neur_id(2)}, {8, 9});
for i = neur_id
    [~, all_figs{fig_dict(i)}] = ...
        my_model_fig4_c.AdaptiveDmdc_obj.plot_reconstruction(true,false,true,i);
end

%---------------------------------------------
% Save figures
%---------------------------------------------
for i = 1:length(all_figs)
    if isempty(all_figs{i})
        continue;
    end
    fname = sprintf('%sfigure_4_%d', foldername, i);
    this_fig = all_figs{i};
    prep_figure_no_axis(this_fig)
    saveas(this_fig, fname, 'png');
end

%==========================================================================


%% Figure 4: Fixed points
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
all_figs = cell(1,1);
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
my_model_fig5 = CElegansModel(filename, settings);

all_figs{1} = my_model_fig5.plot_colored_data(false, 'o');
view(0,60)
[~, b] = all_figs{1}.Children.Children;
alpha(b, 0.3)
my_model_fig5.plot_colored_fixed_point('REVSUS', true, all_figs{1});
my_model_fig5.plot_colored_fixed_point('FWD', true, all_figs{1});
% all_figs{2} = my_model_fig5.plot_colored_fixed_point('SLOW', true);
% all_figs{2} = my_model_fig5.plot_colored_data(false, 'o');
% view(10,40)
% [~, b] = all_figs{2}.Children.Children;
% alpha(b, 0.3)
% my_model_fig5.plot_colored_fixed_point('FWD', true, all_figs{2});

% Save figures
for i = 1:length(all_figs)
    fname = sprintf('%sfigure_5_%d', foldername, i);
    this_fig = all_figs{i};
    prep_figure_no_axis(this_fig)
    zoom(1.15) % Decided by hand
    saveas(this_fig, fname, 'png');
end
%==========================================================================



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
all_figs = cell(3,1);

filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID';

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
for i=1:5
    % Use the dynamic attractor
    [all_roles_dynamics{i,1}, all_roles_dynamics{i,2}] = ...
        all_models{i}.calc_neuron_roles_in_transition(true, [], true);
    % Just use centroid of a behavior
    [all_roles_centroid{i,1}, all_roles_centroid{i,2}] = ...
        all_models{i}.calc_neuron_roles_in_transition(true, [], false);
    % Global mode actuation
    [all_roles_global{i,1}, all_roles_global{i,2}] = ...
        all_models{i}.calc_neuron_roles_in_global_modes(true);
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
this_ind = find(sum(role_counts,2)>1);
all_figs{1} = figure('DefaultAxesFontSize',14);
bar(role_counts(this_ind,:), 'stacked')
legend(possible_roles)
xticks(1:length(this_ind));
xticklabels(all_labels_dynamic(this_ind))
xtickangle(90)
title('Neuron roles using similarity to attractors (no 4th worm)')

%---------------------------------------------
% Bar graph of transition kicks (centroids, no worm 4)
%---------------------------------------------
these_worms = [1, 2, 3, 5];
role_counts = zeros(num_neurons,length(possible_roles));
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_centroid(:,these_worms), possible_roles{i}),2);
end
this_ind = find(sum(role_counts,2)>1);
all_figs{2} = figure('DefaultAxesFontSize',14);
bar(role_counts(this_ind,:), 'stacked')
legend(possible_roles)
xticks(1:length(this_ind));
xticklabels(all_labels_centroid(this_ind))
xtickangle(90)
title('Neuron roles using similarity to attractors (no 4th worm)')

%---------------------------------------------
% Roles for global neurons
%---------------------------------------------
possible_roles = unique(combined_dat_global);
possible_roles(cellfun(@isempty,possible_roles)) = [];
role_counts = zeros(num_neurons,length(possible_roles));
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_global, possible_roles{i}),2);
end
this_ind = find(sum(role_counts,2)>1);
all_figs{3} = figure('DefaultAxesFontSize',14);
bar(role_counts(this_ind,:), 'stacked')
legend(possible_roles)
xticks(1:length(this_ind));
xticklabels(all_labels_global(this_ind))
xtickangle(90)
title('Neuron roles using similarity to global mode activation')

%---------------------------------------------
% Save figures
%---------------------------------------------
for i = 1:length(all_figs)
    fname = sprintf('%sfigure_s2_%d', foldername, i);
    this_fig = all_figs{i};
    set(this_fig, 'Position', get(0, 'Screensize'));
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Supplementary Figure 3: sparse lambda errors for all worms (ID signal)
all_figs = cell(5,1);

filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';

model_settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
% global_signal_modes = {{'ID'}};
global_signal_modes = {{'ID','ID_binary'}};
% global_signal_modes = {{'ID','ID_simple','ID_binary'}};
lambda_vec = linspace(0.02,0.1,100);
settings = struct(...
    'base_settings', model_settings,...
    'iterate_settings',struct('global_signal_mode',global_signal_modes),...
    'x_vector', lambda_vec,...
    'x_fieldname', 'lambda_sparse',...
    'fields_to_plot',{{{'AdaptiveDmdc_obj','calc_reconstruction_error'},...
                        {'S_sparse_nnz'}}});

all_pareto_objs = cell(5,1);
for i=1:5
    settings.file_or_dat = sprintf(filename_template, i);
    all_pareto_objs{i} = ParetoFrontObj('CElegansModel', settings);
    all_pareto_objs{i}.save_combined_y_val(...
        'ID_AdaptiveDmdc_objcalc_reconstruction_error', 'ID_S_sparse_nnz');
    all_figs{i} = all_pareto_objs{i}.plot_pareto_front('combine');
end

%---------------------------------------------
% Save figures
%---------------------------------------------
for i = 1:length(all_figs)
    fname = sprintf('%sfigure_s3_%d', foldername, i);
    this_fig = all_figs{i};
    set(this_fig, 'Position', get(0, 'Screensize'));
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Supplementary Figure 4: global lambda errors for all worms
all_figs = cell(5,1);

filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';

model_settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'global_signal_mode', 'RPCA',...
    'max_rank_global', 0);
% sparse_lambda_vals = {{0.05}};
% global_lambda_vec = linspace(0.0018,0.01,1);
sparse_lambda_vals = {{0.025, 0.04, 0.055}};
global_lambda_vec = linspace(0.0018, 0.01, 20);
settings = struct(...
    'base_settings', model_settings,...
    'iterate_settings',struct('lambda_sparse',sparse_lambda_vals),...
    'x_vector', global_lambda_vec,...
    'x_fieldname', 'lambda_global',...
    'fields_to_plot',{{{'AdaptiveDmdc_obj','calc_reconstruction_error'}}});

all_pareto_objs = cell(5,1);
for i=1:5
    settings.file_or_dat = sprintf(filename_template, i);
    all_pareto_objs{i} = ParetoFrontObj('CElegansModel', settings);
    all_figs{i} = all_pareto_objs{i}.plot_pareto_front();
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





