


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


%% Figure 1: Intro to control
%---------------------------------------------
% First just do the cartoon outline
%---------------------------------------------
to_plot_figure_1 = false;
if to_plot_figure_1
    filename = 'C:\cygwin64\home\charl\GitWormSim\Model\simdata_original.csv';
    WormView(filename,struct('pauseAt',7.16,'startAtPause',true,'quitAtPause',true))
    error('Need to zoom by hand here')
    fig = prep_figure_no_axis();
    fname = sprintf('%sfigure_1_%d', foldername, 1);
    saveas(fig, fname, 'png');
end
%---------------------------------------------
% Next get some representative neurons
%---------------------------------------------
all_figs = cell(3,1);
% Use CElegans model to preprocess
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'filter_window_dat',6,...
    'use_deriv',false,...
    'to_normalize_deriv',false,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_preprocess = CElegansModel(filename, settings);

% Get the neurons
%   tspans decided by hand
sensory_neuron = 2;
sensory_tspan = 800:1800;
inter_neuron = 45;
inter_tspan = 600:1600;
motor_neuron = 114;
motor_tspan = 2021:3021;

% Actually plot
all_figs{1} = figure;
plot(my_model_preprocess.dat(sensory_neuron,sensory_tspan),...
    'LineWidth',8,'Color',my_cmap_3d(1,:))

all_figs{2} = figure;
plot(my_model_preprocess.dat(inter_neuron,inter_tspan),...
    'LineWidth',8,'Color',my_cmap_3d(2,:))

all_figs{3} = figure;
plot(my_model_preprocess.dat(motor_neuron,motor_tspan),...
    'LineWidth',8,'Color',my_cmap_3d(3,:))

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
        prep_figure_no_axis(this_fig)
        saveas(this_fig, fname, 'png');
    end
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


%% Figure 3: Hold-out cross-validation
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


%% Figure 4,5: Reconstructions (multiple methods)
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
all_figs = cell(12,1);

to_calc_double_RPCA = false;
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
% Simplify the labeling for better visualization
label_dict = containers.Map(...
    {1,2,3,4,5,6,7,8},...
    {1,1,2,3,4,4,4,5});
new_labels_key = ...
    {'Simple Forward',...
    'Dorsal Turn',...
    'Ventral Turn',...
    'Simple Reverse',...
    'NOSTATE'};
f = @(x) label_dict(x);
new_labels_ind = arrayfun(f, dat_struct.SevenStates(filter_window:end));
% Actually plot
all_figs{1} = plot_colored(proj3d,...
    new_labels_ind, new_labels_key, 'o');
title('Dynamics of the low-rank component (reconstructed)')
view(my_viewpoint)
% Now make the colormap match the bar graphs
for i=1:length(new_labels_key)
    all_figs{1}.Children(2).Children(i).CData = ...
        my_cmap_3d(my_cmap_dict(i),:);
end

% Now a single neuron reconstruction
neur_id = [38, 59];
fig_dict = containers.Map({neur_id(1), neur_id(2)}, {2, 3});
for i = neur_id
    [~, all_figs{fig_dict(i)}] = ...
        ad_obj_fig4.plot_reconstruction(true,false,true,i);
end

if to_calc_double_RPCA
    %---------------------------------------------
    % Get double RPCA model
    %---------------------------------------------
    settings = struct(...
        'to_subtract_mean',true,...
        'to_subtract_mean_sparse',false,...
        'to_subtract_mean_global',false,...
        'dmd_mode','func_DMDc',...
        'lambda_sparse', 0.04,...
        'filter_window_dat',6,...
        'use_deriv',true,...
        'to_normalize_deriv',true);
    my_model_fig4_b = CElegansModel(filename, settings);
    
    % Use original control; 3d pca plot
    % NOTE: reconstruction is not really that impressive here!
    my_model_fig4_b.set_simple_labels();
    all_figs{4} = my_model_fig4_b.plot_colored_reconstruction();
    view(my_viewpoint)
    % Now make the colormap match the bar graphs
    for i=1:length(new_labels_key)
        all_figs{4}.Children(2).Children(i).CData = ...
            my_cmap_3d(my_cmap_dict(i),:);
    end
    
    % Reconstruct some individual neurons
    neur_id = [38, 59];
    fig_dict = containers.Map({neur_id(1), neur_id(2)}, {5, 6});
    for i = neur_id
        [~, all_figs{fig_dict(i)}] = ...
            my_model_fig4_b.AdaptiveDmdc_obj.plot_reconstruction(true,false,true,i);
    end
    
end

%---------------------------------------------
% Get experimentalist labels model
%---------------------------------------------
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'filter_window_dat',3,...
    'use_deriv',true,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_fig4_c = CElegansModel(filename, settings);

% Use original control; 3d pca plot
my_model_fig4_c.set_simple_labels();
all_figs{7} = my_model_fig4_c.plot_colored_reconstruction();
view(my_viewpoint)
% Now make the colormap match the bar graphs
for i=1:length(new_labels_key)
    all_figs{7}.Children(2).Children(i).CData = ...
        my_cmap_3d(my_cmap_dict(i),:);
end

% Also original data; same for all models
% all_figs{10} = my_model_fig4_c.plot_colored_data(false, 'o');
% for i=1:length(new_labels_key)
%     all_figs{10}.Children(2).Children(i).CData = ...
%         my_cmap_3d(my_cmap_dict(i),:);
% end
% view(my_viewpoint)

% Reconstruct some individual neurons
neur_id = [38, 59, 90];
fig_dict = containers.Map(...
    {neur_id(1), neur_id(2), neur_id(3)},...
    {8, 9, 10});
for i = neur_id
    [~, all_figs{fig_dict(i)}] = ...
        my_model_fig4_c.AdaptiveDmdc_obj.plot_reconstruction(true,false,true,i);
end

%---------------------------------------------
% Compare with global signals only
%---------------------------------------------
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    ...'filter_window_dat',3,...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_fig4_d = CElegansModel(filename, settings);

my_model_fig4_d.set_simple_labels();
all_figs{11} = my_model_fig4_d.plot_colored_reconstruction();
view(my_viewpoint)
% Now make the colormap match the bar graphs
for i=1:length(new_labels_key)
    all_figs{11}.Children(2).Children(i).CData = ...
        my_cmap_3d(my_cmap_dict(i),:);
end

%---------------------------------------------
% Simplest comparison: no control at all!
%---------------------------------------------
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','tdmd',...
    'add_constant_signal',false,...
    ...'filter_window_dat',3,...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'lambda_sparse',0);
settings.global_signal_mode = 'None';
my_model_fig4_e = CElegansModel(filename, settings);

my_model_fig4_e.set_simple_labels();
all_figs{12} = my_model_fig4_e.plot_colored_reconstruction();
view(my_viewpoint)
% Now make the colormap match the bar graphs
for i=1:length(new_labels_key)
    all_figs{12}.Children(2).Children(i).CData = ...
        my_cmap_3d(my_cmap_dict(i),:);
end

%---------------------------------------------
% Save figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = sprintf('%sfigure_4_%d_scratch', foldername, i);
        this_fig = all_figs{i};
        ax = this_fig.Children(2);
        ax.Clipping = 'Off';
        prep_figure_no_axis(this_fig)
        zoom(1.17)
        saveas(this_fig, fname, 'png');
    end
end

%==========================================================================


%% Figure 5: Fixed points

filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
all_figs = cell(1,1);
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'global_signal_mode', 'ID_binary_and_grad');
my_model_fig5 = CElegansModel(filename, settings);
my_model_fig5.set_simple_labels();

% Plot the original data
all_figs{1} = my_model_fig5.plot_colored_data(false, 'o');
for i=1:length(my_model_fig5.state_labels_key)
    all_figs{1}.Children(2).Children(i).CData = ...
        my_cmap_3d(my_cmap_dict(i),:);
end
view(my_viewpoint)
% Now plot the fixed points
my_model_fig5.run_with_only_global_control(@(x)...
    plot_colored_fixed_point(x,'Simple Reverse', true, all_figs{1}) );
my_model_fig5.run_with_only_global_control(@(x)...
    plot_colored_fixed_point(x,'Simple Forward', true, all_figs{1}) );
[~, b] = all_figs{1}.Children.Children;

% Add invisible axes on top and place the stars there so they will be
% visible
ax = all_figs{1}.Children(2);
axHidden = axes('Visible','off','hittest','off'); % Invisible axes
linkprop([ax axHidden],{'CameraPosition' 'XLim' 'YLim' 'ZLim' 'Position'}); % The axes should stay aligned
set(b(1), 'Parent', axHidden)
set(b(2), 'Parent', axHidden)

% Save figures
if to_save
    fname = sprintf('%sfigure_5_%d', foldername, 1);
    this_fig = all_figs{1};
    prep_figure_no_axis(this_fig)
    ax.Visible = 'Off';
%     axes(ax)
%     zoom(1.175) % Decided by hand
%     axes(axHidden)
%     zoom(1.175) % Decided by hand
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Figure 6 (simple): Neuron classifications
all_figs = cell(1,1);

filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'filter_window_dat',0,...
    'use_deriv',false);
settings.global_signal_mode = 'ID_binary';
% settings.global_signal_mode = 'ID_binary_and_grad';

%---------------------------------------------
% Calculate 5 worms and get roles
%---------------------------------------------
all_models = cell(5,1);
all_roles_global = cell(5,2);
for i=1:5
    filename = sprintf(filename_template,i);
%     if i==4
%         settings.lambda_sparse = 0.035; % Decided by looking at pareto front
%     else
%         settings.lambda_sparse = 0.05;
%     end
    all_models{i} = CElegansModel(filename,settings);
end
%% Now we have the models
max_err_percent = 0.4;
for i=1:5
    % Global mode actuation
    [all_roles_global{i,1}, all_roles_global{i,2}] = ...
        all_models{i}.calc_neuron_roles_in_global_modes(true, [], max_err_percent);
end

%---------------------------------------------
% Data preprocessing
%---------------------------------------------
[ combined_dat_global, all_labels_global ] =...
    combine_different_trials( all_roles_global );
num_neurons = size(combined_dat_global,1);

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
role_counts = zeros(num_neurons,length(possible_roles));
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_global, possible_roles{i}),2);
end
% Long names mean it was an ambiguous identification
name_lengths = cellfun(@(x) length(x)<6, all_labels_global)';
this_ind = find((sum(role_counts,2)>3).*name_lengths);
all_figs{1} = figure('DefaultAxesFontSize',14);
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



