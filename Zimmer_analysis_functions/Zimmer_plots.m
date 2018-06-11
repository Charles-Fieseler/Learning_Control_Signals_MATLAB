

%% Figure 1: intro to control
% Maybe have a trace of some controller neurons?
%==========================================================================


%% Figure 2: Outlier detection
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
    fname = sprintf('../figures/figure_2_%d',i);
    this_fig = all_figs{i};
    set(this_fig, 'Position', get(0, 'Screensize'));
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Figure 3: Robust PCA
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
all_figs = cell(3,1);

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

title2 = 'Large \lambda low-rank component';
title1 = 'Large \lambda sparse component';
dat2 = my_model_fig3.L_sparse;
dat1 = my_model_fig3.S_sparse;
plot_mode = '2 1';
[ all_figs{2} ] = plot_2imagesc_colorbar( ...
    dat1, dat2, plot_mode, title1, title2 );

title2 = 'Small \lambda low-rank component';
title1 = 'Small \lambda sparse component';
dat2 = my_model_fig3.L_global;
dat1 = my_model_fig3.S_global;
plot_mode = '2 1';
[ all_figs{3} ] = plot_2imagesc_colorbar( ...
    dat1, dat2, plot_mode, title1, title2 );

% Save figures
for i = 1:length(all_figs)
    fname = sprintf('../figures/figure_3_%d',i);
    this_fig = all_figs{i};
    set(this_fig, 'Position', get(0, 'Screensize'));
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Figure 4: Reconstructions (multiple methods)
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
all_figs = cell(9,1);

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
approx_data = ad_obj_fig4.plot_reconstruction(true, true).';

% Now use robust PCA and visualize this using the same algorithm as above
lambda = 0.05;
[L_reconstruct, S_reconstruct] = RobustPCA(approx_data, lambda);
% Plot the 2nd low-rank component
filter_window = 10;
L_filter2 = my_filter(L_reconstruct,filter_window)';
[u,s,v,proj3d] = plotSVD(L_filter2(:,filter_window:end),...
    struct('PCA3d',false,'sigma',false));
all_figs{2} = plot_colored(proj3d,...
    dat_struct.SevenStates(filter_window:end),dat_struct.SevenStatesKey,'o');
title('Dynamics of the low-rank component (reconstructed)')

% Now a single neuron reconstruction
neur_id = [59];
[~, all_figs{3}] = ad_obj_fig4.plot_reconstruction(true, [], true, neur_id);

%---------------------------------------------
% Get double RPCA model
%---------------------------------------------
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
my_model_fig4_b = CElegansModel(filename, settings);

% Use original control; 3d pca plot
% NOTE: reconstruction is not really that impressive here!
my_model_fig4_b.add_partial_original_control_signal();
my_model_fig4_b.plot_reconstruction_user_control();
all_figs{4} = my_model_fig4_b.plot_colored_user_control([],false);

% Reconstruct some individual neurons
neur_id = [45, 59];
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
all_figs{1} = my_model_fig4_c.plot_colored_data(false, 'o');

% Reconstruct some individual neurons
neur_id = [45, 59];
fig_dict = containers.Map({neur_id(1), neur_id(2)}, {8, 9});
for i = neur_id
    [~, all_figs{fig_dict(i)}] = ...
        my_model_fig4_c.AdaptiveDmdc_obj.plot_reconstruction(true,false,true,i);
end

%---------------------------------------------
% Save figures
%---------------------------------------------
for i = 1:length(all_figs)
    fname = sprintf('../figures/figure_4_%d',i);
    this_fig = all_figs{i};
    set(this_fig, 'Position', get(0, 'Screensize'));
    saveas(this_fig, fname, 'png');
end

%==========================================================================


%% Figure 5: Fixed points
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
all_figs = cell(3,1);
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
my_model_fig5 = CElegansModel(filename, settings);

all_figs{1} = my_model_fig5.plot_colored_fixed_point('REVSUS',true);
all_figs{2} = my_model_fig5.plot_colored_fixed_point('SLOW', true);
all_figs{3} = my_model_fig5.plot_colored_fixed_point('FWD', true);

% Save figures
for i = 1:length(all_figs)
    fname = sprintf('../figures/figure_5_%d',i);
    this_fig = all_figs{i};
    set(this_fig, 'Position', get(0, 'Screensize'));
    saveas(this_fig, fname, 'png');
end
%==========================================================================






