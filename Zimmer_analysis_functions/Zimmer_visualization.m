%% Use a t-SNE for visualization
filename = 'C:\Users\charl\Documents\MATLAB\Collaborations/Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
dat1 = importdata(filename);
[mapped_data2, mapping2] = compute_mapping(dat1.traces,'t-SNE',3);

plot_colored(mapped_data2, dat1.SevenStates, dat1.SevenStatesKey);
title('t-SNE visualization of Zimmer data')
to_save = false;
if to_save
    % Save the plot
    folder_name = 'C:\Users\charl\Documents\Current_work\Presentations\Zimmer_skype\';
    savefig(fig, [folder_name 't-SNE visualization']);
end

to_plot_line = false;
if to_plot_line
    % Plots a line through all the points
    %   messy but visualizes transitions between clusters
    plot3(mapped_data2(:,1),mapped_data2(:,2),mapped_data2(:,3));
end
%==========================================================================


%% Use t-SNE to find similar trajectories
% use the t-SNE 'tubes' as labels instead of hand-labeled behaviors

%---------------------------------------------
% Do t-SNE
%---------------------------------------------
filename = 'C:\Users\charl\Documents\MATLAB\Collaborations/Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
dat1 = importdata(filename);
[mapped_data2, mapping2] = compute_mapping(dat1.traces,'t-SNE',3);


to_plot = false;
if to_plot
    categories = dat1.SevenStates;
    num_states = length(unique(categories));
    cmap = lines(num_states);
    if num_states>7
        % Makes the 8th and later categories black (otherwise would repeat)
        cmap(8:end,:) = zeros(size(cmap(8:end,:)));
    end
    figure;
    hold on
    f = gca;
    for jC = unique(categories)
        ind = find(categories==jC);
        plot3(mapped_data2(ind,1),mapped_data2(ind,2),mapped_data2(ind,3),...
            'o','LineWidth',2);
        f.Children(1).Color = cmap(jC,:);
    end
    legend(dat1.SevenStatesKey)
    title('t-SNE visualization of Zimmer data')
    to_plot_line = false;
    if to_plot_line
        % Plots a line through all the points
        %   messy but visualizes transitions between clusters
        plot3(mapped_data2(:,1),mapped_data2(:,2),mapped_data2(:,3));
    end
end

%---------------------------------------------
% Get labels
%---------------------------------------------
max_dist = 2;
out = reconstruct_trajecories(mapped_data2, max_dist);

%---------------------------------------------
% Do patchDMD on sequential labels
%---------------------------------------------
settings = struct();
dat_seq = dat1;
dat_seq.SevenStates = out.seq_vec;
dat_seq.SevenStatesKey = arrayfun(@(x)num2str(x),...
    1:length(dat_seq.SevenStates),...
    'UniformOutput',false);
dat_patch_seq = patchDMD(dat_seq, settings);

dat_patch_seq.initialize_all_similarity_objects();
dat_patch_seq.calc_all_label_similarities();

%---------------------------------------------
% Do patchDMD on stitched-together labels
%---------------------------------------------
dat_stitch = dat1;
dat_stitch.SevenStates = out.stitch_vec;
dat_stitch.SevenStatesKey = arrayfun(@(x)num2str(x),...
    1:length(dat_stitch.SevenStates),...
    'UniformOutput',false);
dat_patch_stitch = patchDMD(dat_stitch, settings);

dat_patch_stitch.initialize_all_similarity_objects();
dat_patch_stitch.calc_all_label_similarities();

%==========================================================================


%% Use t-SNE for visualization on the derivatives
filename = 'C:\Users\charl\Documents\MATLAB\Collaborations/Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
dat1 = importdata(filename);
[mapped_data2, mapping2] = compute_mapping(dat1.tracesDif,'t-SNE',3);

% This vector is for the original data, which is one frame longer than the
% derivative time series... for now, just throw out the last point
categories = dat1.SevenStates(1:end-1);
num_states = length(unique(categories));
cmap = lines(num_states);
if num_states>7
    % Makes the 8th and later categories black (otherwise would repeat)
    cmap(8:end,:) = zeros(size(cmap(8:end,:)));
end
fig = figure;
hold on
f = gca;
for jC = unique(categories)
    ind = find(categories==jC);
    plot3(mapped_data2(ind,1),mapped_data2(ind,2),mapped_data2(ind,3),...
        'o','LineWidth',2);
    f.Children(1).Color = cmap(jC,:);
end
legend(dat1.SevenStatesKey)
title('t-SNE visualization of Zimmer data')
to_save = false;
if to_save
    % Save the plot
    folder_name = 'C:\Users\charl\Documents\Current_work\Presentations\Zimmer_skype\';
    savefig(fig, [folder_name 't-SNE visualization']);
end

to_plot_line = false;
if to_plot_line
    % Plots a line through all the points
    %   messy but visualizes transitions between clusters
    plot3(mapped_data2(:,1),mapped_data2(:,2),mapped_data2(:,3));
end
%==========================================================================


%% Visualization without the (very noisy) sensory neurons
% Throw away sensory neurons (not well correlated)
if ~exist('dat_hierarchy','var')
    filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
    patch_settings = struct('patch_settings',struct('windowSize',30));
    settings = struct('patchDMD_settings', patch_settings,...
        'use_derivatives',false);
    dat_hierarchy = hierarchicalDMD(filename, settings);
    
end

dat = importdata(filename);
%---------------------------------------------
% Use hierarchy object to get rid of sensory neurons
%---------------------------------------------
sensory_label = dat_hierarchy.class_type2index('sensory');
sensory_indices = (dat_hierarchy.class_ids==sensory_label);
% Use derivatives or data and derivatives stacked
% my_pca_dat = dat.traces(:,~sensory_indices)';
% my_pca_dat = dat.tracesDif(:,~sensory_indices)';
my_pca_dat = [dat.traces(1:end-1,~sensory_indices)'; 
              dat.tracesDif(:,~sensory_indices)'];
%---------------------------------------------
% Simple PCA plot
%---------------------------------------------
pca_settings = struct('PCA3d',true,'PCA_opt','');
plotSVD(my_pca_dat, pca_settings);
%---------------------------------------------
% t-SNE plot
%---------------------------------------------
[mapped_data2, mapping2] = compute_mapping(my_pca_dat','t-SNE',5);

% This vector is for the original data, which is one frame longer than the
% derivative time series... for now, just throw out the last point
categories = dat.SevenStates(1:end-1);
num_states = length(unique(categories));
cmap = lines(num_states);
if num_states>7
    % Makes the 8th and later categories black (otherwise would repeat)
    cmap(8:end,:) = zeros(size(cmap(8:end,:)));
end
fig = figure;
hold on
f = gca;
for jC = unique(categories)
    ind = find(categories==jC);
    plot3(mapped_data2(ind,1),mapped_data2(ind,2),mapped_data2(ind,3),...
        'o','LineWidth',2);
    f.Children(1).Color = cmap(jC,:);
end
legend(dat1.SevenStatesKey)
title('t-SNE visualization of Zimmer data')
to_save = false;
%==========================================================================


%% ROBUST PCA of the raw data
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);

% opt_just_PCA3d = struct('PCA3d',true,'sigma',false);
opt_just_sigma = struct('PCA3d',false,'sigma',true);

% default lambda is ~0.018
% If lambda = 0.01, the rank drastically decrease in the low-rank side
%   (from ~120 to 7)
lambda = 0.01;
[L_raw, S_raw] = RobustPCA(dat5.traces, lambda);

plot_2imagesc_colorbar(S_raw',L_raw','',...
    'Raw data matrix (sparse))','Raw data matrix (low-rank)')

% plotSVD(L_raw.',opt_just_PCA3d);
% filter_window = 20;
% filter_L_raw = filter(ones(1,filter_window)/filter_window,1,...
%     L_raw);
% plot_colored(filter_L_raw.',dat5.SevenStates,dat5.SevenStatesKey,'');
% title('Raw data matrix (low rank)')
% plotSVD(filter_L_raw,opt_just_sigma);
% title('Raw data matrix (low rank)')
plot_colored(L_raw.',dat5.SevenStates,dat5.SevenStatesKey,'');
title('Raw data matrix (low rank)')
plotSVD(L_raw,opt_just_sigma);
title('Raw data matrix (low rank)')


% plotSVD(S_raw.',opt_just_PCA3d);
% which_modes_to_plot = [1 2 4];
% plot_colored(S_raw(:,which_modes_to_plot),...
%     dat5.SevenStates,dat5.SevenStatesKey);
plot_colored(S_raw,dat5.SevenStates,dat5.SevenStatesKey);
title('Raw data matrix (sparse)')
[u,s,v,proj3d] = plotSVD(S_raw,opt_just_sigma);
title('Raw data matrix (sparse)')
%==========================================================================


%% PLOT ME
%% Use Robust PCA twice to separate into motor-, inter-, and sensory neurons
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat5.traces,3).';
% this_deriv = my_filter(dat5.tracesDif,3).';
% this_dat = [this_dat(:,10:end); this_deriv(:,9:end)];

% Expect that the motor neuron dynamics are ~4d, vis the eigenworms
%   This value of lambda (heuristically) gives a dimension of 4 for the raw
%   data
lambda = 0.0065;
[L_raw, S_raw] = RobustPCA(this_dat, lambda);
% Plot the VERY low-rank component
L_filter = my_filter(L_raw,10);
% plotSVD(L_filter(:,15:end)',struct('PCA3d',true,'sigma',false));
% title('Dynamics of the low-rank component')
[u_low_rank, s_low_rank] = ...
    plotSVD(L_filter(:,15:end)',struct('sigma_modes',1:3));
title('3 svd modes (very low rank component)')
s_low_rank = diag(s_low_rank);
% u_low_rank = plotSVD(L_raw,struct('sigma_modes',1:4));
% figure;
% imagesc(L_filter);
% title('Reconstruction of the very low-rank component (data)')

% 2nd RobustPCA, with much more sparsity
lambda = 0.05;
[L_2nd, S_2nd] = RobustPCA(this_dat, lambda);
% [L_2nd, S_2nd] = RobustPCA(S_raw, lambda);
% Plot the 2nd low-rank component
filter_window = 1;
L_filter2 = my_filter(L_2nd,filter_window);
% plotSVD(L_filter2(:,filter_window:end),struct('sigma_modes',1:3));
[~,~,~,proj3d] = plotSVD(L_filter2(:,filter_window:end),...
    struct('PCA3d',true,'sigma',false));
plot_colored(proj3d,...
    dat5.SevenStates(filter_window:end),dat5.SevenStatesKey,'o');
title('Dynamics of the low-rank component (data)')
% figure;
% imagesc(S_2nd')
% title('Sparsest component')
drawnow

% Augment full data with 2nd Sparse signal AND lowest-rank signal
%   Note: just the sparse signal doesn't work
ad_dat = this_dat - mean(this_dat,1);
S_2nd = S_2nd - mean(S_2nd,1);
tol = 1e-2;
S_2nd_nonzero = S_2nd(:,max(abs(S_2nd),[],1)>tol);
L_low_rank = u_low_rank(:,1:4)*s_low_rank(1:4,1:4);
L_low_rank = L_low_rank - mean(L_low_rank,1);
% Allow these signals to affect a larger phase space
% L_low_rank = repmat(L_low_rank,1,5);

% ad_dat = [ad_dat.'; S_2nd.'];
L_low_rank = L_low_rank';
num_pts = min([size(L_low_rank,2) size(S_2nd_nonzero,2) size(ad_dat,2)]);
ad_dat = [ad_dat(:,1:num_pts);...
    S_2nd_nonzero(:,1:num_pts);...
    L_low_rank(:,1:num_pts)];
% Adaptive dmdc
x_ind = 1:size(this_dat,1);
% Just move these 'sensory' neurons to the controller
% sensory_neurons_eyeball = [1, 44, 90, 121];
% x_ind(sensory_neurons_eyeball) = []; 
id_struct = struct('ID',{dat5.ID},'ID2',{dat5.ID2},'ID3',{dat5.ID3});
settings = struct('to_plot_cutoff',true,...
    'to_plot_data_and_outliers',true,...
    'id_struct',id_struct,...
    'sort_mode','user_set',...
    'x_indices',x_ind,...
    'use_optdmd',false);
ad_obj_augment2 = AdaptiveDmdc(ad_dat, settings);

% Plot reconstructions
dat_approx_control = ad_obj_augment2.plot_reconstruction(true,true);

interesting_neurons = [41, 58, 55, 42, 45];
for i=interesting_neurons
    ad_obj_augment2.plot_reconstruction(true,true,true,i);
end
%==========================================================================


%% PLOT ME
%% Use Robust PCA twice on reconstructed data (partial)
%---------------------------------------------
% Do first AdaptiveDmdc (get reconstruction)
%---------------------------------------------
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat5.traces,3).';

use_deriv = true;
if use_deriv
    this_deriv = my_filter(dat5.tracesDif,3).';
    this_dat = [this_dat(:,10:end); this_deriv(:,9:end)];
end
% Adaptive dmdc separates out neurons to use as control signals
id_struct = struct('ID',{dat5.ID},'ID2',{dat5.ID2},'ID3',{dat5.ID3});
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',true,...
    'to_plot_data',true,...
    'id_struct',id_struct,...
    'to_plot_data_and_outliers',true,...
    'to_print_error',false);
ad_obj = AdaptiveDmdc(this_dat,settings);

%---------------------------------------------
% Plot the first pass reconstruction
%---------------------------------------------
interesting_neurons = [40, 37, 55, 42];
for i=interesting_neurons
    ad_obj.plot_reconstruction(true,true,true,i);
end

approx_dat = ad_obj.plot_reconstruction(true,true)';

%---------------------------------------------
% Do first Robust PCA (very low-rank component)
%---------------------------------------------
% Expect that the motor neuron dynamics are ~4d, vis the eigenworms
%   This value of lambda (heuristically) gives a dimension of 4 for the raw
%   data
lambda = 0.0065;
[L_raw_reconstruct, S_raw_reconstruct] = RobustPCA(approx_dat, lambda);
% Plot the VERY low-rank component
% L_filter = my_filter(L_raw_reconstruct,10)';
plotSVD(L_raw_reconstruct,struct('PCA3d',true));
title('Dynamics of the low-rank component')
figure;
imagesc(L_raw_reconstruct);
title('Reconstruction of the very low-rank component')

%---------------------------------------------
% 2nd Robust PCA (very sparse component)
%---------------------------------------------
lambda = 0.05;
[L_2nd_reconstruct, S_2nd_reconstruct] = RobustPCA(approx_dat, lambda);
% Plot the 2nd low-rank component
filter_window = 10;
L_filter2 = my_filter(L_2nd_reconstruct,filter_window)';
plotSVD(L_filter2(:,filter_window:end)',struct('sigma_modes',1:3));
[u,s,v,proj3d] = plotSVD(L_filter2(:,filter_window:end),...
    struct('PCA3d',true,'sigma',false));
plot_colored(proj3d,...
    dat5.SevenStates(2*filter_window-1:end),dat5.SevenStatesKey,'o');
title('Dynamics of the low-rank component (reconstructed)')

%---------------------------------------------
% Augment full data with 2nd Sparse signal (dmdc again)
%---------------------------------------------
ad_dat = approx_dat - mean(approx_dat,1);
S_2nd = S_2nd_reconstruct - mean(S_2nd_reconstruct,1);
ad_dat = [ad_dat.'; S_2nd_reconstruct.'];
% Adaptive dmdc
settings = struct('to_plot_cutoff',true,...
    'to_plot_data_and_outliers',true,...
    'to_print_error',false,...
    'sort_mode','user_set',...
    'x_indices',1:size(approx_dat,2));
ad_obj_augment = AdaptiveDmdc(ad_dat, settings);

% Plot reconstructions
ad_obj_augment.plot_reconstruction(true,true);

interesting_neurons = [41, 58, 55, 42];
for i=interesting_neurons
    ad_obj_augment.plot_reconstruction(true,true,true,i);
end
%==========================================================================


%% Compare experimental and dynamic connectomes

settings = struct('to_subtract_diagonal',true);
compare_obj = compare_connectome(ad_obj_augment_sparse, settings);
compare_obj.plot_imagesc()
%==========================================================================


%% Visualize using only the (normalized) derivative
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

% lambda_global needs to be larger; default value gives rank(low-rank)=0
lambda_global = 0.012;
% lambda_sparse = 0.01;
settings = struct('use_only_deriv',true,'to_normalize_deriv',true,...
    'lambda_global',lambda_global);%, 'lambda_sparse',lambda_sparse);
my_model_deriv = CElegansModel(filename, settings);
my_model_deriv.plot_colored_data();
ad_obj = my_model_deriv.AdaptiveDmdc_obj;
ad_obj.plot_reconstruction(true,true);

interesting_neurons = [84, 45, 58, 46, 15, 174, 167];
for i=interesting_neurons
    ad_obj.plot_reconstruction(true,false,true,i);
end
%==========================================================================


%% Plot control fixed points and direction arrows
% filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
% my_model = CElegansModel(filename);

% Actuate a single mode for a long time to equilibrate
ctr_ind = 131;
num_neurons = 129;
custom_signal = max(max(my_model.dat_with_control(num_neurons+ctr_ind,:))) *...
    ones(length(ctr_ind),1000);
t_start = 500;
is_original_neuron = false;
my_model.add_partial_original_control_signal(ctr_ind,...
    custom_signal, t_start, is_original_neuron)
% my_model.ablate_neuron(neurons_to_ablate);
my_model.plot_reconstruction_user_control();
fig = my_model.plot_colored_user_control();
title(sprintf('Global mode %d',ctr_ind))
my_model.reset_user_control()

% Compare this to the arrow of the control displacement
my_model.plot_colored_control_arrow(ctr_ind, [], [], fig);

% Also plot the original control signal
figure;
if ~is_original_neuron
    original_ind = ctr_ind+num_neurons;
else
    original_ind = ctr_ind;
end
plot(my_model.dat_with_control(original_ind,:).')
title(sprintf('Original control signal for mode %d',ctr_ind))
%==========================================================================


%% Actuate original neurons and look at the fixed point
ctr_ind = 45;
num_neurons = 129;
custom_signal = 0.5*max(my_model.dat_with_control(num_neurons+ctr_ind,:)) *...
ones(length(ctr_ind),1000);
t_start = 500;
is_original_neuron = true;
my_model.add_partial_original_control_signal(ctr_ind,...
custom_signal, t_start, is_original_neuron)
% my_model.ablate_neuron(neurons_to_ablate);
my_model.plot_reconstruction_user_control();
fig = my_model.plot_colored_user_control();
title(sprintf('Global mode %d',ctr_ind))
my_model.reset_user_control()
% Compare this to the arrow of the control displacement
my_model.plot_colored_control_arrow(ctr_ind, [], 10, fig);
% Also plot the original control signal
figure;
if ~is_original_neuron
original_ind = ctr_ind+num_neurons;
else
original_ind = ctr_ind;
end
plot(my_model.dat_with_control(original_ind,:))
title(sprintf('Original control signal for mode %d',ctr_ind))
%==========================================================================


%% Actuate original neurons and their control shadows; observe FP
ctr_ind = 72;
num_neurons = 129;
custom_signal = 0.5*max(my_model.dat_with_control(num_neurons+ctr_ind,:)) *...
    ones(length(ctr_ind),1000);
t_start = 500;
is_original_neuron = true;
my_model.add_partial_original_control_signal(ctr_ind,...
    custom_signal, t_start, is_original_neuron)
% Also add control signal version
is_original_neuron = false;
my_model.add_partial_original_control_signal(ctr_ind,...
    custom_signal, t_start, is_original_neuron)

my_model.plot_reconstruction_user_control();
fig = my_model.plot_colored_user_control();
title(sprintf('Global mode %d',ctr_ind))
my_model.reset_user_control()
% Compare this to the arrow of the control displacement
my_model.plot_colored_control_arrow(ctr_ind, [], 10, fig);
% Also plot the original control signal
figure;
if ~is_original_neuron
original_ind = ctr_ind+num_neurons;
else
original_ind = ctr_ind;
end
plot(my_model.dat_with_control(original_ind,:))
title(sprintf('Original control signal for mode %d',ctr_ind))
%==========================================================================


%% Plot control fixed points over time for global modes
% filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
% my_model = CElegansModel(filename);

% Actuate a single mode for a long time to equilibrate
num_neurons = my_model.dat_sz(1);
ctr_ind = (num_neurons+1):(my_model.total_sz(1)-num_neurons);
% custom_signal = max(max(my_model.dat_with_control(num_neurons+ctr_ind,:))) *...
%     ones(length(ctr_ind),1000);
% t_start = 500;
is_original_neuron = false;
my_model.add_partial_original_control_signal(ctr_ind,...
    [], [], is_original_neuron)
% my_model.ablate_neuron(neurons_to_ablate);
my_model.plot_reconstruction_user_control();

my_model.plot_user_control_fixed_points('FWD');
my_model.plot_user_control_fixed_points('SLOW');
my_model.plot_user_control_fixed_points('REVSUS');
my_model.plot_user_control_fixed_points('DT');
my_model.plot_user_control_fixed_points('VT');
my_model.plot_user_control_fixed_points('REV1');
my_model.plot_user_control_fixed_points('REV2');

my_model.plot_user_control_fixed_points();
% Plot multiple things on this next one
fig = my_model.plot_colored_user_control();
title(sprintf('Global mode %d',ctr_ind))
my_model.reset_user_control()

% Compare this to the arrow of the control displacement
my_model.plot_colored_control_arrow(ctr_ind, [], [], fig);

% Also plot the original control signal
figure;
if ~is_original_neuron
    original_ind = ctr_ind+num_neurons;
else
    original_ind = ctr_ind;
end
plot(my_model.dat_with_control(original_ind,:).')
title(sprintf('Original control signal for mode %d',ctr_ind))
%==========================================================================


% SHOWME
%% Plot fixed points of various labeled behaviors
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
my_model = CElegansModel(filename);

% Plot all fixed points
my_model.plot_colored_fixed_point();

% Plot all individually labeled behaviors (as clouds)
my_model.plot_colored_fixed_point('FWD');
my_model.plot_colored_fixed_point('REVSUS');
my_model.plot_colored_fixed_point('DT');
my_model.plot_colored_fixed_point('VT');
my_model.plot_colored_fixed_point('REV1');
my_model.plot_colored_fixed_point('REV2');
my_model.plot_colored_fixed_point('SLOW');

% Now as centroids
my_model.plot_colored_fixed_point('REVSUS',true);
my_model.plot_colored_fixed_point('SLOW', true);
my_model.plot_colored_fixed_point('FWD', true);
my_model.plot_colored_fixed_point('DT', true);
my_model.plot_colored_fixed_point('VT', true);
my_model.plot_colored_fixed_point('REV1', true);
my_model.plot_colored_fixed_point('REV2', true);
%==========================================================================


%% Actuate unlabeled original neurons and look at the fixed point
ctr_ind = 1;
num_neurons = 129;
custom_signal = 0.05*max(my_model.dat_with_control(num_neurons+ctr_ind,:)) *...
ones(length(ctr_ind),1000);
t_start = 500;
is_original_neuron = true;
my_model.add_partial_original_control_signal(ctr_ind,...
custom_signal, t_start, is_original_neuron)
% my_model.ablate_neuron(neurons_to_ablate);
my_model.plot_reconstruction_user_control();
fig = my_model.plot_colored_user_control();
title(sprintf('Global mode %d',ctr_ind))
my_model.reset_user_control()
% Compare this to the arrow of the control displacement
my_model.plot_colored_control_arrow(ctr_ind, [], 10, fig);
% Also plot the original control signal
figure;
if ~is_original_neuron
original_ind = ctr_ind+num_neurons;
else
original_ind = ctr_ind;
end
plot(my_model.dat_with_control(original_ind,:))
title(sprintf('Original control signal for mode %d',ctr_ind))
%==========================================================================


%% Look at even lower rank "global modes"
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
% lambda_global = 0.004; % rank=1
% lambda_global = 0.005; % rank=2
% lambda_global = 0.0055; % rank=3
lambda_global = 0.0065; % rank=4; default
settings = struct('lambda_global',lambda_global);
my_model2 = CElegansModel(filename, settings);

my_model2.plot_colored_fixed_point();

% Plot all individually labeled behaviors (as clouds)
my_model2.plot_colored_fixed_point('FWD');
my_model2.plot_colored_fixed_point('REVSUS');

% 3d visualization
my_model2.add_partial_original_control_signal();
my_model2.plot_reconstruction_user_control();
my_model2.plot_colored_user_control([],false);

%==========================================================================


%% Figure 3 draft plots (reconstruction)
filename5 = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

% Get first model and error
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'AdaptiveDmdc_settings',struct('what_to_do_dmd_explosion','project'));
% settings.global_signal_mode = 'ID_and_offset';
settings.global_signal_mode = 'ID_and_offset';
my_model_fig3 = CElegansModel(filename5, settings);

% Use original control; 3d pca plot
% NOTE: reconstruction is not really that impressive here!
my_model_fig3.add_partial_original_control_signal();
my_model_fig3.plot_reconstruction_user_control();
my_model_fig3.plot_colored_user_control([],false);

% Reconstruct some individual neurons
neur_id = [45, 77, 93];
for i = neur_id
    my_model_fig3.AdaptiveDmdc_obj.plot_reconstruction(true,false,true,i);
end
%==========================================================================


%% Look at which neurons are actuated by global signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
settings.global_signal_mode = 'ID';
my_model_ID_test = CElegansModel(filename, settings);

% Baseline L2 error
disp(my_model_ID_test.AdaptiveDmdc_obj.calc_reconstruction_error())

%---------------------------------------------
% Global signals
%---------------------------------------------
% Look at control matrix (B)
figure;
num_neurons = my_model_ID_test.original_sz(1);
B_global_all = my_model_ID_test.AdaptiveDmdc_obj.A_original(1:num_neurons,...
    (2*num_neurons+1):end);
imagesc(B_global_all);
title('B matrix')

% Look at control matrix (B) without DC offset
figure;
num_neurons = my_model_ID_test.original_sz(1);
B_global = my_model_ID_test.AdaptiveDmdc_obj.A_original(1:num_neurons,...
    (2*num_neurons+1):end-2);
imagesc(B_global);
title('B matrix (only learned/IDed modes)')

% Narrow these down to which neurons are important for which behaviors
%   Assume a single control signal (ID); ignore offset
tol = 0.003;
group1 = find(B_global > tol);
group2 = find(B_global < -tol);

disp('Neurons important for group 1:')
my_model_ID_test.AdaptiveDmdc_obj.get_names(group1);

disp('Neurons important for group 2:')
my_model_ID_test.AdaptiveDmdc_obj.get_names(group2);

% Look at global control signals
figure;
u = my_model_ID_test.dat_with_control((2*num_neurons+1):end,:);
plot(u.')
title('Global control signals')

%---------------------------------------------
% Sparse signals
%---------------------------------------------
% Look at sparse control signals
figure;
u = my_model_ID_test.dat_with_control((num_neurons+1):(2*num_neurons),:);
imagesc(u)
title('Global control signals')

% Look at which neurons are important based on how they actuate either
% group1 or group2 of the global modes above
%   TO DO: look at activity * eigenvalues or some other dynamic properties
B = my_model_ID_test.AdaptiveDmdc_obj.A_original(1:num_neurons,...
    (num_neurons+1):(2*num_neurons));
tol = 0.001;
group1_ind = B_global > tol;
group2_ind = B_global < -tol;
u_sum = sum(u,2);

group1_scores = zeros(length(u_sum),1);
group2_scores = zeros(length(u_sum),1);
for i=1:length(group1)
    % Do not want to sum the abs
    group1_scores(i) = sum(u_sum(i) * B(group1_ind,i));
    group2_scores(i) = sum(u_sum(i) * B(group2_ind,i));
end

figure;
subplot(2,1,1)
plot(group1_scores)
title('group1 (probably revsus)')
subplot(2,1,2)
plot(group2_scores)
title('group2 (probably fwd)')


%==========================================================================


%% Look at the eigenvalues for motor neurons
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
settings.global_signal_mode = 'ID_and_offset';
my_model_ID_test = CElegansModel(filename, settings);

% Find motor neurons
num_neurons = my_model_ID_test.original_sz(1);
all_names = my_model_ID_test.AdaptiveDmdc_obj.get_names(1:num_neurons);
class_A = contains(all_names, {'A0','A1'});
class_B = contains(all_names, {'B0','B1'});

% Get eigenvectors and eigenvalues of intrinsic dynamics
A = my_model_ID_test.AdaptiveDmdc_obj.A_separate(1:num_neurons,1:num_neurons);
[V, D] = eig(A, 'vector');
actual_values = abs(D)>1e-5;
V = V(:,actual_values);
D = D(actual_values);

% Get columns that have appreciable loading on motor neurons
% tol = 0.1;
% for i=1:length(D)
%     A_loading = sum(real(V(class_A,i)));
%     if abs(A_loading)>tol
%         fprintf('Eigenvalue for class_A eigenvector: %f+%f (loading: %f)\n',...
%             real(D(i)), imag(D(i)), A_loading)
%     end
%     B_loading = sum(real(V(class_B,i)));
%     if abs(B_loading)>tol
%         fprintf('Eigenvalue for class_B eigenvector: %f+%f (loading: %f)\n',...
%             real(D(i)), imag(D(i)), B_loading)
%     end
% end

% Scatterplots of absolute weighting
figure;
subplot(2,1,1)
my_colormap = sum(real(V(class_A,:)).*abs(V(class_A,:)),1);
scatter(real(D),imag(D),[],my_colormap,'filled')
title('Colored by A-class loading')
colorbar
subplot(2,1,2)
my_colormap = sum(real(V(class_B,:)).*abs(V(class_B,:)),1);
scatter(real(D),imag(D),[],my_colormap,'filled')
title('Colored by B-class loading')
colorbar

%==========================================================================


%% Look transition kicks for all controllers
filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
settings.global_signal_mode = 'ID_and_offset';
my_model_ID_test = CElegansModel(filename, settings);

% Get neuron names
num_neurons = my_model_ID_test.original_sz(1);
all_names = my_model_ID_test.AdaptiveDmdc_obj.get_names(1:num_neurons);
known_names = find(~cellfun(@isempty,all_names));

% Plot all of them!
for i=1:length(known_names)
    this_neuron = known_names(i);
    fig = my_model_ID_test.plot_colored_control_arrow(this_neuron,[],50);
    title(sprintf('Control kick of neuron %d (%s)',...
        this_neuron, all_names{this_neuron}));
%     [~, b] = fig.Children.Children;
%     alpha(b, 0.3)
    
    fig = my_model_ID_test.plot_colored_control_arrow(...
        this_neuron,[],50, [], true);
    title(sprintf('Intrinsic dynamics kick of neuron %d (%s)',...
        this_neuron, all_names{this_neuron}));
%     [~, b] = fig.Children.Children;
%     alpha(b, 0.3)
    pause
end

% Plot only some interesting ones
interesting_ones = [46, 84, 90];
for i=1:length(interesting_ones)
    this_neuron = interesting_ones(i);
%     fig = my_model_ID_test.plot_colored_control_arrow(this_neuron,[],50);
%     title(sprintf('Control kick of neuron %d (%s)',...
%         this_neuron, all_names{this_neuron}));

    fig = my_model_ID_test.plot_colored_control_arrow(...
        this_neuron,[],10, [], true);
    title(sprintf('Intrinsic dynamics kick of neuron %d (%s)',...
        this_neuron, all_names{this_neuron}));
end

%==========================================================================


%% Look at movie of control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
settings.global_signal_mode = 'ID_and_offset';
my_model_ID_test = CElegansModel(filename, settings);

% my_model_ID_test.plot_colored_arrow_movie();

% Save a movie
my_model_ID_test.plot_colored_arrow_movie(...
    true, false, '../scratch/worm1.avi', 'mean');

% Save another movie; the default is to plot both arrows from the data
my_model_ID_test.plot_colored_arrow_movie(...
    true, false, '../scratch/worm1_diff.avi');

% Save a movie with the reconstruction
my_model_ID_test.plot_colored_arrow_movie(...
    true, false, '../scratch/worm1_reconstruct.avi')

%==========================================================================


%% Look at suspiciously spike-like neurons
% Has very good SMB oscillations
filename4 = '../../Zimmer_data/WildType_adult/simplewt4/wbdataset.mat';

% filename4 = '../../Zimmer_data/WildType_adult/simplewt4/wbdataset.mat';
dat4 = importdata(filename4);

labels_of_interest = {...'RIVL','RIVR', ...
    ...'SMDVR', 'SMDVL',...
    'SMBDR', 'SMBDL',...
    'RIBR', 'RIBL',...
    ...'RID',...
    ...'RIMR','RIML',...
    ...'AVAL', 'AVAR',...
    ...'RIS','ALA',...
    ...'SAAVR','SAAVL', 'ASKL','ASKR'};
    };
neurons_of_interest = zeros(size(labels_of_interest));
for i = 1:length(labels_of_interest)
    this_neuron = find(cellfun(...
        @(x) strcmp(x,labels_of_interest{i}), dat4.ID));
    if isempty(this_neuron)
        continue
    end
    neurons_of_interest(i) = this_neuron;
end
% neurons_of_interest = [80, 85, 43, 125, 82, 83, 41, 45];%, 56, 48];

for i=1:length(neurons_of_interest)
    if neurons_of_interest(i)==0
        continue
    end
    plot_colored(dat4.traces(:,neurons_of_interest(i)),...
        dat4.SevenStates,dat4.SevenStatesKey);
    title(sprintf('Neuron ID: %s',...
        dat4.ID{neurons_of_interest(i)}))
end

%==========================================================================


%% Plot possible Fitzhugh-Nagumo hidden variables
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
% filename = '../../Zimmer_data/WildType_adult/simplewt4/wbdataset.mat';
dat = importdata(filename);

% WORM 5
interesting_neuron = 90; %SMBDL, a spike-like guy
% interesting_neuron = 113; %SMBDR, a spike-like guy
% interesting_neuron = 71; %RIVL
% interesting_neuron = 79; %RIVR
% interesting_neuron = 44; %SMDVR
% interesting_neuron = 42; %SMDVL


% WORM 4
% RIVL/R, SMDVR, SMBDR, RIMR/L, SAAVR/L
% neurons_of_interest = [80, 85, 43, 125, 82, 83, 41, 45]; % Divergence...
% interesting_neuron = 125;

% Calculate FHN model variables
v = dat.traces(1:end-1,interesting_neuron);
vt = dat.tracesDif(:,interesting_neuron);
w_minus_I = v - (v.^3)/3 - vt;
% figure;
% plot(w_minus_I)
% title('Hidden variable (w) - Input current')

my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
filter_w_minus_I = my_filter(w_minus_I, 10);
% figure;
% plot(filter_w_minus_I)
% title('Hidden variable (w) - Input current')

% Try robust PCA to get the spikes (I) out
to_plot_labmdas = false;
if to_plot_labmdas
    all_lambdas = linspace(0.05,0.03,10);
    for i=1:length(all_lambdas)
        lambda = all_lambdas(i);
        [L, S, ~, nnz_S] = RobustPCA(w_minus_I, lambda);
        figure;
        plot(L);
        hold on;
        plot(S);
        legend({'L','S'})
        title(sprintf('Lambda=%.4f, nnz_S=%d',lambda,nnz_S))
    end
end

% Looking a the graphs;
good_lambda = 0.0456;

% Plot this variable vs. the transition colors
plot_colored(filter_w_minus_I,...
    dat.SevenStates(1:end-1),dat.SevenStatesKey);
title('Hidden variable (w) - Input current')
plot_colored(v,...
    dat.SevenStates(1:end-1),dat.SevenStatesKey);
title('Original neuron trace')
% plot_colored(vt,...
%     dat.SevenStates(1:end-1),dat.SevenStatesKey);
% title('Original neuron derivative')

% Normalize the traces to a max of 3
%   Chosen because this is worm5 neuron 90's max, which gives a very good
%   separation of behaviors
z = 3.5/max(v);
normalized_w_minus_I = z*v - ((z*v).^3)/3 - z*vt;

plot_colored(my_filter(normalized_w_minus_I, 10),...
    dat.SevenStates(1:end-1),dat.SevenStatesKey);
title('Hidden variable (w) - Input current')


%==========================================================================


%% Supplementary figure? looking at neuron errors
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

% First the settings and model
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_errors = CElegansModel(filename, settings);

% Plot the errors 
my_model_errors.AdaptiveDmdc_obj.plot_data_and_outliers();
num_neurons = my_model_errors.original_sz(1)/2;
xlim([0,num_neurons]);

%==========================================================================


%% Understand the distribution of final errors
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat = importdata(filename);
mu = 0;
sigma = 1e-4;
sz = size(dat.traces');

% Calculate the errors for each 
rng default;  % For reproducibility
err = normrnd(mu, sigma, sz(1), sz(2));  % Simulate errors
total_err = sum(abs(err),2);

figure;
bin_width = 1e-3;
histogram(total_err,'BinWidth',bin_width);

%==========================================================================


%% PLOT ME
%% Use plots from paper plots to look at a subset of bar charts
all_figs = cell(1,1);

ad_settings = struct('sparsity_goal', 0.7);
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    ...'dmd_mode','sparse',...
    'add_constant_signal',false,...
    'filter_window_dat',0,...
    'to_add_stimulus_signal', false,...
    ...'use_deriv',true,...
    'lambda_sparse', 0,...
    'AdaptiveDmdc_settings', ad_settings,...
    'use_deriv',false,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary';

%---------------------------------------------
% Build filename array (different data formats...)
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
% Calculate all worms and get roles
%---------------------------------------------
all_models = cell(n,1);
all_roles_dynamics = cell(n,2);
all_roles_centroid = cell(n,2);
all_roles_global = cell(n,2);
for i = 1:n
    filename = all_filenames{i};
    dat = importdata(filename);
    if i > num_type_1
        % The pre-let datasets have some strange artifacts at the beginning
        dat.traces = dat.traces(200:end, :);
    end
    all_models{i} = CElegansModel(dat, settings);
end

%% Use above models to produce a bar chart
disp('Calculating roles...')
max_err_percent = 0.30;
% max_err_percent = 0;
% class_tol = 0.01;
class_tol = 0.00;
only_named_neurons = true;
to_include_turns = false;
for i = 1:n
    [all_roles_global{i,1}, all_roles_global{i,2}] = ...
        all_models{i}.calc_neuron_roles_in_global_modes(...
        only_named_neurons, class_tol, max_err_percent,...
        to_include_turns);
end

%---------------------------------------------
% Data preprocessing
%---------------------------------------------
[ combined_dat_global, all_labels_global ] =...
    combine_different_trials( all_roles_global, false );

%---------------------------------------------
% Roles for global neurons
%---------------------------------------------
disp('Producing bar chart...')
% First make the field names the same
if ~to_include_turns
    d = containers.Map(...
        {'group 1', 'group 2', 'other', '', 'group 1 2', 'high error'},...
        {'simple REVSUS', 'simple FWD', 'other', '', 'both', 'z_error'});
    combined_dat_global = cellfun(@(x) d(x), combined_dat_global,...
        'UniformOutput', false);
else
    d = containers.Map(...
        {'group 1', 'group 2', 'other', '', 'group 1 2',...
        'group 3', 'group 4', 'group 3 4', 'high error'},...
        {'simple REVSUS', 'simple FWD', 'other', '', 'simple REV and FWD',...
        'DT', 'VT', 'both turns', 'z_error'});
    combined_dat_global = cellfun(@(x) d(x), combined_dat_global,...
        'UniformOutput', false);
end
possible_roles = unique(combined_dat_global);
possible_roles(cellfun(@isempty,possible_roles)) = [];
num_neurons = size(combined_dat_global,1);
role_counts = zeros(num_neurons,length(possible_roles));
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_global, possible_roles{i}),2);
end
all_figs{1} = figure('DefaultAxesFontSize',14);
if only_named_neurons
    % Long names mean it was an ambiguous identification
    name_lengths = cellfun(@(x) length(x)<6, all_labels_global)';
    this_ind = find((sum(role_counts,2)>1).*name_lengths);
    b = bar(role_counts(this_ind,:), 'stacked');
    xticks(1:length(this_ind));
    xticklabels(all_labels_global(this_ind))
    xtickangle(90)
else
    b = bar(role_counts, 'stacked');
end
if length(b) < 6
    for i=1:length(b)
        b(i).FaceColor = my_cmap_3d(i,:);
    end
else
    warning('Colormap failed; too many entries')
end
if max_err_percent > 0
    legend([possible_roles(1:end-1); {'high error'}])
else
    legend(possible_roles)
end
yticks(1:max(max((role_counts))))
ylabel('Number of times identified')
title('Neuron roles using global mode activation')

%% Diagnostics to see how well the models do
%---------------------------------------------
% Look at AVA for all of them
%---------------------------------------------
this_neuron = 'AVAL';
ind = contains(all_labels_global, this_neuron);
for i = 1:length(all_models)
    all_models{i}.plot_reconstruction_interactive(true, this_neuron);
    title(sprintf('Classification is %s',...
        combined_dat_global{ind, i}))
    pause
end

%---------------------------------------------
% Look at a model with VERY bad AVB reconstructions
%---------------------------------------------
% i=2;
% tol = 0.001;
% n = all_models{i}.original_sz(1)/2;
% tmp = all_models{i}.AdaptiveDmdc_obj.A_separate;
% tmp(abs(tmp)<tol) = 0;
% figure
% imagesc(tmp(1:n,(2*n+1):(2*n+9)))
% xticklabels(all_models{i}.state_labels_key)
% all_models{i}.plot_reconstruction_interactive(true, 'AVB');
%==========================================================================


%% Look at the eigenvalues for motor neurons (updated)
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_eigen = CElegansModel(filename, settings);

% Find motor neurons
num_neurons = my_model_eigen.original_sz(1)/2;
all_names = my_model_eigen.AdaptiveDmdc_obj.get_names(1:num_neurons);
class_A = contains(all_names, {'A0','A1'});
class_A = class_A(1:num_neurons);
class_B = contains(all_names, {'B0','B1'});
class_B = class_B(1:num_neurons);

% Get eigenvectors and eigenvalues of intrinsic dynamics
A = my_model_eigen.AdaptiveDmdc_obj.A_separate(1:num_neurons,1:num_neurons);
[V, D] = eig(A, 'vector');
tol = 1e-3;
actual_values = abs(D)>tol;
V = V(:,actual_values);
D = D(actual_values);

% Get columns that have appreciable loading on motor neurons
% tol = 0.1;
% for i=1:length(D)
%     A_loading = sum(real(V(class_A,i)));
%     if abs(A_loading)>tol
%         fprintf('Eigenvalue for class_A eigenvector: %f+%f (loading: %f)\n',...
%             real(D(i)), imag(D(i)), A_loading)
%     end
%     B_loading = sum(real(V(class_B,i)));
%     if abs(B_loading)>tol
%         fprintf('Eigenvalue for class_B eigenvector: %f+%f (loading: %f)\n',...
%             real(D(i)), imag(D(i)), B_loading)
%     end
% end

% Scatterplots of absolute weighting
figure;
subplot(2,1,1)
my_colormap = sum(real(V(class_A,:)).*abs(V(class_A,:)),1);
scatter(real(D),imag(D),[],my_colormap,'filled')
title('Colored by A-class loading')
colorbar
subplot(2,1,2)
my_colormap = sum(real(V(class_B,:)).*abs(V(class_B,:)),1);
scatter(real(D),imag(D),[],my_colormap,'filled')
title('Colored by B-class loading')
colorbar
xlabel('Real part (decay/growth)')
ylabel('Imaginary part (1/Hz)')

% Only plot dots with high loadings
tol = 1e-5;
A_colormap = sum(real(V(class_A,:)).*abs(V(class_A,:)),1);
A_ind = (abs(A_colormap)>tol);
B_colormap = sum(real(V(class_B,:)).*abs(V(class_B,:)),1);
B_ind = (abs(B_colormap)>tol);

f = figure;
subplot(2,1,1)
scatter(real(D(A_ind)),imag(D(A_ind)),[],A_colormap(A_ind),'filled')
title('Colored by A-class loading')
colorbar
subplot(2,1,2)
scatter(real(D(B_ind)),imag(D(B_ind)),[],B_colormap(B_ind),'filled')
title('Colored by B-class loading')
colorbar
xlabel('Real part (decay/growth)')
ylabel('Imaginary part (1/Hz)')

%==========================================================================


%% Use Isomap to visualize data
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat = importdata(filename);

% Only do the first couple PCA dimensions because the dataset is too big!
% [U, S, V] = svd(dat.traces);
% svd_dims = 20;
[U, S, V] = svd([dat.traces(1:end-1,:), dat.tracesDif]);
svd_dims = 40;
X = U(:,1:svd_dims);
labels = dat.SevenStates(1:size(X,1));

% figure
% scatter3(X(:,1), X(:,2), X(:,3), 5, labels); 
plot_colored(X,labels,dat.SevenStatesKey)
title('Original dataset'), drawnow
no_dims = round(intrinsic_dim(X, 'MLE'));
disp(['MLE estimate of intrinsic dimensionality: ' num2str(no_dims)]);

[mappedX, mapping] = compute_mapping(X, 'Isomap', 3);
% figure
% scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3), 5, labels); 
plot_colored(mappedX,labels,dat.SevenStatesKey)
title('Result of Isomap')



%==========================================================================


%% Eigenvalues for motor neurons (built-in function)
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_eigen = CElegansModel(filename, settings);

% DB01
my_model_eigen.plot_eigenvalues_and_frequencies(114);
%==========================================================================


%% Look at sparse matrices
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
ad_settings = struct('sparsity_goal',0.6); % Default
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'dmd_mode','sparse',...
    'AdaptiveDmdc_settings',ad_settings);
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_sparse = CElegansModel(filename, settings);

settings.global_signal_mode = 'ID_binary';
my_model_sparse2 = CElegansModel(filename, settings);

% EVEN SPARSER
ad_settings = struct('sparsity_goal',0.25);
settings.AdaptiveDmdc_settings = ad_settings;
my_model_very_sparse = CElegansModel(filename, settings);

% Look at matrices
my_model_sparse.plot_matrix_A(true,true,true);
my_model_sparse.plot_matrix_A(false,true,true);

my_model_sparse.plot_matrix_B(true,'ID_binary',true);

my_model_sparse2.plot_matrix_B(false,'ID_binary',true);
my_model_sparse2.plot_matrix_B(true,'ID_binary',true);

%==========================================================================


%% Sparse matrices with augmented data
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
ad_settings = struct('sparsity_goal',0.35);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'to_normalize_deriv',true,...
    'dmd_mode','sparse',...
    'augment_data',2,...
    'lambda_sparse',0,...
    'AdaptiveDmdc_settings',ad_settings);
settings.global_signal_mode = 'ID_binary';
my_model_sparse_aug2 = CElegansModel(filename, settings);

%==========================================================================


%% Look at reversal bout lengths for regularities
filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';

num_worms = 5;
bout_lengths = cell(num_worms,1);
figure;
for individual = 1:num_worms
    dat_struct = importdata(sprintf(filename_template,individual));
    dat = dat_struct.traces';

    % Get the bout lengths
    rev_label = find(strcmp(dat_struct.SevenStatesKey,'REVSUS'));
    rev_ind_vec = (dat_struct.SevenStates==rev_label);
    rev_ind_cumsum = cumsum(rev_ind_vec);
    rev_ind_diff = diff(rev_ind_vec);
    for i = 1:(length(rev_ind_vec)-1)
        if rev_ind_diff(i) == -1
            rev_ind_cumsum(i:end) = rev_ind_cumsum(i:end) - rev_ind_cumsum(i);
        end
    end

    rev_ind_cumsum_diff = diff(rev_ind_cumsum);
    bout_lengths{individual} = rev_ind_cumsum(rev_ind_cumsum_diff<0);
    
    subplot(num_worms, 1, individual)
    histogram(bout_lengths{individual},'BinWidth',10)
    title(sprintf('Worm %d',individual))
    xlim([0,150])
    ylim([0 10])
end
    
% Histograms
% figure;
% histogram(bout_lengths,'BinWidth',10)


%==========================================================================


%% Look at correlation coefficients between d(state)/dt and turning neurons
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);

% Get the indices of some neurons
labels_of_interest = {'AIBL','AIBR', ...
    'SMDVL', 'SMDVR', 'SMBDL', 'SMBDR',...
    'RIML','RIMR',...
    'AVAL', 'AVAR',...
    'AVEL','AVER',...
    'RMED','RMEV',...
    'AVBL','AVBR'};
neurons_of_interest = zeros(size(labels_of_interest));
for i = 1:length(labels_of_interest)
    this_neuron = find(cellfun(...
        @(x) strcmp(x,labels_of_interest{i}), dat_struct.ID));
    if isempty(this_neuron)
        continue
    end
    neurons_of_interest(i) = this_neuron;
end

% Look at neuron differences and a spiking one
all_traces = zeros(length(neurons_of_interest), size(dat_struct.traces,1));
for i=1:length(neurons_of_interest)
    if neurons_of_interest(i)==0
        continue
    end
%     plot_colored(dat_struct.traces(:,neurons_of_interest(i)),...
%         dat_struct.SevenStates,dat_struct.SevenStatesKey);
%     figure;
%     plot(dat_struct.traces(:,neurons_of_interest(i)))
%     title(sprintf('Neuron ID: %s',...
%         dat_struct.ID{neurons_of_interest(i)}))
    all_traces(i,:) = dat_struct.traces(:,neurons_of_interest(i));
end
% Get rid of neurons that weren't in this dataset
bad_ind = (neurons_of_interest==0);
all_traces(bad_ind,:) = [];
labels_of_interest(bad_ind) = [];
% Small filter
all_traces = CElegansModel.flat_filter(all_traces', 5)';

% Get all possible pairwise differences
sz = size(all_traces);
% all_diff_sz = 0;
% for i = 1:(sz(1)-1)
%     all_diff_sz = all_diff_sz + i;
% end
% all_diffs = zeros(all_diff_sz, sz(2));
all_diffs = [];
all_names = [];
for i = 1:sz(1)
    for j = (i+1):sz(1)
        if isempty(all_diffs)
            all_diffs = all_traces(i,:) - all_traces(j,:);
        else
            all_diffs = [all_diffs; all_traces(i,:) - all_traces(j,:)];
        end
        if isempty(all_names)
            all_names = {[labels_of_interest{i} '-' labels_of_interest{j}]};
        else
            all_names = [all_names ...
                {[labels_of_interest{i} '-' labels_of_interest{j}]} ];
        end
    end
end

% Get the cross correlation between all of these differences and the
% spiking neurons, i.e. SMDXX
all_dat_and_traces = [all_diffs' all_traces'];
all_names_with_diffs = [all_names labels_of_interest];

all_corrs = corrcoef(all_dat_and_traces);
fig = figure;
imagesc(all_corrs)
yticks(1:size(all_corrs,1))
yticklabels(all_names_with_diffs)
xticks(1:size(all_corrs,1))
xticklabels(all_names_with_diffs)
xtickangle(90)

c = fig.Children.Children;
t = 1:sz(2);
c.ButtonDownFcn = @(~,x) ...
    evalin('base',sprintf(...
    'figure;plot(all_dat_and_traces(:,%d));hold on;plot(all_dat_and_traces(:,%d));',...
    round(x.IntersectionPoint(1)),round(x.IntersectionPoint(2))));
% c.ButtonDownFcn = @(~,x) ...
%     evalin('base',sprintf(...
%     'figure;plot(all_dat_and_traces(:,%d));hold on;plot(all_dat_and_traces(:,%d)); legend([{%s}, {%s}])',...
%     round(x.IntersectionPoint(1)),round(x.IntersectionPoint(2)),...
%     all_names_with_diffs{round(x.IntersectionPoint(1))},...
%     all_names_with_diffs{round(x.IntersectionPoint(2))} ));

%==========================================================================


%% Plot the histograms of neuron errors across models
all_figs = cell(4,1);

filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'filter_window_dat',0,...
    'use_deriv',false);

% First, a no-control model
settings.lambda_sparse = 0;
settings.global_signal_mode = 'None';
my_model_naive = CElegansModel(filename, settings);

% Second, a global-control only model
settings.lambda_sparse = 0;
settings.global_signal_mode = 'ID_binary';
my_model_global = CElegansModel(filename, settings);

% Third, a global+sparse control model
settings = rmfield(settings,'lambda_sparse');
settings.global_signal_mode = 'ID_binary';
my_model_full = CElegansModel(filename, settings);

% Fourth, a model that uses filtering and the derivatives
settings.use_deriv = true;
settings.normalize_deriv = true;
settings.filter_window_dat = 3;
settings.global_signal_mode = 'ID_binary';
my_model_deriv = CElegansModel(filename, settings);

% Now plot the histograms of ALL neurons 
corr_naive = my_model_naive.calc_correlation_matrix();
corr_global = my_model_global.calc_correlation_matrix();
corr_full = my_model_full.calc_correlation_matrix();
corr_deriv = my_model_deriv.calc_correlation_matrix();

figure;
bin_width = 0.05;
subplot(4,1,1)
histogram(corr_naive, 'BinWidth',bin_width)
xlim([0,1]);ylim([0,25])
title('All neurons: No control')
subplot(4,1,2)
histogram(corr_global, 'BinWidth',bin_width)
xlim([0,1]);ylim([0,25])
title('Add global state labels')
subplot(4,1,3)
histogram(corr_full, 'BinWidth',bin_width)
xlim([0,1]);ylim([0,25])
title('Add sparse control signals')
subplot(4,1,4)
histogram(corr_deriv, 'BinWidth',bin_width)
xlim([0,1]);ylim([0,25])
title('Add derivatives and filter the data')
xlabel('Correlation coefficient (data and reconstruction)')

% Now just do the named neurons
names = my_model_naive.AdaptiveDmdc_obj.get_names([],[],false,false);
ind = ~cellfun(@isempty, names);

all_figs{1} = figure;
bin_width = 0.05;
histogram(corr_naive(ind), 'BinWidth',bin_width)
xlim([0,1]);ylim([0,10])
title('Named neurons: No control')
all_figs{2} = figure;
histogram(corr_global(ind), 'BinWidth',bin_width)
xlim([0,1]);ylim([0,10])
title('Add global state labels')
all_figs{3} = figure;
histogram(corr_full(ind), 'BinWidth',bin_width)
xlim([0,1]);ylim([0,10])
title('Add sparse control signals')
all_figs{4} = figure;
histogram(corr_deriv(ind), 'BinWidth',bin_width)
xlim([0,1]);ylim([0,10])
title('Add derivatives and filter the data')
xlabel('Correlation coefficient (data and reconstruction)')

% Save figures
if to_save
    for i = 1:length(all_figs)
        fname = sprintf('%sfigure_s5_%d', foldername, i);
        this_fig = all_figs{i};
        if isempty(this_fig)
            continue
        end
        set(this_fig, 'Position', get(0, 'Screensize'));
        saveas(this_fig, fname, 'png');
        matlab2tikz('figurehandle',this_fig,'filename',[fname '_named_neurons.tex']);
    end
end
%==========================================================================


%% Basic: Use Slow Feature Analysis on the data
filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
dat_struct = importdata(filename);

% Linear sfa
y1 = sfa1(dat_struct.traces);

% Quadratic sfa
y2 = sfa2(dat_struct.traces);

% Plot
i = 1;

figure;
subplot(2,1,1)
plot(y1(:,i))
title(sprintf('Feature %d (linear sfa)', i))
subplot(2,1,2)
plot(y2(:,i))
title(sprintf('Feature %d (quadratic sfa)', i))
% Plot colored
plot_colored(y1(:,i), dat_struct.SevenStates, dat_struct.SevenStatesKey);

plot_colored(y1(:,1:3), dat_struct.SevenStates, dat_struct.SevenStatesKey);
%==========================================================================


%% Basic: Use Slow Feature Analysis on RPCA filtered data
filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
dat_struct = importdata(filename);

% Use model as filter
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'filter_window_dat',0,...
    'use_deriv',false);

% Don't bother with state labels
settings.global_signal_mode = 'None';
my_model_filter = CElegansModel(filename, settings);

% Linear sfa
[y, hdl] = sfa1(my_model_filter.dat');

% Quadratic sfa
% y2 = sfa2(dat_struct.traces);

% Plot
% i = 1;
%  
% figure;
% subplot(2,1,1)
% plot(y1(:,i))
% title(sprintf('Feature %d (linear sfa)', i))
% subplot(2,1,2)
% plot(y2(:,i))
% title(sprintf('Feature %d (quadratic sfa)', i))
% Plot colored
plot_colored(y1(:,i), dat_struct.SevenStates, dat_struct.SevenStatesKey);

plot_colored(y1(:,1:3), dat_struct.SevenStates, dat_struct.SevenStatesKey);
%==========================================================================


%% Network: Use Slow Feature Analysis on RPCA filtered data
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);

% Use model as filter
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'filter_window_dat',2,...
    'lambda_sparse', 0.04,...
    'use_deriv',false);

% Don't bother with state labels
settings.global_signal_mode = 'None';
my_model_filter = CElegansModel(filename, settings);

% 1st Linear sfa (kind of like PCA)
[y1, hdl] = sfa1(my_model_filter.dat(:,5:end-4)');

% 2nd: Quadratic sfa
r = 40;
y2 = sfa2(y1(:,1:r));

% 3rd: Quadratic sfa
y3 = sfa2(y2(:,1:r));

% 4th: Quadratic sfa
y4 = sfa2(y3(:,1:r));
% 5th-8th: Quadratic sfa
y5 = sfa2(y4(:,1:r));
y6 = sfa2(y5(:,1:r));
y7 = sfa2(y6(:,1:r));
y8 = sfa2(y7(:,1:r));

% Plot
i = 1;

figure;
subplot(2,1,1)
plot(y1(:,i))
title(sprintf('Feature %d (linear sfa)', i))
subplot(2,1,2)
plot(y2(:,i))
title(sprintf('Feature %d (quadratic sfa)', i))

% Plot colored
% plot_colored(y1(:,i), dat_struct.SevenStates, dat_struct.SevenStatesKey);

plot_colored(y1(:,1:3), dat_struct.SevenStates, dat_struct.SevenStatesKey);
title('First sfa1')
plot_colored(y2(:,1:3), dat_struct.SevenStates, dat_struct.SevenStatesKey);
title('Second sfa1-2')
plot_colored(y1(:,1:3), dat_struct.SevenStates, dat_struct.SevenStatesKey);
title('Third sfa1-2-2')
%==========================================================================


%% Quick demo video
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);
dat = dat_struct.traces';
n = size(dat,2);
num_neurons = size(dat,1);

% To save
movie_filename = '';
if ~isempty(movie_filename)
    video_obj = VideoWriter(movie_filename);
    open(video_obj);
end

% SVD
[u, ~, ~] = svd(dat);
modes = u(:,1:3);
proj3d = (modes.')*dat;

% Make plots (loop)
fig = figure;
subplot(2,1,1)
imagesc(dat)
hold on
% subplot(2,1,2)
% plot3(proj3d(1,:), proj3d(2,:), proj3d(3,:))
% plot(proj3d(1,:), proj3d(2,:))
% hold on

pause

for i = 1:n
    subplot(2,1,1)
    line([i i], [0 num_neurons])
    subplot(2,1,2)
%     plot3(proj3d(1,i), proj3d(2,i), proj3d(3,i), '*k', 'LineWidth', 3)
    plot(proj3d(1,1:i), proj3d(2,1:i), 'k', 'LineWidth', 2)
    hold on
    drawnow
%     pause(0.1)
    if ~isempty(movie_filename)
        frame = getframe(fig);
        writeVideo(video_obj, frame);
    end
    c = fig.Children(2).Children;
    delete(c(1));
end

if ~isempty(movie_filename)
    close(video_obj);
end

%==========================================================================


%% Visualize with demixed PCA
folder_name = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\npr1_1_PreLet\AN20140730a_ZIM575_PreLet_6m_O2_21_s_1TF_47um_1330_\';
filename = [folder_name 'wbdataset.mat'];
dat_struct = importdata(filename);
dat = dat_struct.traces';
time = dat_struct.timeVectorSeconds;

% Recast toy data in the correct format
% trialNum: N x S x D
% firingRates: N x S x D x T x maxTrialNum
% firingRatesAverage: N x S x D x T
%
% N is the number of neurons
% S is the number of stimuli conditions (F1 frequencies in Romo's task)
% D is the number of decisions (D=2)
% T is the number of time-points (note that all the trials should have the
% same length in time!)
%
% NOTE: here we don't have any "decision" columns

N = size(dat, 1); 
S = 1; % Same O2 stimulus every time
T = length(time);

% Just have one trial for each neuron for now
trialNum = ones(N, S);
% "firingRates" is actually a graded potential
firingRates = reshape(dat, [N, S, T, 1]);
% Only one trial, so no average needed
firingRatesAverage = reshape(dat, [N, S, T]);

% Define parameter grouping
% For two parameters (e.g. stimulus and time, but no decision), we would have
% firingRates array of [N S T E] size (one dimension less, and only the following
% possible marginalizations:
%    1 - stimulus
%    2 - time
%    [1 2] - stimulus/time interaction
combinedParams = {{1, [1 2]}, {2}};
margNames = {'Stimulus', 'S/D Interaction', 'Condition-independent'};
margColours = [23 100 171; 187 20 25; 114 97 171]/256;

% Time events of interest (e.g. stimulus onset/offset, cues etc.)
% They are marked on the plots with vertical lines
timeEvents = time(round(length(time)/2));

% check consistency between trialNum and firingRates
% for n = 1:size(firingRates,1)
%     for s = 1:size(firingRates,2)
%         for d = 1:size(firingRates,3)
%             assert(isempty(find(isnan(firingRates(n,s,d,:,1:trialNum(n,s,d))), 1)), 'Something is wrong!')
%         end
%     end
% end

% Step 1: PCA of the dataset

X = firingRatesAverage(:,:);
X = bsxfun(@minus, X, mean(X,2));

[W,~,~] = svd(X, 'econ');
W = W(:,1:20);

% minimal plotting
dpca_plot(firingRatesAverage, W, W, @dpca_plot_default);

% computing explained variance
explVar = dpca_explainedVariance(firingRatesAverage, W, W, ...
    'combinedParams', combinedParams);

% a bit more informative plotting
dpca_plot(firingRatesAverage, W, W, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours);


% Step 2: PCA in each marginalization separately

dpca_perMarginalization(firingRatesAverage, @dpca_plot_default, ...
   'combinedParams', combinedParams);

% Step 3: dPCA without regularization and ignoring noise covariance

% This is the core function.
% W is the decoder, V is the encoder (ordered by explained variance),
% whichMarg is an array that tells you which component comes from which
% marginalization

tic
[W,V,whichMarg] = dpca(firingRatesAverage, 20, ...
    'combinedParams', combinedParams);
toc

explVar = dpca_explainedVariance(firingRatesAverage, W, V, ...
    'combinedParams', combinedParams);

dpca_plot(firingRatesAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3, ...
    'legendSubplot', 16);


% Step 4: dPCA with regularization

% This function takes some minutes to run. It will save the computations 
% in a .mat file with a given name. Once computed, you can simply load 
% lambdas out of this file:
%   load('tmp_optimalLambdas.mat', 'optimalLambda')

% Please note that this now includes noise covariance matrix Cnoise which
% tends to provide substantial regularization by itself (even with lambda set
% to zero).

optimalLambda = dpca_optimizeLambda(firingRatesAverage, firingRates, trialNum, ...
    'combinedParams', combinedParams, ...
    'simultaneous', ifSimultaneousRecording, ...
    'numRep', 2, ...  % increase this number to ~10 for better accuracy
    'filename', 'tmp_optimalLambdas.mat');

Cnoise = dpca_getNoiseCovariance(firingRatesAverage, ...
    firingRates, trialNum, 'simultaneous', ifSimultaneousRecording);

[W,V,whichMarg] = dpca(firingRatesAverage, 20, ...
    'combinedParams', combinedParams, ...
    'lambda', optimalLambda, ...
    'Cnoise', Cnoise);

explVar = dpca_explainedVariance(firingRatesAverage, W, V, ...
    'combinedParams', combinedParams);

dpca_plot(firingRatesAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3,           ...
    'legendSubplot', 16);




%==========================================================================


