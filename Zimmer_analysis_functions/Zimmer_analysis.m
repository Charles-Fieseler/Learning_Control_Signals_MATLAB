

%% Do the Robinson analysis for the Zimmer data

%---------------------------------------------
% Do the windowed DMD
%---------------------------------------------
%Settings for the windowDMD function
cutoff = 0.1;
windowSize = 200;
windowStep = 20;
%Pick out 'interesting' modes: not too fast or slow, and high energy
sortFunc = @(x,omega,xbar,sigma) ...
    find( (x > 1.1*xbar)' .* ...
    (abs(omega) > (max(abs(omega))*0.05)) .*...
    (abs(omega) < (max(abs(omega))*0.95)) ); %Throw out very fast AND very slow

windowSet = struct('sortFunc',sortFunc,...
    'windowSize',windowSize,'windowStep',windowStep,...
    'libUseCoeffSign',false,'libUseAbs',false,...
    'toSubtractMean', true);

%Power spectra from data with slow modes subtracted
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt3/wbdataset.mat';
dat = importdata(filename);
windowdat1 = windowDMD(dat.traces', windowSet);

% Cluster using kmeans or gmm
numClusters = 3;
clusterMode = 'gmm';
windowdat1.cluster_library(20, numClusters, clusterMode)


%---------------------------------------------
% Produce the plots
%---------------------------------------------
windowdat1.plot_power_classes(true, true)

whichModes = 1:numClusters; %Showing the centroids of all the modes; a little messy
windowdat1.plot_centroids(whichModes);

%---------------------------------------------
% Explore modes with a GUI
%---------------------------------------------
windowdat1.plot_power_and_data(1, true, true)
%==========================================================================


%% Alternative analysis using hand-labeled behavior data
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt2/wbdataset.mat';
settings = struct();
dat_patch1 = patchDMD(filename, settings);

dat_patch1.calc_all_label_similarities();
%==========================================================================


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


%% Separate data into subsets with different types of dynamics
% i.e. interneurons, motor, and sensory

filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
settings = struct();
dat_hierarchy = hierarchicalDMD(filename, settings);

%==========================================================================


%% patchDMD with longer windows
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
settings = struct('patch_settings',struct('windowSize',30));
dat_patch_long = patchDMD(filename, settings);

dat_patch_long.calc_all_label_similarities();
%==========================================================================


%% hierarchicalDMD with longer patches
% i.e. interneurons, motor, and sensory

filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
patch_settings = struct('patch_settings',struct('window_size',30));
settings = struct('patchDMD_settings', patch_settings);
dat_hierarchy30 = hierarchicalDMD(filename, settings);

% Box plots with reasonable titles
folder_name = 'C:\Users\charl\Documents\Current_work\Presentations\Zimmer_skype\';
to_save = false;

behavior_str = 'FWD';

neuron_class = 'motor';
a = dat_hierarchy30.patch_DMD_obj.(neuron_class).similarity_objects(behavior_str); 
f = a.plot_box(true);
title(sprintf('Similarity between %s neurons in behavior: %s',...
    neuron_class, behavior_str))
if to_save
    savefig(f, [folder_name neuron_class '_FWD_1_outlier']);
end

neuron_class = 'inter';
a = dat_hierarchy30.patch_DMD_obj.(neuron_class).similarity_objects(behavior_str);
f = a.plot_box(true);
title(sprintf('Similarity between %s neurons in behavior: %s',...
    neuron_class, behavior_str))
if to_save
    savefig(f, [folder_name neuron_class '_FWD_1_outlier']);
end

neuron_class = 'sensory';
a = dat_hierarchy30.patch_DMD_obj.(neuron_class).similarity_objects(behavior_str);
f = a.plot_box(false);
title(sprintf('Similarity between %s neurons in behavior: %s',...
    neuron_class, behavior_str))
if to_save
    savefig(f, [folder_name neuron_class '_FWD_outliers_kept']);
end

neuron_class = 'NaN';
a = dat_hierarchy30.patch_DMD_obj.(neuron_class).similarity_objects(behavior_str);
f = a.plot_box(false);
title(sprintf('Similarity between %s neurons in behavior: %s',...
    neuron_class, behavior_str))
if to_save
    savefig(f, [folder_name neuron_class '_FWD_0_outliers']);
end
%==========================================================================


%% Small plots to illustrate single neurons

to_save = false;
folder_name = 'C:\Users\charl\Documents\Current_work\Presentations\Zimmer_skype\';

if ~exist('dat1','var')
    filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
    dat1 = importdata(filename);
end
f = figure('DefaultAxesFontSize',14);
plot(dat1.traces(:,68)')
title('Time series for neuron 68 (RIMR)')
savefig(f, [folder_name 'neuron_trace_RIMR']);

f = figure('DefaultAxesFontSize',14);
plot(dat1.traces(:,72)')
title('Time series for neuron 72 (RIML)')
savefig(f, [folder_name 'neuron_trace_RIML']);

filename = 'C:\Users\charl\Documents\MATLAB\Collaborations/Zimmer_data/WildType_adult/simplewt2/wbdataset.mat';
dat2 = importdata(filename);
f = figure('DefaultAxesFontSize',14);
plot(dat1.traces(:,87)')
title('Time series for neuron 87 (RIVL)')
savefig(f, [folder_name 'neuron_trace_RIVL']);


%==========================================================================


%% patchDMD using derivatives
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt3/wbdataset.mat';
settings = struct('patch_settings',struct('windowSize',40), ...
    'use_derivative',true);
dat_patch_long = patchDMD(filename, settings);

dat_patch_long.calc_all_label_similarities();
%==========================================================================


%% hierarchicalDMD on derivatives of data
% i.e. interneurons, motor, and sensory

filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
patch_settings = struct('patch_settings',struct('windowSize',30));
settings = struct('patchDMD_settings', patch_settings,...
    'use_derivatives',true);
dat_hierarchy_deriv = hierarchicalDMD(filename, settings);

% Box plots with reasonable titles
folder_name = 'C:\Users\charl\Documents\Current_work\Presentations\Zimmer_skype\';
to_save = false;

behavior_str = 'FWD';

neuron_class = 'motor';
a = dat_hierarchy_deriv.patch_DMD_obj.(neuron_class).similarity_objects(behavior_str); 
f = a.plot_box(true);
title(sprintf('Similarity between %s neurons in behavior: %s',...
    neuron_class, behavior_str))
if to_save
    savefig(f, [folder_name neuron_class '_FWD_1_outlier']);
end

neuron_class = 'inter';
a = dat_hierarchy_deriv.patch_DMD_obj.(neuron_class).similarity_objects(behavior_str);
f = a.plot_box(true);
title(sprintf('Similarity between %s neurons in behavior: %s',...
    neuron_class, behavior_str))
if to_save
    savefig(f, [folder_name neuron_class '_FWD_1_outlier']);
end

neuron_class = 'sensory';
a = dat_hierarchy_deriv.patch_DMD_obj.(neuron_class).similarity_objects(behavior_str);
f = a.plot_box(false);
title(sprintf('Similarity between %s neurons in behavior: %s',...
    neuron_class, behavior_str))
if to_save
    savefig(f, [folder_name neuron_class '_FWD_outliers_kept']);
end

neuron_class = 'NaN';
a = dat_hierarchy_deriv.patch_DMD_obj.(neuron_class).similarity_objects(behavior_str);
f = a.plot_box(false);
title(sprintf('Similarity between %s neurons in behavior: %s',...
    neuron_class, behavior_str))
if to_save
    savefig(f, [folder_name neuron_class '_FWD_0_outliers']);
end
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


%% Test data for AdaptiveDmdc
% Saved from a previous run
% filename = './dat_test/test_dat_noise0_1.mat';

[ dat, tspan, dyn_mat, ctr_mat ] = ...
    generate_test_data_DMDc(struct('amp_trigger',2.0));

ad_obj = AdaptiveDmdc(real(dat).');
%==========================================================================


%% Use AdaptiveDmdc to get a control signal from real data
filename = 'C:\Users\charl\Documents\MATLAB\Collaborations/Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
dat = importdata(filename);

X = dat.traces';   
sort_mode = 'DMD_error';
options = struct('sort_mode',sort_mode,...
    'cutoff_multiplier', 1.5);
ad_obj = AdaptiveDmdc(X, options);
%==========================================================================


%% Plot 'control neurons' by patch
if ~exist('dat_patch_long','var')
    filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt2/wbdataset.mat';
    settings = struct('patch_settings',struct('windowSize',30));
    dat_patch_long = patchDMD(filename, settings);
end
    
opt = struct('to_plot_cutoff',false, 'to_plot_data', false);
dat_patch_long.calc_AdaptiveDmdc_all(opt);
dat_patch_long.plot_control_neurons(true);

% Cluster them using kmeans
[idx,C,sumd,D] = dat_patch_long.cluster_control_neurons(2, true);
%==========================================================================


%% Demo of AdaptiveDmdc with outlier detection
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);

settings = struct('sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',true,...
    'to_print_error',true);
[ u_indices, sep_error, original_error, error_mat ] = ...
    AdaptiveDmdc(dat5.traces.',settings);
%==========================================================================


%% Plot 'control neurons' by patch (outlier calculations)
if ~exist('dat_patch_long','var')
    filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt3/wbdataset.mat';
    settings = struct('patch_settings',struct('windowSize',70));
    dat_patch_long = patchDMD(filename, settings);
end
    
opt = struct('to_plot_cutoff',false, 'to_plot_data', false,...
    'to_plot_data_and_outliers',true,...
    'sort_mode','DMD_error_outliers');
dat_patch_long.calc_AdaptiveDmdc_all(opt, true);
dat_patch_long.plot_control_neurons(true);

% Cluster them using kmeans
[idx,C,sumd,D] = dat_patch_long.cluster_control_neurons(2, true);
figure;
imagesc(C')
title('Cluster centroids')
%==========================================================================


%% Adaptive_dmdc (outlier) and PCA of the error matrices
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);

settings = struct('to_plot_cutoff',false,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',true,...
    'to_print_error',true);
ad_obj = AdaptiveDmdc(dat5.traces.',settings);
u_indices = ad_obj.u_indices;
% sep_error
% original_error
error_mat = ad_obj.error_mat;

opt_just_PCA3d = struct('PCA3d',true,'sigma',false);
opt_just_sigma = struct('PCA3d',false,'sigma',true);

plotSVD(dat5.traces.',opt_just_PCA3d);
title('Raw data matrix')
plotSVD(dat5.traces,opt_just_sigma);
title('Raw data matrix')

num_controllers = length(find(u_indices));
filter_window = 10;
filter_error_mat = filter(ones(1,filter_window)/filter_window,1,...
    error_mat(end-num_controllers:end,:));
plotSVD(filter_error_mat,opt_just_PCA3d);
title('Error matrix')
plotSVD(filter_error_mat',opt_just_sigma);
title('Error matrix')
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


%% Adaptive_dmdc (outlier) on raw data and ROBUST PCA of the error matrices
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);

settings = struct('to_plot_cutoff',false,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',true,...
    'to_print_error',true);
ad_obj = AdaptiveDmdc(dat5.traces.',settings);
u_indices = ad_obj.u_indices;
% sep_error
% original_error
error_mat = ad_obj.error_mat;

opt_just_PCA3d = struct('PCA3d',true,'sigma',false);
opt_just_sigma = struct('PCA3d',false,'sigma',true);

num_controllers = length(find(u_indices));
filter_window = 10;
filter_error_mat = filter(ones(1,filter_window)/filter_window,1,...
    error_mat(end-num_controllers:end,:));
% Do robust PCA
[L_error, S_error] = RobustPCA(filter_error_mat);

plotSVD(L_error,opt_just_PCA3d);
title('Error matrix (low rank)')
plotSVD(L_error.',opt_just_sigma);
title('Error matrix (low rank)')

plotSVD(S_error,opt_just_PCA3d);
title('Error matrix (sparse)')
plotSVD(S_error.',opt_just_sigma);
title('Error matrix (sparse)')
%==========================================================================


%% Adaptive_dmdc (outlier) on L and S from Robust PCA
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
lambda = 0.01; % Very low rank of L
[L_raw, S_raw] = RobustPCA(dat5.traces, lambda);
% Adaptive dmdc
settings = struct('to_plot_cutoff',false,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',true,...
    'to_print_error',true);
% Note: low rank of L will make the dynamic matrix only have that many
% non-zero columns
ad_obj_L_raw = AdaptiveDmdc(L_raw.',settings);
u_indices_L_raw = ad_obj_L_raw.u_indices;
% sep_error
% original_error
error_mat_L_raw = ad_obj_L_raw.error_mat;

ad_obj_S_raw = AdaptiveDmdc(S_raw.',settings);
u_indices_S_raw = ad_obj_S_raw.u_indices;
% sep_error
% original_error
error_mat_S_raw = ad_obj_S_raw.error_mat;
%==========================================================================


%% patchDMD on 2 patches (i.e. capture the transition)
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct('use_2_patches',true,...
    'patch_settings',struct('windowSize',130));
dat_patch_long_2patch = patchDMD(filename, settings);

opt = struct('to_plot_cutoff',false, 'to_plot_data', false,...
    'sort_mode','DMD_error_outliers');
dat_patch_long_2patch.calc_AdaptiveDmdc_all(opt);
dat_patch_long_2patch.plot_control_neurons(true);

% Cluster them using kmeans
[idx,C,sumd,D] = dat_patch_long_2patch.cluster_control_neurons(2, true);
%==========================================================================


%% Rank of low-rank component as a function of lambda
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
filter_window = 10;
this_dat = dat5.traces;
this_dat = filter(ones(1,filter_window)/filter_window,1,this_dat);

num_lambda = 10;
lambda_vec = linspace(0.018,0.005,num_lambda);
all_ranks = zeros(size(lambda_vec));
all_mats_L = zeros([size(this_dat) num_lambda]);
all_mats_S = zeros(size(all_mats_L));
for i = 1:num_lambda
    lambda = lambda_vec(i);
    [L, S] = RobustPCA(this_dat,lambda);
    all_ranks(i) = rank(L);
    all_mats_L(:,:,i) = L;
    all_mats_S(:,:,i) = S;
end
figure;
plot(lambda_vec, all_ranks,'LineWidth',2)
ylabel('Rank of low-rank component')
xlabel('Lambda (sparsity penalty)')
title('Rank vs sparsity penalty')

to_show = 5;
plot_2imagesc_colorbar(...
    all_mats_S(:,:,to_show)',...
    all_mats_L(:,:,to_show)','2_figures',...
    sprintf('Sparse, lambda=%.3f',lambda_vec(to_show)),...
    sprintf('Low-rank, rank=%d',all_ranks(to_show)))
[u,s,v,proj3d] = plotSVD(all_mats_S(:,:,to_show),opt_just_sigma);
title('Raw data matrix (sparse)')
%==========================================================================


%% Augment low-rank original neurons with sparse 'control signal' pseudo-neurons
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
this_dat = dat5.traces;

lambda = 0.01; % Chosen based on Pareto front... kind of a guess
[L, S] = RobustPCA(this_dat,lambda);
[u,s,v,proj3d] = plotSVD(S);
s = diag(s);

cutoff_percent = 0.75;
cutoff_ind = 1:find(cumsum(s)/sum(s)>cutoff_percent,1);
control_signal = u(:,cutoff_ind)*s(cutoff_ind,cutoff_ind);
augmented_dat = [L control_signal];
% Adaptive dmdc
settings = struct('to_normalize_envelope', false,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',true,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',true,...
    'to_print_error',true);

ad_obj_aug = AdaptiveDmdc(augmented_dat.',settings);
u_indices_aug = ad_obj_aug.u_indices;
% sep_error
% original_error
error_mat_aug = ad_obj_aug.error_mat;
%==========================================================================


%% Shift the control signal back in time
% Try to predict the full data set using the sparse control signal
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
this_dat = dat5.traces;

%Filter it
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);

% Do robust PCA
lambda = 0.01; % Chosen based on Pareto front... kind of a guess
this_dat = my_filter(this_dat,10);
this_dat = this_dat(10:end,:);
[L, S] = RobustPCA(this_dat,lambda);
[u,s,v,proj3d] = plotSVD(S);
s = diag(s);
% Augment data with top modes from sparse component
cutoff_percent = 0.75;
cutoff_ind = 1:find(cumsum(s)/sum(s)>cutoff_percent,1);
control_signal = u(:,cutoff_ind)*s(cutoff_ind,cutoff_ind);
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',false,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false,...
    'to_plot_data',false,...
    'to_print_error',true);
% shift the control signal up in time a bit
max_shift = 30;
all_ctr_neurons = zeros(max_shift,1);
all_ctr_psuedo_neurons = zeros(max_shift,1);
original_sz = size(this_dat);
for control_shift = 1:max_shift
    augmented_dat = [this_dat ...
        [control_signal(control_shift+1:end,:); ...
        zeros(control_shift,size(control_signal,2))]];
    ad_obj = AdaptiveDmdc(augmented_dat.',settings);
    u_indices_augment = ad_obj.u_indices;
    
    all_ctr_neurons(control_shift) = length(find(u_indices_augment));
    all_ctr_pseudo_neurons(control_shift) = ...
        length(find(u_indices_augment(original_sz(2)+1:end))); %#ok<SAGROW>
end

figure;
hold on
plot(all_ctr_neurons);  
plot(all_ctr_pseudo_neurons);
legend({'All control neurons','Control pseudo-neurons'})
ylabel('Number of neurons')
xlabel('Number of frames control signal is shifted')
%==========================================================================


%% Use DMD as a 1-layer filter
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
this_dat = dat5.traces;
cutoff = 0.001;
DMDSet = struct('toSubtractMean',true,...
    'cutoffFunc',@real,'cutoff',cutoff, 'numLayers', 1);
mrdat = mrDMD(this_dat', DMDSet);

%Plot original data for a single neuron
neur = 93;
layerNum = 1;
showOriginal = true;
mrdat.plot_neuron_foreground(neur,[], layerNum, showOriginal);

% neur = 45;
% layerNum = 1;
% showOriginal = true;
% mrdat.plot_neuron_foreground(neur, layerNum, showOriginal)

% neur = 28;
% layerNum = 1;
% showOriginal = true;
% mrdat.plot_neuron_foreground(neur, layerNum, showOriginal)

numModes = 100;
mrdat.plot_neuron_background(neur, numModes, layerNum, false);

%==========================================================================


%% Mess around with a Kalman filter (no control)
% Trying to filter the data...
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);

settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',true,...
    'to_plot_data',false,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false,...
    'to_print_error',false);
x = dat5.traces.';
ad_obj = AdaptiveDmdc(x,settings);
A_original = ad_obj.A_original;
% x = (L-mean(L,1))';
kalman = dsp.KalmanFilter(...
    'StateTransitionMatrix',A_original,...
    'MeasurementMatrix',eye(size(x,1)),...
    'ProcessNoiseCovariance', 0.001*eye(size(x,1)),...
    'MeasurementNoiseCovariance', 0.1*eye(size(x,1)),...
    'InitialStateEstimate', zeros(size(x,1),1),...
    'InitialErrorCovarianceEstimate', 1*eye(size(x,1)),...
    'ControlInputPort',false); %Create Kalman filter
est_x = zeros(size(x));
for i=1:length(x)
    est_x(:,i) = kalman.step(x(:,i));
end

plot_2imagesc_colorbar(x,est_x,'2 1')

interesting_neurons = [103, 33, 106];
for i=interesting_neurons
    figure;
    title(sprintf('Neuron %d',i))
    plot(est_x(i,:)); 
    hold on; 
    plot(x(i,:))
    legend({'Estimate','Original data'})
end
%==========================================================================


%% Kalman filter (WITH control)
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
% my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
% this_dat = my_filter(dat5.traces,3).';
% this_dat = this_dat(:,10:end);
this_dat = dat5.traces.';
% Adaptive dmdc to get the error signals
settings = struct('to_normalize_envelope', false,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',true,...
    'to_plot_data',false,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false,...
    'to_print_error',false);
ad_obj = AdaptiveDmdc(this_dat,settings);

% Redo, trying to reconstruct the entire original data set with just the
% error matrix from the above run
settings = struct('to_normalize_envelope', false,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',false,...
    'to_plot_data_and_outliers',false,...
    'sort_mode','user_set',...
    'x_indices',1:size(this_dat,1),...
    'to_print_error',false,...
    'to_plot_data',false,...
    'min_number_outliers',ad_obj.u_len);
augmented_dat = [this_dat(:,2:end);...
    ad_obj.error_mat(ad_obj.x_len+1:end,:)];
ad_obj_augment = AdaptiveDmdc(augmented_dat,settings);

x = ad_obj_augment.dat;
A_original = ad_obj_augment.A_original;

% Create kalman filters
kalman_no_control = dsp.KalmanFilter(...
    'StateTransitionMatrix',A_original,...
    'MeasurementMatrix',eye(size(x,1)),...
    'ProcessNoiseCovariance', 0.001*eye(size(x,1)),...
    'MeasurementNoiseCovariance', 0.1*eye(size(x,1)),...
    'InitialStateEstimate', zeros(size(x,1),1),...
    'InitialErrorCovarianceEstimate', 1*eye(size(x,1)),...
    'ControlInputPort', false);

dat_sz = ad_obj_augment.x_len;
x_only_dat = ad_obj_augment.dat(1:dat_sz,:);
u = ad_obj_augment.dat(dat_sz+1:end,:);
A_only_dat = ad_obj_augment.A_original(1:dat_sz,1:dat_sz);
B_only_control = ad_obj_augment.A_original(1:dat_sz,dat_sz+1:end);
kalman_w_control = dsp.KalmanFilter(...
    'StateTransitionMatrix',A_only_dat,...
    'MeasurementMatrix',eye(dat_sz),...
    'ProcessNoiseCovariance', 0.001*eye(dat_sz),...
    'MeasurementNoiseCovariance', 0.1*eye(dat_sz),...
    'InitialStateEstimate', zeros(dat_sz,1),...
    'InitialErrorCovarianceEstimate', 1*eye(dat_sz),...
    'ControlInputMatrix', B_only_control);

est_x_w_control_init = zeros(dat_sz,size(x,2));
est_x_w_control_init(:,1) = x_only_dat(:,1);
est_x_no_control_init = zeros(size(x));
est_x_no_control_init(:,1) = x(:,1);
for i=2:length(x)
    est_x_no_control_init(:,i) = ...
        kalman_no_control.step(est_x_no_control_init(:,i-1));
    est_x_w_control_init(:,i) = ...
        kalman_w_control.step(est_x_w_control_init(:,i-1), u(:,i-1));
end

% plot_2imagesc_colorbar(x,est_x_no_control,'2 1')
plot_2imagesc_colorbar(x,est_x_no_control_init,'2 1')
plot_2imagesc_colorbar(x,est_x_w_control_init,'2 1')

interesting_neurons = [15, 44, 55];
for i=interesting_neurons
    figure;
    hold on;
    plot(est_x_w_control_init(i,:));
    plot(x(i,:))
    plot(est_x_no_control_init(i,:)); 
    title(sprintf('Neuron %d',i))
    legend({'Estimate (w/control)','Original data','Estimate (w/o control)'})
end
%==========================================================================


%% Use reconstruction error
% I don't actually care about the L2 error, but rather if we can
% reconstruct!

filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);

%First, no control
my_plotter = DMDplotter(dat5.traces',struct('useOptdmd',false));
X_approx = my_plotter.get_reconstruction();
% The reconstruction is very bad
plot_2imagesc_colorbar(dat5.traces',real(X_approx),'2 1')

% Also try a reconstruction directly from the matrix
x0_data = my_plotter.raw(:,1);
all_x = zeros(length(x0),length(tspan));
all_x(:,1) = my_plotter.Phi\x0_data;
tspan = 1:3021;
for i=2:length(tspan)
    all_x(:,i) = A*all_x(:,i-1);
end
% for jM=1:size(all_x,1)
%     all_x(jM,:) = all_x(jM,:) + mean(my_plotter.raw(jM,:));
% end
plot_2imagesc_colorbar(dat5.traces',real(my_plotter.Phi*all_x),'2 1')

% Compare the two approximations
% plot_2imagesc_colorbar(real(my_plotter.Phi*all_x),real(X_approx),'2 1')
figure; 
imagesc(real(my_plotter.Phi*all_x)-real(X_approx));
colorbar
title('Difference between the two approximations')
%==========================================================================


%% Reconstruct data with and without control (all methods)
% Adaptive dmdc
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',true,...
    'to_plot_data',false,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false,...
    'to_print_error',false);
ad_obj = AdaptiveDmdc(dat5.traces.',settings);

dat_approx_control = ad_obj.plot_reconstruction(true,true);
ad_obj.plot_reconstruction(false);

er = @(x,y)norm(x - y)/numel(x);
er_print = @(x,y) ...
    fprintf('Error in reconstruction with control: %f\n', er(x,y));
er_print(dat_approx_control,ad_obj.dat);

% Redo it with all sort modes and get the error
all_sort_modes = {'random', 'sparsePCA', 'DMD_error', 'DMD_error_normalized'};

settings = struct('to_normalize_envelope', false,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',false,...
    'to_plot_data_and_outliers',false,...
    'to_print_error',false,...
    'to_plot_data',false,...
    'min_number_outliers',ad_obj.u_len);
for i=1:length(all_sort_modes)
    settings.sort_mode = all_sort_modes{i};
    ad_obj_rand = AdaptiveDmdc(dat5.traces.',settings);
    approx_rand = ad_obj_rand.plot_reconstruction(true,true);
    fprintf('(sort mode: %s)\n',all_sort_modes{i})
    er_print(approx_rand,ad_obj_rand.dat);
end
%==========================================================================


%% Reconstruct ALL of data with augmented control signals
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat5.traces,3).';
this_dat = this_dat(:,10:end);
% Adaptive dmdc to get the error signals
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',true,...
    'to_plot_data',false,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false,...
    'to_print_error',false);
ad_obj = AdaptiveDmdc(this_dat,settings);

er = @(x,y)norm(x - y)/numel(x);
er_print = @(x,y) ...
    fprintf('Error in reconstruction with control: %f\n', er(x,y));

disp('(Partial data)')
approx_rand = ad_obj.plot_reconstruction(true,true);
er_print(approx_rand,ad_obj.dat);

% Redo, trying to reconstruct the entire original data set with just the
% error matrix from the above run
settings = struct('to_normalize_envelope', false,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',false,...
    'to_plot_data_and_outliers',false,...
    'sort_mode','user_set',...
    'x_indices',1:size(this_dat,1),...
    'to_print_error',false,...
    'to_plot_data',false,...
    'min_number_outliers',ad_obj.u_len);

augmented_dat = [this_dat(:,2:end);...
    ad_obj.error_mat(ad_obj.x_len+1:end,:)];
ad_obj_augment = AdaptiveDmdc(augmented_dat,settings);
approx_rand = ad_obj_augment.plot_reconstruction(true,true);
disp('(Full data)')
er_print(approx_rand,ad_obj_augment.dat);

interesting_neurons = [33, 58, 44, 15];
for i=interesting_neurons
    ad_obj_augment.plot_reconstruction(true,true,true,i);
end
%==========================================================================


%% Shift the control signal (DMD error matrix) back in time
% Try to predict the full data set using the sparse control signal
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
this_dat = dat5.traces;

%Filter it
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);

% Do robust PCA
lambda = 0.01; % Chosen based on Pareto front... kind of a guess
this_dat = my_filter(this_dat,10);
this_dat = this_dat(10:end,:)';
% Adaptive dmdc to get the error signals
settings = struct('to_normalize_envelope', false,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',true,...
    'to_plot_data',false,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false,...
    'to_print_error',false);
ad_obj = AdaptiveDmdc(this_dat,settings);
% shift the control signal up in time a bit
max_shift = 20;
all_errors = zeros(max_shift,1);
all_objects = cell(size(all_errors));
original_sz = size(this_dat);
% Settings for manually determining the controller set
settings.x_indices = 1:size(this_dat,1);
settings.to_plot_cutoff = false;
settings.sort_mode = 'user_set';
settings.x_indices = 1:ad_obj.x_len;

control_signal = ad_obj.error_mat(ad_obj.x_len+1:end,:);
for control_shift = 1:max_shift
    augmented_dat = [this_dat; ...
        [control_signal(:,control_shift+1:end) ...
        zeros(size(control_signal,1),control_shift+1)]];
    all_objects{control_shift} = AdaptiveDmdc(augmented_dat,settings);
    
    all_errors(control_shift) = ...
        all_objects{control_shift}.calc_reconstruction_error();
end

figure;
hold on
plot(log(all_errors));  
ylabel('log(Reconstruction error)')
xlabel('Number of frames control signal is shifted')
%==========================================================================


%% Plot reconstructions of individual neurons
% Adaptive dmdc
id_struct = struct('ID',dat5.ID,'ID2',dat5.ID2,'ID3',dat5.ID3);
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',true,...
    'id_struct',id_struct,...
    'to_plot_data',true,...
    'to_plot_data_and_outliers',true,...
    'cutoff_multiplier',3.0);
ad_obj = AdaptiveDmdc(dat5.traces.',settings);
ad_obj.plot_reconstruction(true,true);

interesting_neurons = [33, 58, 70, 14];
for i=interesting_neurons
    ad_obj.plot_reconstruction(true,true,true,i);
end
%==========================================================================


%% Reconstruct ALL of data with augmented control signals
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat5.traces,2).';
this_dat = this_dat(:,10:end);
% Adaptive dmdc to get the error signals
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',false,...
    'to_plot_cutoff',true,...
    'to_plot_data',false,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false,...
    'to_print_error',false);
ad_obj = AdaptiveDmdc(this_dat,settings);
% Redo, trying to reconstruct the entire original data set with just the
% error matrix from the above run
settings = struct('to_normalize_envelope', false,...
    'to_subtract_mean',false,...
    'to_plot_cutoff',false,...
    'to_plot_data_and_outliers',false,...
    'sort_mode','user_set',...
    'x_indices',1:size(this_dat,1),...
    'to_print_error',false,...
    'to_plot_data',false,...
    'min_number_outliers',ad_obj.u_len);

augmented_dat = [this_dat(:,2:end);...
    ad_obj.error_mat(ad_obj.x_len+1:end,:)];
ad_obj_augment = AdaptiveDmdc(augmented_dat,settings);
approx_augment = ad_obj_augment.plot_reconstruction(true,true);

interesting_neurons = [1, 15, 58, 93, 42];
for i=interesting_neurons
    ad_obj_augment.plot_reconstruction(true,true,true,i);
end
%==========================================================================


%% Reconstruct partial data (augment with derivatives)
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat5.traces,3).';
this_deriv = my_filter(dat5.tracesDif,3).';
this_dat = [this_dat(:,10:end); this_deriv(:,9:end)];
% Adaptive dmdc separates out neurons to use as control signals
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',true,...
    'to_plot_data',true,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false);
ad_obj_stable = AdaptiveDmdc(this_dat,settings);

dat_approx_control = ad_obj_stable.plot_reconstruction(true,true);

interesting_neurons = [41, 58, 55, 42];
for i=interesting_neurons
    ad_obj_stable.plot_reconstruction(true,true,true,i);
end
%==========================================================================


%% patchDMD on 2 patches with AdaptiveDmdc
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct('use_2_patches',true,...
    'patch_settings',struct('windowSize',140,'windowStep',10));
dat_patch_long_2patch = patchDMD(filename, settings);

opt = struct('to_plot_cutoff',false, 'to_plot_data', false,...
    'sort_mode','DMD_error_outliers');
dat_patch_long_2patch.calc_AdaptiveDmdc_all(opt);

for i = 1:length(dat_patch_long_2patch.patch_starts)
    ad_obj = dat_patch_long_2patch.AdaptiveDmdc_objects{i};
    if ~isempty(ad_obj)
        dat_approx_control = ad_obj.plot_reconstruction(true,true);
    end
end
%==========================================================================


%% Reconstruct ALL of data (and derivatives) with augmented control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat5.traces,3).';
this_deriv = my_filter(dat5.tracesDif,3).';
this_dat = [this_dat(:,10:end); this_deriv(:,9:end)];
% Adaptive dmdc to get the error signals
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',false,...
    'to_plot_cutoff',true,...
    'to_plot_data',true,...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false,...
    'to_print_error',false,...
    'to_augment_error_signals',true);
ad_obj_augment = AdaptiveDmdc(this_dat,settings);

interesting_neurons = [1, 15, 58, 98, 41];
for i=interesting_neurons
    ad_obj_augment.plot_reconstruction(true,true,true,i);
end
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


%% Reconstruct partial data with different thresholds
filename = '../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat5.traces,3).';
this_dat = this_dat(:,10:end);
this_deriv = my_filter(dat5.tracesDif,3).';
this_deriv = this_deriv(:,9:end);
this_dat = [this_dat; this_deriv];

% Adaptive dmdc separates out neurons to use as control signals
% cutoff_multipliers = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0];
cutoff_multipliers = linspace(1,10,30);
all_errors = zeros(length(cutoff_multipliers),1);
all_u_length = zeros(size(all_errors));
for i = 1:length(cutoff_multipliers)
    settings = struct('to_normalize_envelope', true,...
        'to_subtract_mean',true,...
        'to_plot_nothing',true,...
        'cutoff_multiplier',cutoff_multipliers(i));
    ad_obj_thresholds = AdaptiveDmdc(this_dat,settings);

%     ad_obj_stable.plot_reconstruction(true,true);
%     all_errors(i) = ...
%         ad_obj_stable.calc_reconstruction_error('flat_then_2norm', 0.05);
    all_errors(i) = ...
        ad_obj_thresholds.calc_reconstruction_error();
    all_u_length(i) = ad_obj_thresholds.u_len;
end
figure;
max_error = 1e3;
all_errors(all_errors>max_error) = NaN;
plot(all_u_length, all_errors)
ylabel('Magnitude of error')
xlabel('Number of control neurons')
title('Error vs # of ctr neurons (missing data = divergence of reconstruction)')

% interesting_neurons = [41, 58, 55, 42];
% % interesting_neurons = [41, 58];
% for i=interesting_neurons
%     ad_obj_stable.plot_reconstruction(true,true,true,i);
% end
%==========================================================================


%% Play around with controllability subspaces

% Define control matrices
% this_obj = ad_obj_augment2;
ind = 1:this_obj.x_len;
A_ctr = this_obj.A_original(ind,ind);
B_ctr = this_obj.A_original(ind,this_obj.x_len+1:end);
C_ctr = eye(this_obj.x_len);

[Abar,Bbar,Cbar,T,k] = ctrbf(A_ctr,B_ctr,C_ctr);
fprintf('Number of controllable states: %d/%d\n',...
    sum(k), this_obj.x_len)

%==========================================================================


%% Play with thresholding the A matrix (plot L2 errors, not reconstruction)
% get ad_obj
X = ad_obj.dat;

X1 = X(:,1:end-1);
X2 = X(:,2:end);
A = X2/X1;

err_mat = A*X1-X2;
err_mat = err_mat(1:129,:); % Keep only non-control signals
err = norm(err_mat)/norm(X2);
fprintf('L2 error of full matrix equation is %f\n',err);

% Now threshold the matrix A
tol = linspace(1e-2,1e-1,21);
all_errors = zeros(size(tol));
for i=1:length(tol)
    this_tol = tol(i);
    A_thresh = A;
    A_thresh(abs(A)<this_tol) = 0;
    err_mat_thresh = A_thresh*X1-X2;
    err_mat_thresh = err_mat_thresh(1:129,:); % Keep only non-control signals
%     err_thresh = norm(err_mat_thresh)/numel(err_mat_thresh);
    all_errors(i) = norm(err_mat_thresh)/norm(X2);
%     fprintf('L2 error of thresholded matrix equation is %f\n',err_thresh);
end

% Visualize sparsity pattern
% figure;spy(A_thresh)
figure
plot(tol,all_errors)
hold on
plot(tol, err*ones(size(tol)))
legend('Thresholded matrices','Full inverse')
xlabel('Threshold')
ylabel('Error/norm(X2)')
title('Error vs threshold values')

%==========================================================================


%% Look at reconstructions with a thresholded A matrix
tol = 1e-3; % Very similar to the full fit
% tol = 5e-3; % Decent
% tol = 9e-3; % Pretty bad, but no divergence
ad_obj.set_A_thresholded(tol);
ad_obj.plot_reconstruction(true);
interesting_neurons = [41, 114, 51, 42, 44];
for i=interesting_neurons
    ad_obj.plot_reconstruction(true,true,true,i);
end
figure;spy(ad_obj.A_thresholded)
%==========================================================================


%% Play with thresholding the A matrix (plot reconstruction errors)

ad_obj.reset_threshold();
error_mode = '';
err = ad_obj.calc_reconstruction_error(error_mode);

% Now threshold the matrix A
tol = linspace(1e-3,1e-1,21);
all_errors = zeros(size(tol));
for i=1:length(tol)
    this_tol = tol(i);
    ad_obj.set_A_thresholded(this_tol);
    all_errors(i) = ad_obj.calc_reconstruction_error(error_mode);
end
% If the reconstruction diverges, make sure it doesn't ruin the plot
max_error = 1e3;
all_errors(all_errors>max_error) = NaN;

% Visualize sparsity pattern
% figure;spy(A_thresh)
figure
plot(tol, all_errors, 'o')
hold on
plot(tol, err*ones(size(tol)))
legend('Thresholded matrices','Full inverse')
xlabel('Threshold')
ylabel('Error/norm(X2)')
title('Error vs threshold values')

%==========================================================================


%% Use cvx to do sparse DMD
% Import data
filename = '../../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
this_dat = dat5.traces';
X1 = this_dat(:,1:end-1);
X2 = this_dat(:,2:end);

% Do naive dmd as a check
A_original = X2/X1;

% solver variables
n = size(this_dat,1);
gamma = 0.1;

cvx_begin
    variable A_sparse(n,n)
    minimize( norm(A_sparse*X1-X2,2) + gamma*norm(A_sparse,1) )
cvx_end


%==========================================================================


%% cvx tests: sparsity in matrices
n = 20;
tol = 0.9;
rng default;
X1 = rand(n,5*n);
A_true = rand(n,n);
% Make the matrix and data sparse
X1(abs(X1)<tol) = 0;
A_true(abs(A_true)<tol) = 0;
spread = 3*(2*rand(size(A_true))-1);
spread(abs(spread)<0.5) = 0; % Make sure no entries are noise-level
A_true = A_true.*spread;
% This isn't necessarily sparse and is used as data
noise = 0.5;
min_tol = 2*noise + 1e-6;
X2 = A_true*X1 + normrnd(0.0,noise,size(X1));
gamma_list = linspace(0,noise*10,11);
A_sparse_nnz = zeros(size(gamma_list));
for i=1:length(gamma_list)
    % sparsity term
    gamma = gamma_list(i);
    % Actually solve
    cvx_begin quiet
        variable A_sparse(n,n)
        minimize( norm(A_sparse*X1-X2,2) + ...
            gamma*norm(A_sparse,1))
%          + (1-gamma)*norm(A_sparse,2)
    cvx_end
    % Postprocess a bit because there are some tiny terms that survive
    A_sparse(abs(A_sparse)<min_tol) = 0;
    
    A_sparse_nnz(i) = nnz(A_sparse);
    if  (A_sparse_nnz(i) == nnz(A_true)) ||...
            A_sparse_nnz(i) > A_sparse_nnz(1)
        A_sparse_nnz(i+1:end) = NaN;
        break
    end
    
end
% Also get backslash solution for comparison
A_matlab = X2/X1;
A_matlab(abs(A_matlab)<min_tol) = 0;
% Plot
figure;
plot(gamma_list, nnz(A_true)*ones(size(gamma_list)), 'LineWidth',2)
hold on
plot(gamma_list, A_sparse_nnz, 'o')
plot(gamma_list, nnz(A_matlab)*ones(size(gamma_list)), '--',...
    'LineWidth',2)
title('Number of nnz elements vs. sparsity penalty')
legend('nnz of fit', 'nnz of backslash')

figure;
subplot(2,1,1);
spy(A_sparse);
title(sprintf('Solved with gamma=%.1f',gamma))
subplot(2,1,2);
spy(A_true);
title('True matrix')
plot_2imagesc_colorbar(A_true,A_sparse,'2 1')
%figure;
%subplot(2,1,1);spy(X1);
%subplot(2,1,2);spy(X2);
%==========================================================================


%% cvx tests: sequential thresholding for sparsity in matrices
n = 50;
tol = 0.9;
rng default;
X1 = rand(n,5*n);
A_true = rand(n,n);
% Make the matrix and data sparse
X1(abs(X1)<tol) = 0;
A_true(abs(A_true)<tol) = 0;
spread = 3*(2*rand(size(A_true))-1);
spread(abs(spread)<0.5) = 0; % Make sure no entries are noise-level
A_true = A_true.*spread;
% This isn't necessarily sparse and is used as data
noise = 0.5;
% min_tol = 2*noise + 1e-6;
min_tol = 3e-1;
max_iter = 10;
X2 = A_true*X1 + normrnd(0.0,noise,size(X1));
% gamma_list = linspace(0,noise*10,2);
gamma_list = [0.1];
A_sparse_nnz = zeros(size(gamma_list));
% sparsity_pattern = abs(A_true)==0;
sparsity_pattern = false(size(A_true));
num_nnz = zeros(max_iter,1);
for i=1:length(gamma_list)
    % sparsity term
    gamma = gamma_list(i);
    for i2=1:max_iter
        num_nnz(i2) = numel(A_true) - length(find(sparsity_pattern));
        fprintf('Iteration %d; %d nonzero-entries\n',...
            i2, num_nnz(i2))
        if i2>1 && (num_nnz(i2-1)==num_nnz(i2))
            disp('Stall detected; quitting early')
            break
        end
        % Actually solve
        cvx_begin quiet
            variable A_sparse(n,n)
%             minimize( norm(A_sparse*X1-X2,2) + ...
%                 gamma*norm(A_sparse,1))
            minimize( norm(A_sparse*X1-X2,2) )
            A_sparse(sparsity_pattern) == 0
        cvx_end
        % Postprocess a bit because there are some tiny terms that survive
        sparsity_pattern = abs(A_sparse)<min_tol;
%         A_sparse(abs(A_sparse)<min_tol) = 0;

        A_sparse_nnz(i) = nnz(A_sparse);
        if  (A_sparse_nnz(i) == nnz(A_true)) ||...
                A_sparse_nnz(i) > A_sparse_nnz(1)
            A_sparse_nnz(i+1:end) = NaN;
            break
        end
    end
    A_sparse(abs(A_sparse)<1e-6) = 0;
    
end
% Also get backslash solution for comparison
A_matlab = X2/X1;
A_matlab(abs(A_matlab)<min_tol) = 0;
% Plot
figure;
plot(gamma_list, nnz(A_true)*ones(size(gamma_list)), 'LineWidth',2)
hold on
plot(gamma_list, A_sparse_nnz, 'o')
plot(gamma_list, nnz(A_matlab)*ones(size(gamma_list)), '--',...
    'LineWidth',2)
title('Number of nnz elements vs. sparsity penalty')
legend('nnz of fit', 'nnz of backslash')

figure;
subplot(2,1,1);
spy(A_sparse);
title(sprintf('Solved with gamma=%.1f',gamma))
subplot(2,1,2);
spy(A_true);
title('True matrix')
plot_2imagesc_colorbar(A_true,A_sparse,'2 1')
%figure;
%subplot(2,1,1);spy(X1);
%subplot(2,1,2);spy(X2);
%==========================================================================


%% Reconstruct partial augmented data; sparseDMD
filename = '../../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat5.traces,3).';
use_deriv = false;
if use_deriv
    this_deriv = my_filter(dat5.tracesDif,3).';
    this_dat = [this_dat(:,10:end); this_deriv(:,9:end)];
else
    this_dat = this_dat(:,10:end);
end
% Adaptive dmdc separates out neurons to use as control signals
id_struct = struct('ID',{dat5.ID},'ID2',{dat5.ID2},'ID3',{dat5.ID3});
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',true,...
    'to_plot_cutoff',true,...
    'to_plot_data',true,...
    'dmd_mode','sparse',...
    'sort_mode','DMD_error_outliers',...
    'to_plot_data_and_outliers',false,...
    'id_struct',id_struct);
ad_obj_stable = AdaptiveDmdc(this_dat,settings);

dat_approx_control = ad_obj_stable.plot_reconstruction(true,false);

interesting_neurons = [41, 39, 55, 21];
for i=interesting_neurons
    ad_obj_stable.plot_reconstruction(true,true,true,i);
end
%==========================================================================


%% Do Robust dmd twice to augment dat, then sparsify
filename = '../../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat5.traces,3).';
% Adaptive dmdc separates out neurons to use as control signals
lambda_2nd_list = linspace(0.025,0.045,11);
all_adj_obj = cell(size(lambda_2nd_list));

for i = 1:length(lambda_2nd_list)
    id_struct = struct('ID',{dat5.ID},'ID2',{dat5.ID2},'ID3',{dat5.ID3});
    x_ind = 1:size(this_dat, 1);
    settings = struct('to_normalize_envelope', true,...
        'to_subtract_mean',true,...
        'to_plot_cutoff',true,...
        'to_plot_data',true,...
        'dmd_mode','sparse',...
        'sort_mode','user_set',...
        'x_indices',x_ind,...
        'to_plot_data_and_outliers',false,...
        'id_struct',id_struct);
    all_adj_obj{i} = robustPCA_twice_then_dmdc(...
        false, false, 3, false, lambda_2nd_list(i), settings);
end
% dat_approx_control = ad_obj_augment_sparse.plot_reconstruction(true,false);

% interesting_neurons = [41, 39, 55, 21];
% for i=interesting_neurons
%     ad_obj_augment_sparse.plot_reconstruction(true,true,true,i);
% end
%==========================================================================


%% Compare experimental and dynamic connectomes

settings = struct('to_subtract_diagonal',true);
compare_obj = compare_connectome(ad_obj_augment_sparse, settings);
compare_obj.plot_imagesc()
%==========================================================================


%% Use C elegans model object to explore custom actuations
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

my_model = CElegansModel(filename);
ad_obj = my_model.AdaptiveDmdc_obj;
ad_obj.plot_reconstruction(true,true);

% Try actuating just AVA
my_model.reset_user_control();
my_model.add_manual_control_signal(45, 1, 500:550, 0.2)
my_model.plot_reconstruction_user_control()
title('Custom acuation of AVA')

% Global signals only
my_model.reset_user_control();
my_model.add_partial_original_control_signal([260:262])
my_model.plot_reconstruction_user_control()
title('Global signals only')

% Sparse signals only
my_model.reset_user_control();
my_model.add_partial_original_control_signal(130:(130+129))
my_model.plot_reconstruction_user_control()
title('Sparse signals only')
%==========================================================================


%% Use mean pre-transition control signals only
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

my_model = CElegansModel(filename);
ad_obj = my_model.AdaptiveDmdc_obj;
ad_obj.plot_reconstruction(true,true);

my_model.reset_user_control();
% Get the mean transition control signal
which_label = 'FWD';
[~, ~, control_signal_fwd] = my_model.get_control_signal_during_label(...
    which_label, 1);
which_label = 'REVSUS';
[~, ~, control_signal_revsus] = my_model.get_control_signal_during_label(...
    which_label, 1);
% Add it to the model
num_neurons = my_model.dat_sz(1);
ctr_ind = (num_neurons+1):my_model.total_sz(1);
control_signal_fwd = ...
    [zeros(num_neurons, size(control_signal_fwd,2));...
    control_signal_fwd];
control_signal_fwd_long = repmat(control_signal_fwd, [1,20]);
control_signal_revsus = ...
    [zeros(num_neurons, size(control_signal_revsus,2));...
    control_signal_revsus];
control_signal_revsus_long = repmat(control_signal_revsus, [1,20]);
% Add several instances of the control signal
t_start = 500;
padding = zeros(size(control_signal_fwd,1), 100);
this_signal = ...
    [control_signal_fwd_long, padding,...
    control_signal_revsus_long, padding];
my_model.add_partial_original_control_signal(ctr_ind,...
    this_signal, t_start)

my_model.plot_reconstruction_user_control()
title('Controller using FWD transition signal')
my_model.plot_colored_user_control()
my_model.reset_user_control()

%==========================================================================


%% Look at all of the behaviors in sequence
my_model.reset_user_control();
% Get the mean transition control signal
[~, ~, control_signal_fwd] = my_model.get_control_signal_during_label(...
    'FWD', 1);
[~, ~, control_signal_revsus] = my_model.get_control_signal_during_label(...
    'REVSUS', 1);
[~, ~, control_signal_dt] = my_model.get_control_signal_during_label(...
    'DT', 1);
[~, ~, control_signal_vt] = my_model.get_control_signal_during_label(...
    'VT', 1);
[~, ~, control_signal_slow] = my_model.get_control_signal_during_label(...
    'SLOW', 1);
% Add it to the model
num_neurons = my_model.dat_sz(1);
num_channels = size(control_signal_fwd,2);
ctr_ind = (num_neurons+1):my_model.total_sz(1);
f = @(x) repmat(x, [1, 30]);
pad_func = @(x) [zeros(num_neurons, size(x,2)); x];
control_signal_fwd = f( pad_func( control_signal_fwd ));
control_signal_revsus = f( pad_func( control_signal_revsus ));
control_signal_slow = f( pad_func( control_signal_slow ));
control_signal_vt = f( pad_func( control_signal_vt ));
control_signal_dt = f( pad_func( control_signal_dt ));
% Add several instances of the control signal
t_start = 500;
padding = zeros(size(control_signal_fwd,1), 100);
this_signal = ...
    [control_signal_fwd, padding,...
    control_signal_revsus, padding,...
    control_signal_slow, padding,...
    control_signal_vt, padding,...
    control_signal_dt, padding];
my_model.add_partial_original_control_signal(ctr_ind,...
    this_signal, t_start)

my_model.plot_reconstruction_user_control()
title('Controller using all transition signal')
my_model.plot_colored_user_control()
my_model.reset_user_control()
%==========================================================================


%% "ablate" various neurons (the sparse control signal part)
my_model.reset_user_control();

num_neurons = my_model.dat_sz(1);
% ctr_ind = (num_neurons+1):my_model.total_sz(1);
% No global modes
ctr_ind = (num_neurons+1):(2*num_neurons);

% No ablations, but no global modes
my_model.add_partial_original_control_signal(ctr_ind)
my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('No global modes')
my_model.reset_user_control()

% First do AVA
neurons_to_ablate = [45 46];
ctr_ind_no_AVA = ctr_ind;
ctr_ind_no_AVA(neurons_to_ablate) = [];

my_model.add_partial_original_control_signal(ctr_ind_no_AVA)
my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('AVA ablated')
my_model.reset_user_control()

% Next do AVB
neurons_to_ablate = [72 84];
ctr_ind_no_AVB = ctr_ind;
ctr_ind_no_AVB(neurons_to_ablate) = [];

my_model.add_partial_original_control_signal(ctr_ind_no_AVB)
my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('AVB ablated')
my_model.reset_user_control()

% Next ablate AVB, but with the global modes included
neurons_to_ablate = [72 84];
ctr_ind_no_AVB = (num_neurons+1):my_model.total_sz(1);
ctr_ind_no_AVB(neurons_to_ablate) = [];

my_model.add_partial_original_control_signal(ctr_ind_no_AVB)
my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('AVB ablated; includes global modes')
my_model.reset_user_control()

% Remove the neurons important in the transition into FWD... note that
% these do not really have clear interpretability!
[~, ~, control_signal_fwd] = my_model.get_control_signal_during_label(...
    'FWD', 1);
tol = 0.001;
neurons_to_ablate = find(abs(mean(control_signal_fwd,2))>tol);
ctr_ind_no_FWD = (num_neurons+1):my_model.total_sz(1);
ctr_ind_no_FWD(neurons_to_ablate) = [];

my_model.add_partial_original_control_signal(ctr_ind_no_FWD)
my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('FWD transition neurons ablated')
my_model.reset_user_control()

%==========================================================================


%% ablate various neurons (control signal and neuron connectivity)
my_model.reset_user_control();

num_neurons = my_model.dat_sz(1);
ctr_ind = (num_neurons+1):my_model.total_sz(1);
% No global modes
% ctr_ind = (num_neurons+1):(2*num_neurons);

% No ablations, but no global modes
my_model.add_partial_original_control_signal(ctr_ind)
my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('No global modes')
my_model.reset_user_control()

% First do AVA
neurons_to_ablate = [45 46];
ctr_ind_no_AVA = ctr_ind;
ctr_ind_no_AVA(neurons_to_ablate) = [];

my_model.add_partial_original_control_signal(ctr_ind_no_AVA)
my_model.ablate_neuron(neurons_to_ablate);

my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('AVA ablated')
my_model.reset_user_control()

% Next do AVB
neurons_to_ablate = [72 84];
ctr_ind_no_AVB = ctr_ind;
ctr_ind_no_AVB(neurons_to_ablate) = [];

my_model.add_partial_original_control_signal(ctr_ind_no_AVB)
my_model.ablate_neuron(neurons_to_ablate);

my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('AVB ablated')
my_model.reset_user_control()

% Do SMBD and an unknown couple
neurons_to_ablate = [90 113];% 64 121];
ctr_ind_no_AVB = ctr_ind;
ctr_ind_no_AVB(neurons_to_ablate) = [];

my_model.add_partial_original_control_signal(ctr_ind_no_AVB)
my_model.ablate_neuron(neurons_to_ablate);

my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('SMBD ablated')
my_model.reset_user_control()

% Remove the neurons important in the transition into FWD... note that
% these do not really have clear interpretability!
[~, ~, control_signal_fwd] = my_model.get_control_signal_during_label(...
    'FWD', 1);
tol = 0.0683;
neurons_to_ablate = find(abs(mean(control_signal_fwd,2))>tol);
ctr_ind_no_FWD = (num_neurons+1):my_model.total_sz(1);
ctr_ind_no_FWD(neurons_to_ablate) = [];

my_model.add_partial_original_control_signal(ctr_ind_no_FWD)
my_model.ablate_neuron(neurons_to_ablate);

my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('FWD transition neurons ablated')
my_model.reset_user_control()

%==========================================================================


%% Use CElegansModel with a sparsified matrix
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

settings = struct('dmd_mode','sparse');
my_model = CElegansModel(filename, settings);
ad_obj = my_model.AdaptiveDmdc_obj;
ad_obj.plot_reconstruction(true,true);

%==========================================================================


%% Look at just the global modes or sparse signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
my_model = CElegansModel(filename);

% All global
my_model.add_partial_original_control_signal(130:133)
% my_model.ablate_neuron(neurons_to_ablate);

my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('Global modes 130:133')
my_model.reset_user_control()

% All global, but with a custom signal
ctr_ind = 130:130;
custom_signal = 4*ones(length(ctr_ind),200);
t_start = 500;
my_model.add_partial_original_control_signal(ctr_ind,...
    custom_signal, t_start)
% my_model.ablate_neuron(neurons_to_ablate);

my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('Global modes 130:133')
my_model.reset_user_control()

% All sparse
num_neurons = my_model.dat_sz(1);
% ctr_ind = (num_neurons+1):my_model.total_sz(1);
% No global modes
ctr_ind = (num_neurons+1):(2*num_neurons);
my_model.add_partial_original_control_signal(ctr_ind)
% my_model.ablate_neuron(neurons_to_ablate);

my_model.plot_reconstruction_user_control()
my_model.plot_colored_user_control()
title('Only sparse signals')
my_model.reset_user_control()
%==========================================================================


%% Analyze data and normalized derivatives
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

settings = struct('use_deriv',true,'to_normalize_deriv',true);
my_model_deriv = CElegansModel(filename, settings);
ad_obj = my_model_deriv.AdaptiveDmdc_obj;
ad_obj.plot_reconstruction(true,true);

interesting_neurons = [84, 45, 58, 46, 15, 174, 167];
for i=interesting_neurons
    ad_obj.plot_reconstruction(true,false,true,i);
end
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


%% Predict global signals using DMDc
% I expect that I would really need a nonlinear function here, but why not
% this first?

% Original analysis
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
lambda_global = 0.0055; % rank=3
settings = struct('lambda_global',lambda_global);
my_model = CElegansModel(filename, settings);
% Get new "data set": global modes... predict them with sparse signals
S_sparse = my_model.S_sparse;
L_global = my_model.L_global_modes.';
dat = [L_global; S_sparse(:,1:size(L_global,2))];
% Make new AdaptiveDmdc object
augment_data = 1;
x_indices = 1:(size(L_global,1)*augment_data);
settings = struct(...
    'sort_mode','user_set',...
    'x_indices', x_indices,...
    'augment_data', augment_data);
predict_L_global = AdaptiveDmdc(dat, settings);

predict_L_global.plot_reconstruction(true, false);

for i=x_indices
    predict_L_global.plot_reconstruction(true, true, true, i);
end
%==========================================================================


%% Augment data and look at the fixed points
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
lambda_global = 0.0035;
settings = struct('augment_data', 4, 'max_rank_global', 3, ...
    'lambda_global', lambda_global);
my_model_augment = CElegansModel(filename, settings);

% Plot all fixed points
my_model_augment.plot_colored_fixed_point();

%==========================================================================


%% Export data for use in tensorflow
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'lambda_sparse', 0.035);
my_model_export = CElegansModel(filename, settings);

dat_with_control = my_model_export.dat_with_control;
target = 'C:\Users\charl\Documents\Current_work\Zimmer_nn_predict\dat_with_control';
save(target, 'dat_with_control');

% Save another one with much simpler global modes
settings = struct(...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'lambda_sparse', 0.05,...
    'dmd_mode','func_DMDc',...
    'global_signal_mode','ID');
my_model_export2 = CElegansModel(filename, settings);

dat_with_control = my_model_export2.dat_with_control;
target = 'C:\Users\charl\Documents\Current_work\Zimmer_nn_predict\dat_with_control_ID';
save(target, 'dat_with_control');

% Save another one with much simpler global modes
%   (modes are behavioralist identified signals)
settings = struct(...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'lambda_sparse', 0.05,...
    'dmd_mode','func_DMDc',...
    'global_signal_mode','ID_simple');
my_model_export3 = CElegansModel(filename, settings);

dat_with_control = my_model_export3.dat_with_control;
target = 'C:\Users\charl\Documents\Current_work\Zimmer_nn_predict\dat_with_control_ID_simple';
save(target, 'dat_with_control');

%==========================================================================


%% Use dynamics from worm5 on worm4
% Note: learn the control signal from the worm though
filename5 = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
filename3 = '../../Zimmer_data/WildType_adult/simplewt3/wbdataset.mat';

% Get first model and error
settings = struct('to_subtract_mean_sparse',false);
my_model5 = CElegansModel(filename5, settings);
fprintf('Reconstruction error for worm 5: %f\n',...
    my_model5.AdaptiveDmdc_obj.calc_reconstruction_error());
my_model5.AdaptiveDmdc_obj.plot_reconstruction(true);

% Get second model and initial error
settings = struct('to_subtract_mean_sparse',false);
my_model3 = CElegansModel(filename3, settings);
fprintf('Reconstruction error for worm 3: %f\n',...
    my_model3.AdaptiveDmdc_obj.calc_reconstruction_error());
my_model3.AdaptiveDmdc_obj.plot_reconstruction(true);

% Get the overlapping neurons
names5 = my_model5.AdaptiveDmdc_obj.get_names([], false, false);
names3 = my_model3.AdaptiveDmdc_obj.get_names([], false, false);
% Indices are different for worms 3 and 5
[ind3] = ismember(names3, names5);
ind3 = logical(ind3.*(~strcmp(names3, '')));
[ind5] = ismember(names5, names3);
ind5 = logical(ind5.*(~strcmp(names5, '')));

% Truncate the data and redo the worms
dat_struct5 = importdata(filename5);
dat_struct5.traces(:,~ind5) = [];
for i={'ID','ID2','ID3'}
    tmp = dat_struct5.(i{1});
    tmp(~ind5) = [];
    dat_struct5.(i{1}) = tmp;
end
settings.lambda_global = 0.01;
settings.lambda_sparse = 0.07;
my_model5_truncate = CElegansModel(dat_struct5, settings);
my_model5_truncate.AdaptiveDmdc_obj.plot_reconstruction(true);

dat_struct3 = importdata(filename3);
dat_struct3.traces(:,~ind3) = [];
for i={'ID','ID2','ID3'}
    tmp = dat_struct3.(i{1});
    tmp(~ind3) = [];
    dat_struct3.(i{1}) = tmp;
end
my_model3_truncate = CElegansModel(dat_struct3, settings);
my_model3_truncate.AdaptiveDmdc_obj.plot_reconstruction(true);

% Get errors and apply worm5 dynamics to worm3
fprintf('Reconstruction error for truncated worm 3: %f\n',...
    my_model3_truncate.AdaptiveDmdc_obj.calc_reconstruction_error());
fprintf('Reconstruction error for truncated worm 5: %f\n',...
    my_model5_truncate.AdaptiveDmdc_obj.calc_reconstruction_error());

my_model3_truncate.AdaptiveDmdc_obj.A_original = ...
    my_model5_truncate.AdaptiveDmdc_obj.A_original;
my_model3_truncate.AdaptiveDmdc_obj.A_separate = ...
    my_model5_truncate.AdaptiveDmdc_obj.A_separate;
my_model3_truncate.AdaptiveDmdc_obj.plot_reconstruction(true);
title("Reconstruction using alternate dynamics")

fprintf('Reconstruction error for worm 3 data and worm 5 A matrix: %f\n',...
    my_model3_truncate.AdaptiveDmdc_obj.calc_reconstruction_error());
%==========================================================================


%% Use dynamics from worm5 on worm4, with ID as global modes
% Note: learn the control signal from the worm though
filename5 = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
filename3 = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';

% Get first model and error
settings = struct('to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false);
settings.global_signal_mode = 'ID';
my_model5 = CElegansModel(filename5, settings);
fprintf('Reconstruction error for worm 5: %f\n',...
    my_model5.AdaptiveDmdc_obj.calc_reconstruction_error());
my_model5.AdaptiveDmdc_obj.plot_reconstruction(true);

% Get second model and initial error
my_model3 = CElegansModel(filename3, settings);
fprintf('Reconstruction error for worm 3: %f\n',...
    my_model3.AdaptiveDmdc_obj.calc_reconstruction_error());
my_model3.AdaptiveDmdc_obj.plot_reconstruction(true);

% Get the overlapping neurons
names5 = my_model5.AdaptiveDmdc_obj.get_names([], false, false);
names3 = my_model3.AdaptiveDmdc_obj.get_names([], false, false);
% Indices are different for worms 3 and 5
[ind3] = ismember(names3, names5);
ind3 = logical(ind3.*(~strcmp(names3, '')));
[ind5] = ismember(names5, names3);
ind5 = logical(ind5.*(~strcmp(names5, '')));

% Truncate the data and redo the worms
dat_struct5 = importdata(filename5);
dat_struct5.traces(:,~ind5) = [];
for i={'ID','ID2','ID3'}
    tmp = dat_struct5.(i{1});
    tmp(~ind5) = [];
    dat_struct5.(i{1}) = tmp;
end
settings.lambda_global = 0.01;
settings.lambda_sparse = 0.07;
my_model5_truncate = CElegansModel(dat_struct5, settings);
my_model5_truncate.AdaptiveDmdc_obj.plot_reconstruction(true);

dat_struct3 = importdata(filename3);
dat_struct3.traces(:,~ind3) = [];
for i={'ID','ID2','ID3'}
    tmp = dat_struct3.(i{1});
    tmp(~ind3) = [];
    dat_struct3.(i{1}) = tmp;
end
my_model3_truncate = CElegansModel(dat_struct3, settings);
my_model3_truncate.AdaptiveDmdc_obj.plot_reconstruction(true);

% Get errors and apply worm5 dynamics to worm3
fprintf('Reconstruction error for truncated worm 3: %f\n',...
    my_model3_truncate.AdaptiveDmdc_obj.calc_reconstruction_error());
fprintf('(Persistence model error for truncated worm 3: %f)\n',...
    my_model3_truncate.AdaptiveDmdc_obj.calc_reconstruction_error([],true));
fprintf('Reconstruction error for truncated worm 5: %f\n',...
    my_model5_truncate.AdaptiveDmdc_obj.calc_reconstruction_error());
fprintf('(Persistence model error for truncated worm 5: %f)\n',...
    my_model5_truncate.AdaptiveDmdc_obj.calc_reconstruction_error([],true));

my_model3_truncate.AdaptiveDmdc_obj.A_original = ...
    my_model5_truncate.AdaptiveDmdc_obj.A_original;
my_model3_truncate.AdaptiveDmdc_obj.A_separate = ...
    my_model5_truncate.AdaptiveDmdc_obj.A_separate;
my_model3_truncate.AdaptiveDmdc_obj.plot_reconstruction(true);
title("Reconstruction using alternate dynamics")
% my_model5_truncate.AdaptiveDmdc_obj.A_original = ...
%     my_model3_truncate.AdaptiveDmdc_obj.A_original;
% my_model5_truncate.AdaptiveDmdc_obj.A_separate = ...
%     my_model3_truncate.AdaptiveDmdc_obj.A_separate;
% my_model5_truncate.AdaptiveDmdc_obj.plot_reconstruction(true);
% title("Reconstruction using alternate dynamics")

fprintf('Reconstruction error for worm 3 data and worm 5 A matrix: %f\n',...
    my_model3_truncate.AdaptiveDmdc_obj.calc_reconstruction_error());
%==========================================================================


%% Plot pareto fronts of the sparse signal
% Note: learn the control signal from the worm though
filename5 = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

% Get model and initial error
settings = struct('to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false);
settings.global_signal_mode = 'ID';
my_model_pareto = CElegansModel(filename5, settings);
fprintf('Reconstruction error for worm 5: %f\n',...
    my_model_pareto.AdaptiveDmdc_obj.calc_reconstruction_error());
% my_model_pareto.AdaptiveDmdc_obj.plot_reconstruction(true);

% Calculate pareto front with different types of global mode calculations
% lambda_vec = linspace(0.04,0.06,50);
lambda_vec = linspace(0.05,0.1,25);
% global_signal_mode = {'ID_simple','ID_and_offset'};
% global_signal_mode = {'ID','ID_simple','ID_and_offset','ID_binary'};
global_signal_mode = {'ID_and_offset'};
% global_signal_mode = {'RPCA','ID','ID_simple','ID_and_offset'};
for i = 1:length(global_signal_mode)
    my_model_pareto.calc_pareto_front(lambda_vec, global_signal_mode{i}, (i>1));
end

my_model_pareto.plot_pareto_front();

%==========================================================================


%% Plot pareto fronts with a better DMD backend
% Note: learn the control signal from the worm though
filename5 = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

% Get model and initial error
settings = struct('to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID';
my_model_pareto = CElegansModel(filename5, settings);
fprintf('FULL reconstruction error for worm 5: %f\n',...
    my_model_pareto.AdaptiveDmdc_obj.calc_reconstruction_error());
% my_model_pareto.AdaptiveDmdc_obj.plot_reconstruction(true);

% Calculate pareto front with different types of global mode calculations
lambda_vec = linspace(0.03,0.07,40);
% lambda_vec = linspace(0.05,0.1,2);
% global_signal_mode = {'ID_simple','ID_and_offset'};
global_signal_mode = {'ID','ID_simple','ID_and_offset','ID_binary'};
% global_signal_mode = {'ID_binary'};
% global_signal_mode = {'RPCA','ID','ID_simple','ID_and_offset'};
for i = 1:length(global_signal_mode)
    my_model_pareto.calc_pareto_front(lambda_vec, global_signal_mode{i}, (i>1));
end

my_model_pareto.plot_pareto_front();

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


%% Examine the FP activated by global signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
settings.global_signal_mode = 'ID_and_offset';
my_model_ID_test = CElegansModel(filename, settings);

% Add just the global signals as controls
num_neurons = my_model_ID_test.dat_sz(1);
ctr_ind = (num_neurons+1):(my_model_ID_test.total_sz(1)-num_neurons);
% custom_signal = max(max(my_model.dat_with_control(num_neurons+ctr_ind,:))) *...
%     ones(length(ctr_ind),1000);
% t_start = 500;
is_original_neuron = false;
my_model_ID_test.add_partial_original_control_signal(ctr_ind,...
    [], [], is_original_neuron)
% my_model.ablate_neuron(neurons_to_ablate);
my_model_ID_test.plot_reconstruction_user_control(false);

my_model_ID_test.plot_reconstruction_user_control(true, 45);

my_model_ID_test.plot_user_control_fixed_points('FWD', true);
my_model_ID_test.plot_user_control_fixed_points('SLOW', true);
my_model_ID_test.plot_user_control_fixed_points('REVSUS', true);

%==========================================================================


%% Export sparse control signals from all worms
filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';
export_folder = '../../Zimmer_data/data_export/';
export_template = 'dat_worm_%d';
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
settings.global_signal_mode = 'ID_and_offset';

my_models = cell(5,1);
new_dat = cell(5,1);
for i=1:5
    filename = sprintf(filename_template, i);
    my_models{i} = CElegansModel(filename, settings);
    
    my_models{i}.plot_reconstruction_interactive(false);
    
    num_neurons = my_models{i}.original_sz(1);
    ind = (num_neurons+1):(2*num_neurons);
    new_dat{i} = importdata(filename);
    new_dat{i}.sparse_control = my_models{i}.dat_with_control(ind,:).';
    new_dat{i}.user_control_reconstruction = ...
        my_models{i}.user_control_reconstruction.';
    
    new_name = sprintf(export_template, i);
    S = struct(new_name,new_dat{i});
    export_filename = [export_folder new_name '.mat' ];
    save(export_filename, '-struct', 'S')
end
%==========================================================================


%% Get CElegansModel to work with alternate struct
foldername = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\npr1_1_PreLet\';
% filename = [foldername ...
%     'AN20150917a_ZIM1027_1mMTF_O2_21_s_47um_1345_PreLet_\wbdataset.mat'];
filename = [foldername ...
    'AN20140807d_ZIM575_PreLet_6m_O2_21_s_1mMTF_47um_1540_\wbdataset.mat'];
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID_and_offset';
my_model_other_struct = CElegansModel(filename, settings);

my_model_other_struct.plot_reconstruction_interactive(false);

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


%% Plot pareto fronts with a better backend and projection of explosive modes
% Also now has automatic dimensionality reduction
filename5 = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';

% Get model and initial error
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID'; % Not what is used
my_model_pareto = CElegansModel(filename5, settings);
fprintf('FULL reconstruction error for worm 5: %f\n',...
    my_model_pareto.AdaptiveDmdc_obj.calc_reconstruction_error());
% my_model_pareto.AdaptiveDmdc_obj.plot_reconstruction(true);

% Calculate pareto front with different types of global mode calculations
lambda_vec = linspace(0.03,0.07,40);
% lambda_vec = linspace(0.05,0.1,2);
% global_signal_mode = {'ID_simple','ID_and_offset'};
global_signal_mode = {'ID','ID_simple','ID_and_offset','ID_binary'};
% global_signal_mode = {'ID_binary'};
% global_signal_mode = {'RPCA','ID','ID_simple','ID_and_offset'};
for i = 1:length(global_signal_mode)
    my_model_pareto.calc_pareto_front(lambda_vec, global_signal_mode{i}, (i>1));
end

my_model_pareto.plot_pareto_front();

%==========================================================================


%% Use external pareto front object
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

model_settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
% global_signal_modes = {{'ID'}};
global_signal_modes = {{'ID','ID_binary'}};
% global_signal_modes = {{'ID','ID_simple','ID_binary'}};
lambda_vec = linspace(0.02,0.1,40);
settings = struct(...
    'file_or_dat', filename,...
    'base_settings', model_settings,...
    'iterate_settings',struct('global_signal_mode',global_signal_modes),...
    'x_vector', lambda_vec,...
    'x_fieldname', 'lambda_sparse',...
    'fields_to_plot',{{{'AdaptiveDmdc_obj','calc_reconstruction_error'}}});
    
my_pareto_obj = ParetoFrontObj('CElegansModel', settings);

%==========================================================================


%% Look at the neuron "roles" vis a vis intrinsic kicks
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'to_plot_nothing',true);
settings.global_signal_mode = 'ID';
my_model_ID_test = CElegansModel(filename, settings);

[attractor_overlap, all_ctr_directions] =...
    my_model_ID_test.calc_attractor_overlap();
all_norms = zeros(size(all_ctr_directions,2),1);
for i=1:size(all_ctr_directions,2)
    all_norms(i) = norm(all_ctr_directions(:,i));
end
% Don't plot the "no-state" stuff
figure;
plot(attractor_overlap(:,1:end-1), 'o', 'LineWidth',2);
legend(my_model_ID_test.state_labels_key(1:1:end-1))

prox_overlap = attractor_overlap;
tol = 0.1;
prox_overlap(abs(prox_overlap)<tol) = 0;
figure;
plot(prox_overlap(:,1:end-1), 'o', 'LineWidth',2);
hold on;
plot(all_norms,'ok','LineWidth',2)
legend([my_model_ID_test.state_labels_key(1:1:end-1), {'Kick strength'}])
title('Large instrinsic kicks compared to overall neuron kick strength')

figure;
normalized_overlap = attractor_overlap./all_norms;
tol = 0.2;
normalized_overlap(abs(normalized_overlap)<tol) = 0;
plot(normalized_overlap(:,1:end-1), 'o', 'LineWidth',2);
legend(my_model_ID_test.state_labels_key(1:1:end-1)) 
title('Intrinsic kicks normalized by overall kick strength')

% Now show just the identified neurons
all_names = my_model_ID_test.AdaptiveDmdc_obj.get_names();
known_ind = ~strcmp(all_names,'');
known_names = all_names(known_ind);

figure
normalized_overlap = attractor_overlap./all_norms;
tol = 0.15;
normalized_overlap(abs(normalized_overlap)<tol) = 0;
plot(normalized_overlap(known_ind,1:end-1), 'o', 'LineWidth',2);
xticks(1:length(known_ind));
xticklabels(known_names)
legend(my_model_ID_test.state_labels_key(1:1:end-1)) 
title('Intrinsic kicks normalized by overall kick strength')

% Now do simplified groups of behaviors
simplified_labels_cell = {{'FWD','SLOW'},{'REVSUS','REV1','REV2'}};
all_labels = my_model_ID_test.state_labels_key;
simplified_labels_ind = cell(size(simplified_labels_cell));
simplified_overlap = zeros(...
    size(attractor_overlap,1), length(simplified_labels_cell));
for i=1:length(simplified_labels_cell)
    simplified_labels_ind(i) = ...
        { contains(all_labels,simplified_labels_cell{i}) };
    simplified_overlap(:,i) = mean(...
        attractor_overlap(:,simplified_labels_ind{i}),2);
end

all_norms = zeros(size(all_ctr_directions,2),1);
for i=1:size(all_ctr_directions,2)
    all_norms(i) = norm(all_ctr_directions(:,i));
end
normalized_overlap = simplified_overlap./all_norms;
tol = 0.1;
normalized_overlap(abs(normalized_overlap)<tol) = 0;

all_names = my_model_ID_test.AdaptiveDmdc_obj.get_names();
known_ind = ~strcmp(all_names,'');
known_names = all_names(known_ind);

figure
% plot(normalized_overlap, 'o', 'LineWidth',2);
plot(normalized_overlap(known_ind,:), 'o', 'LineWidth',2);
xticks(1:length(known_ind));
xticklabels(known_names)
legend({'simple FWD','simple REVSUS'}) 
title('Intrinsic kicks normalized by overall kick strength')

% Also plot a baseline for random matrix connectivity
[attractor_overlap, all_ctr_directions,...
    attractor_reconstruction] =...
    my_model_ID_test.calc_attractor_overlap();
rand_ctr_directions = 2*rand(129)-1;
for i=1:size(rand_ctr_directions,2)
    all_norms(i) = norm(rand_ctr_directions(:,i));
end
baseline = (rand_ctr_directions.'./all_norms)*attractor_reconstruction;
figure;
plot(baseline,'LineWidth',2)

%==========================================================================


%% Neuron roles across worms
filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID';

all_models = cell(5,1);
all_roles_dynamics = cell(5,2);
all_roles_centroid = cell(5,2);
all_roles_global = cell(5,2);
for i=1:5
    filename = sprintf(filename_template,i);
    if i==4
        settings.lambda_sparse = 0.035;
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

[ combined_dat_dynamic, all_labels_dynamic ] =...
    combine_different_trials( all_roles_dynamics );
[ combined_dat_centroid, all_labels_centroid ] =...
    combine_different_trials( all_roles_centroid );
[ combined_dat_global, all_labels_global ] =...
    combine_different_trials( all_roles_global );

% Some histograms of how many times a neuron is identified as what role
possible_roles = unique(combined_dat_dynamic);
possible_roles(cellfun(@isempty,possible_roles)) = [];
num_neurons = size(combined_dat_dynamic,1);
role_counts = zeros(num_neurons,length(possible_roles));
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_dynamic, possible_roles{i}),2);
end
% num_runs = size(combined_dat_dynamic,2);
% times_identified = num_runs - sum(strcmp(combined_dat_dynamic, ''),2);
figure('DefaultAxesFontSize',14)
% bar(times_identified, 'k')
% hold on
bar(role_counts, 'stacked')
% legend([{'Number of times identified'};possible_roles])
legend(possible_roles)
xticks(1:num_neurons);
xticklabels(all_labels_dynamic)
xtickangle(90)
title('Neuron roles using similarity to dynamic attractors')

% Some histograms of how many times a neuron is identified as what role
possible_roles = unique(combined_dat_centroid);
possible_roles(cellfun(@isempty,possible_roles)) = [];
num_neurons = size(combined_dat_centroid,1);
role_counts = zeros(num_neurons,length(possible_roles));
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_centroid, possible_roles{i}),2);
end
figure('DefaultAxesFontSize',14)
bar(role_counts, 'stacked')
legend(possible_roles)
xticks(1:num_neurons);
xticklabels(all_labels_centroid)
xtickangle(90)
title('Neuron roles using similarity to data centroids')
% Same but only with neurons ID'ed more than once
this_ind = find(sum(role_counts,2)>1);
figure('DefaultAxesFontSize',14)
bar(role_counts(this_ind,:), 'stacked')
legend(possible_roles)
xticks(1:length(this_ind));
xticklabels(all_labels_centroid(this_ind))
xtickangle(90)
title('Neuron roles using similarity to data centroids (IDed >=2 times)')
% Same but without the fourth worm
these_worms = [1, 2, 3, 5];
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_centroid(:,these_worms), possible_roles{i}),2);
end
this_ind = find(sum(role_counts,2)>1);
figure('DefaultAxesFontSize',14)
bar(role_counts(this_ind,:), 'stacked')
legend(possible_roles)
xticks(1:length(this_ind));
xticklabels(all_labels_centroid(this_ind))
xtickangle(90)
title('Neuron roles using similarity to data centroids (no 4th worm)')


% Some histograms of how many times a neuron is identified as what role
% (GLOBAL)
possible_roles = unique(combined_dat_global);
possible_roles(cellfun(@isempty,possible_roles)) = [];
num_neurons = size(combined_dat_global,1);
role_counts = zeros(num_neurons,length(possible_roles));
for i=1:length(possible_roles)
    role_counts(:,i) = sum(...
        strcmp(combined_dat_global, possible_roles{i}),2);
end
figure('DefaultAxesFontSize',14)
bar(role_counts, 'stacked')
legend(possible_roles)
xticks(1:num_neurons);
xticklabels(all_labels_global)
xtickangle(90)
title('Neuron roles using similarity to global mode activation')
% Same but ID'ed >=2 times
this_ind = find(sum(role_counts,2)>1);
figure('DefaultAxesFontSize',14)
bar(role_counts(this_ind,:), 'stacked')
legend(possible_roles)
xticks(1:length(this_ind));
xticklabels(all_labels_global(this_ind))
xtickangle(90)
title('Neuron roles using similarity to global mode activation')
%==========================================================================


%% Test Pareto object with 2 y values and a combination
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

model_settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
% global_signal_modes = {{'ID'}};
global_signal_modes = {{'ID','ID_binary'}};
% global_signal_modes = {{'ID','ID_simple','ID_binary'}};
lambda_vec = linspace(0.02,0.1,4);
settings = struct(...
    'file_or_dat', filename,...
    'base_settings', model_settings,...
    'iterate_settings',struct('global_signal_mode',global_signal_modes),...
    'x_vector', lambda_vec,...
    'x_fieldname', 'lambda_sparse',...
    'fields_to_plot',{{{'AdaptiveDmdc_obj','calc_reconstruction_error'},...
                      {'S_sparse_nnz'}}});
    
my_pareto_obj = ParetoFrontObj('CElegansModel', settings);

%---------------------------------------------
% Save baselines and combined values
%---------------------------------------------
f = @(x, use_persistence) ...
    x.AdaptiveDmdc_obj.calc_reconstruction_error([],use_persistence);

baseline_funcs = ...
    {@(x) f(x,true),...
    @(x) x.run_with_only_global_control(@(x2)f(x2,false))};
baseline_names = {'persistence',...
    'global_modes_only'};
for i=1:length(global_signal_modes{1})
    fname = global_signal_modes{1}{i};
    % First get a pareto front; no longer units of L2 error 
    val1 = sprintf('%s_AdaptiveDmdc_obj_calc_reconstruction_error',fname);
    val2 = sprintf('%s_S_sparse_nnz',fname);
    my_pareto_obj.save_combined_y_val(val1, val2);
    % Second calculate the 2 baselines and put them in the same "units"
    for i2=1:length(baseline_funcs)
        b_name = sprintf('%s_%s',fname, baseline_names{i2});
        my_pareto_obj.save_baseline(fname, baseline_funcs{i2}, b_name);
        my_pareto_obj.save_combined_y_val(['baseline__' b_name], 1);
    end
end

%---------------------------------------------
% Plot
%---------------------------------------------

fig = my_pareto_obj.plot_pareto_front('combine');
fig = my_pareto_obj.plot_pareto_front('baseline', true, fig);


%==========================================================================


%% Use RPCA on RECONSTRUCTION residual
clear Xmodes
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

dat = importdata(filename);
this_dat = dat.traces.';
this_dat = this_dat - mean(this_dat,2);

tspan = dat.timeVectorSeconds;
dt = tspan(2)-tspan(1);
X1 = this_dat(:,1:end-1);
X2 = this_dat(:,2:end);

% Do dmd with optimal truncation
r = optimal_truncation(X2);
[ coeff, Omega, Phi, romA ] = dmd( this_dat, dt, r);

% Get reconstruction and error
for jT = length(tspan):-1:1
    Xmodes(:,jT) = coeff.*exp(Omega*tspan(jT));
end
Xapprox = Phi*Xmodes;

reconstruction_error = this_dat - Xapprox;
mean0_error = reconstruction_error - mean(reconstruction_error,2);

% Do RPCA on the residual
% lambda = 0.003;
% lambda = 0.004; % rank 1 with only error mean subtracted
% lambda = 0.005; % rank 1 with all means subtracted
lambda = 0.006; % rank 2 with all means subtracted
[L, S, rank_L, nnz_S] = RobustPCA(mean0_error, lambda);

plotSVD(L,struct('PCA3d',true,'PCA_opt','o','sigma',false,'to_subtract_mean',false));
[u,s,v,proj3d] = plotSVD(L',struct('sigma_modes',1:rank_L,'to_subtract_mean',false));

plot_colored([(1:3021)', real(u(:,1))],dat.SevenStates,dat.SevenStatesKey);
plot_colored([(1:3021)', real(u(:,2))],dat.SevenStates,dat.SevenStatesKey);

plot_2imagesc_colorbar(real(L),real(S),'2 1',...
    'Low-rank component of the residual',...
    'Sparse comonent of the residual')
%==========================================================================


%% Use RPCA on 1-step residual (robustified data)
clear Xmodes
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

dat = importdata(filename);

% Now get a CElegansModel and use the classifier on the reconstructed data
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'use_deriv',true);
settings.global_signal_mode = 'ID';
my_model_residual = CElegansModel(filename, settings);

num_neurons = my_model_residual.original_sz(1);
residual = my_model_residual.AdaptiveDmdc_obj.error_mat(1:num_neurons,:);
mean0_error = residual - mean(residual,2);

% Do RPCA on the residual
% lambda = 0.003;
% lambda = 0.004; % rank 1 with only error mean subtracted
% lambda = 0.005; % rank 1 with all means subtracted
% lambda = 0.011; % rank 1 with all means subtracted
lambda = 0.0115; % rank 2 with all means subtracted
[L, S, rank_L, nnz_S] = RobustPCA(mean0_error, lambda,...
    10*lambda, 1e-6, 3000);

plotSVD(L,struct('PCA3d',true,'PCA_opt','o','sigma',false,'to_subtract_mean',false));
[u,s,v,proj3d] = plotSVD(L',struct('sigma_modes',1:rank_L,'to_subtract_mean',false));

ind = 1:size(u,1);
u = real(u);
plot_colored([ind', u(:,1)],dat.SevenStates(ind),dat.SevenStatesKey);
plot_colored([ind', u(:,2)],dat.SevenStates(ind),dat.SevenStatesKey);
plot_colored([u(:,1),u(:,2),u(:,3)],dat.SevenStates(ind),dat.SevenStatesKey);

plot_2imagesc_colorbar(real(L),real(S),'2 1',...
    'Low-rank component of the residual',...
    'Sparse comonent of the residual')
%==========================================================================


%% Analyze DMDc residual
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

dat = importdata(filename);

% Now get a CElegansModel and use the classifier on the reconstructed data
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID';
my_model_residual = CElegansModel(filename, settings);

residual = my_model_residual.dat_without_control - ...
    my_model_residual.AdaptiveDmdc_obj.calc_reconstruction_control();

% Basic SVD visualizations
residual = residual - mean(residual,2);
plotSVD(residual,struct(...
    'PCA3d',true,'PCA_opt','o','sigma',true,'to_subtract_mean',false));
[u,s,v,proj3d] = plotSVD(residual',struct(...
    'sigma_modes',1:2,'to_subtract_mean',false));

ind = 1:size(u,1);
plot_colored([ind', real(u(:,1))],dat.SevenStates(ind),dat.SevenStatesKey);
title('First SVD mode of full residual')
plot_colored([ind', real(u(:,2))],dat.SevenStates(ind),dat.SevenStatesKey);
title('Second SVD mode of full residual')
plot_colored([u(:,1),u(:,2),u(:,3)],dat.SevenStates(ind),dat.SevenStatesKey);


% Visualizations using RPCA
lambda = 0.007; % rank 2 with all means subtracted
% lambda = 0.006; % rank 1 with all means subtracted
mean0_residual = residual - mean(residual,2);
[L, S, rank_L, nnz_S] = RobustPCA(mean0_residual, lambda);

L = L-mean(L,2);
plotSVD(L,struct('PCA3d',true,'PCA_opt','o','sigma',true,'to_subtract_mean',false));
[u,s,v,proj3d] = plotSVD(L',struct('sigma_modes',1:rank_L,'to_subtract_mean',false));

ind = 1:size(u,1);
u = real(u);
plot_colored([ind', u(:,1)],dat.SevenStates(ind),dat.SevenStatesKey);
title('First SVD mode of L')
plot_colored([ind', u(:,2)],dat.SevenStates(ind),dat.SevenStatesKey);
title('Second SVD mode of L')
plot_colored([u(:,1),u(:,2),u(:,3)],dat.SevenStates(ind),dat.SevenStatesKey);

plot_2imagesc_colorbar(real(L),real(S),'2 1',...
    'Low-rank component of the residual',...
    'Sparse comonent of the residual')

%==========================================================================


%% Use DMDc residual to add a couple neurons to the controller
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

dat = importdata(filename);

% Now get a CElegansModel and use the classifier on the reconstructed data
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID';
my_model_residual = CElegansModel(filename, settings);

residual = my_model_residual.dat_without_control - ...
    my_model_residual.AdaptiveDmdc_obj.calc_reconstruction_control();

% Basic SVD visualizations
%   Also for adding to the controller
residual = residual - mean(residual,2);
[u,s,v,proj3d] = plotSVD(residual,struct(...
    'PCA3d',true,'PCA_opt','o','sigma',true,'to_subtract_mean',false));
[u2,s2,v2,proj3d_2] = plotSVD(residual',struct(...
    'sigma_modes',1:2,'to_subtract_mean',false));

% Look at the top 2 modes OF THE FIRST SVD for neurons to add
[~, mode_1_max] = max(abs(u(:,1)));
[~, mode_2_max] = max(abs(u(:,2)));
% neurons_to_add = mode_1_max;
% neurons_to_add = mode_2_max;
neurons_to_add = [mode_1_max, mode_2_max];

% Add them back in to the dmdc object, then rerun the algorithm
ad_settings = my_model_residual.AdaptiveDmdc_settings;
ad_settings.x_indices(neurons_to_add) = [];
settings.AdaptiveDmdc_settings = ad_settings;

my_model_residual2 = CElegansModel(filename, settings);

residual2 = my_model_residual2.dat_without_control - ...
    my_model_residual2.AdaptiveDmdc_obj.calc_reconstruction_control();
residual2 = residual2 - mean(residual2,2);
[u,s,v,proj3d] = plotSVD(residual2,struct(...
    'PCA3d',true,'PCA_opt','o','sigma',true,'to_subtract_mean',false));
[u2,s2,v2,proj3d_2] = plotSVD(residual2',struct(...
    'sigma_modes',1:2,'to_subtract_mean',false));
ind = 1:size(u2,1);
u2 = real(u2);
plot_colored([ind', u2(:,1)],dat.SevenStates(ind),dat.SevenStatesKey);
title('First SVD mode of L')
plot_colored([ind', u2(:,2)],dat.SevenStates(ind),dat.SevenStatesKey);
title('Second SVD mode of L')
%==========================================================================


%% Train a linear classifier for the experimentalist behavior ID's
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat = importdata(filename);

use_deriv = true;
use_no_state = false;

% With hyperparameter fitting
rng default
this_labels = dat.SevenStates;
if ~use_deriv
    this_dat = dat.traces;
    if ~use_no_state
        this_dat = this_dat(15:end-50,:);
        this_labels = this_labels(15:end-50);
    end
else
    if ~use_no_state
        this_dat = [dat.traces(15:end-50,:),... 
            dat.tracesDif(14:end-50,:)];
        this_labels = this_labels(15:end-50);
    else
        this_dat = [dat.traces(2:end,:), dat.tracesDif];
        this_labels = this_labels(2:end);
    end
end
Mdl = fitcecoc(this_dat, this_labels,...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));

cv_Mdl = crossval(Mdl);
fprintf('10-fold cross validation error is %f\n',...
    kfoldLoss(cv_Mdl))

% Now get a CElegansModel and use the classifier on the reconstructed data
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID';
my_model_classifier = CElegansModel(filename, settings);

reconstructed_data = ...
    my_model_classifier.AdaptiveDmdc_obj.calc_reconstruction_control();
if use_deriv
    reconstructed_data = [reconstructed_data;
        gradient(reconstructed_data)];
end

% Use the linear classifier
predicted_labels = ...
    predict(Mdl, reconstructed_data');

% Plot
tspan = dat.timeVectorSeconds(1:length(this_labels));

figure;
plot(tspan, this_labels, 'LineWidth',2)
hold on
plot(tspan, predicted_labels(15:end-50), 'LineWidth',2)
legend({'Original Labels', 'Predicted Labels (reconstruction)'})
yticklabels(dat.SevenStatesKey)
%==========================================================================


%% Train a linear classifier for the experimentalist behavior ID's
% (w/hyperparameter fitting)
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat = importdata(filename);

use_deriv = true;
use_no_state = false;

% Get a CElegansModel and train the classifier on the original data
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'lambda_sparse',0.015);
settings.global_signal_mode = 'ID';
my_model_classifier2 = CElegansModel(filename, settings);

% THIS COMMAND TAKES A LONG TIME
my_model_classifier2.train_classifier();

% Generate data and watch a movie
my_model_classifier2.generate_time_series(2000);
figure;
num_neurons = my_model_classifier2.original_sz(1);
plot(my_model_classifier2.control_signal_generated(num_neurons+1,:))

my_model_classifier2.plot_colored_arrow_movie([], true);
%==========================================================================


%% Test the new RPCA_reconstruction_residual signal mode
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'lambda_sparse', 0.04,...
    'lambda_global', 0.005,...
    'max_rank_global', 2,...
    'global_signal_mode', 'RPCA_reconstruction_residual');
my_model_residual = CElegansModel(filename, settings);

% Use original control; 3d pca plot
% NOTE: reconstruction is not really that impressive here!
my_model_residual.add_partial_original_control_signal();
my_model_residual.plot_reconstruction_user_control();
my_model_residual.set_simple_labels();
my_model_residual.plot_colored_user_control([],false);

%==========================================================================


%% Test the new RPCA_one_step_residual signal mode
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'lambda_sparse', 0.04,...
    'lambda_global', 0.0115,...
    'max_rank_global', 2,...
    'global_signal_mode', 'RPCA_one_step_residual');
my_model_residual = CElegansModel(filename, settings);

% Use original control; 3d pca plot
% NOTE: reconstruction is not really that impressive here!
my_model_residual.add_partial_original_control_signal();
my_model_residual.plot_reconstruction_user_control();
my_model_residual.set_simple_labels();
my_model_residual.plot_colored_user_control([],false);

%==========================================================================


%% Look at suspiciously spike-like neurons
filename4 = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
% filename4 = '../../Zimmer_data/WildType_adult/simplewt3/wbdataset.mat';
dat4 = importdata(filename4);

labels_of_interest = {'RIVL','RIVR', 'SMDVR', 'SMDVL', 'SMBDR',...
    'RIMR','RIML','RIS','ALA',...
    'SAAVR','SAAVL', 'ASKL','ASKR'};
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


%% Add a couple 'spike-like' neurons to the ID controller
filename = '../../Zimmer_data/WildType_adult/simplewt4/wbdataset.mat';

dat = importdata(filename);

% First calculate the baseline model
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'filter_window_dat',6);
%     'augment_data',6);
settings.global_signal_mode = 'ID_binary';
my_model_residual = CElegansModel(filename, settings);

% Now add some interesting neurons in
%   UPDATE: The "DIVERGENCE" notes below were from a bug... actually adding
%   most of these does better! Adding in all of them does by far the best
%   relatively.
% RIVL/R, SMDVR, SMBDR, RIMR/L, SAAVR/L
% neurons_of_interest = [80, 85, 43, 125, 82, 83, 41, 45]; % Divergence...
% neurons_of_interest = [43]; % does worse...
% neurons_of_interest = [41, 45]; % does worse...
% neurons_of_interest = [82, 83]; % Divergence... % actually still diverges
neurons_of_interest = [80, 85]; % Divergence... % HELPS

ad_settings = my_model_residual.AdaptiveDmdc_settings;
ad_settings.x_indices(neurons_of_interest) = [];
settings.AdaptiveDmdc_settings = ad_settings;
my_model_residual2 = CElegansModel(filename, settings);

% Exploratory plots and headline error
fprintf('Error for base model is %f\n',...
    my_model_residual.AdaptiveDmdc_obj.calc_reconstruction_error())
fprintf('Error for augmented model is %f\n',...
    my_model_residual2.AdaptiveDmdc_obj.calc_reconstruction_error())

my_model_residual.plot_reconstruction_interactive(false);
title('Original model')
my_model_residual2.plot_reconstruction_interactive(false);
title('Augmented model')
%==========================================================================


%% Add a couple 'spike-like' neurons to the RPCA controller
filename = '../../Zimmer_data/WildType_adult/simplewt4/wbdataset.mat';

dat = importdata(filename);

% First calculate the baseline model
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'use_deriv',true,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'RPCA_reconstruction_residual';
my_model_residual = CElegansModel(filename, settings);

% Now add some interesting neurons in
% RIVL/R, SMDVR, SMBDR, RIMR/L, SAAVR/L
neurons_of_interest = [80, 85, 43, 125, 82, 83, 41, 45]; % Divergence...
% neurons_of_interest = [43]; % does worse...
% neurons_of_interest = [41, 45]; % does worse...
% neurons_of_interest = [82, 83]; % Divergence...
% neurons_of_interest = [80, 85]; % Divergence...

ad_settings = my_model_residual.AdaptiveDmdc_settings;
ad_settings.x_indices(neurons_of_interest) = [];
settings.AdaptiveDmdc_settings = ad_settings;
my_model_residual2 = CElegansModel(filename, settings);

% Exploratory plots and headline error
fprintf('Error for base model is %f\n',...
    my_model_residual.AdaptiveDmdc_obj.calc_reconstruction_error())
fprintf('Error for augmented model is %f\n',...
    my_model_residual2.AdaptiveDmdc_obj.calc_reconstruction_error())

my_model_residual.plot_reconstruction_interactive(false);
title('Original model')
my_model_residual2.plot_reconstruction_interactive(false);
title('Augmented model')
%==========================================================================


%% Look at bode plots for input from these 'spike-like' neurons
filename = '../../Zimmer_data/WildType_adult/simplewt4/wbdataset.mat';

dat = importdata(filename);

% First calculate the baseline model
if ~exist('my_model_residual','var')
    settings = struct(...
        'to_subtract_mean',false,...
        'to_subtract_mean_sparse',false,...
        'to_subtract_mean_global',false,...
        'dmd_mode','func_DMDc');
    settings.global_signal_mode = 'ID';
    my_model_residual = CElegansModel(filename, settings);
else
    warning('reusing object')
end

% Now add some interesting neurons in
% RIVL/R, SMDVR, SMBDR, RIMR/L, SAAVR/L
% neurons_of_interest = [80, 85, 43, 125, 82, 83, 41, 45];
input_neurons = [80, 85, 43];
% input_neurons = [71, 74, 39, 46];
% input_neurons = [71, 74, 39, 46, 108];
% input_neurons = [41, 45];
% output_neurons = [39, 46]; %AVAR/L
% output_neurons = [71, 74]; %AVBR/L
output_neurons = [71, 74, 39, 46];
% output_neurons = [39, 74]; %AVAR/AVBL
% output_neurons = [120, 108]; %DA01, VB02

% neurons_of_interest = [2];
% ind = 1:2;

ind = 1:my_model_residual.original_sz(1);
A = my_model_residual.AdaptiveDmdc_obj.A_original(ind,ind);
% Only a subset of outputs and inputs
B = zeros(length(ind),length(input_neurons));
for i2=1:length(input_neurons)
    B(input_neurons(i2), i2) = 1;
end
C = zeros(length(output_neurons),length(ind));
for i2=1:length(output_neurons)
    C(i2,output_neurons(i2)) = 1;
end
D = 0;

% First plot the power spectrum of the neurons of interest
% for i2=input_neurons
%     figure;
%     this_dat = my_model_residual.dat(i2,:);
%     Y = fft(this_dat-mean(this_dat,2));
%     Pyy = Y.*conj(Y)/251;
%     f = 1000/251*(0:127);
%     plot(f,Pyy(1:128))
%     title(sprintf('Power spectral density for neuron %d',i2))
%     xlabel('Frequency (Hz)')
% end

sys = ss(A,B,C,D,1,...
    'inputname',my_model_residual.AdaptiveDmdc_obj.get_names(input_neurons),...
    'outputname',my_model_residual.AdaptiveDmdc_obj.get_names(output_neurons));
% bodeplot(sys)
opts = bodeoptions('cstprefs');
opts.FreqUnits = 'Hz';
[mag,phase,wout] = bode(sys, opts);

figure()
impulse(sys);
figure;
step(sys)

% n = size(mag,3);
% m = size(mag,2);
% for i2=ind
%     figure;
%     subplot(2,1,1)
%     loglog(wout, reshape(mag(i,:,:),[m,n]))
%     title(sprintf('Magnitude for neuron %d',i2))
%     subplot(2,1,2)
%     semilogx(wout, reshape(phase(i,:,:),[m,n]))
%     title('Phase')
%     xlabel('Input frequency (Hz)')
%     pause
% end
%==========================================================================


%% Try new PID-like options
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

dat = importdata(filename);

% First calculate the baseline model
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'filter_window_dat',6,...
    'use_deriv',true,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_and_grad';
% settings.global_signal_mode = 'RPCA_and_grad';
my_model_PID = CElegansModel(filename, settings);

% Exploratory plots and headline error
fprintf('Error for ID_and_gradient model is %f\n',...
    my_model_PID.AdaptiveDmdc_obj.calc_reconstruction_error())
my_model_PID.plot_reconstruction_interactive(false);

my_model_PID.add_partial_original_control_signal();
my_model_PID.plot_reconstruction_user_control();
my_model_PID.plot_colored_user_control([],false);
%==========================================================================


%% Use new ID_binary modes

filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

dat = importdata(filename);

% First calculate the baseline model
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'filter_window_dat',6,...
    'use_deriv',true,...
    'to_normalize_deriv',true);
% settings.global_signal_mode = 'ID_binary';
settings.global_signal_mode = 'ID_binary_and_x_times_state';
% settings.global_signal_mode = 'RPCA_and_grad';
my_model_PID = CElegansModel(filename, settings);
my_model_PID.plot_reconstruction_interactive(false);

%==========================================================================


%% Test new dependent rows
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat = importdata(filename);
num_neurons = size(dat.traces,2);
num_states = 8;

% Define the table for the dependent row objects
row_functions = {XtimesStateDependentRow()};
setup_arguments = {''};
row_indices = {(num_neurons+1):(num_neurons*num_states)};
dependent_rows = table(row_functions, row_indices, setup_arguments);

settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'dependent_rows', dependent_rows);
settings.global_signal_mode = 'ID_binary_and_x_times_state';
my_model_dependent = CElegansModel(filename, settings);
my_model_dependent.plot_reconstruction_interactive(false);

%==========================================================================


%% Try out not subtracting the sparse signal
%   It's still in the controller
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

% First calculate the baseline model
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'filter_window_dat',6,...
    'use_deriv',true,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary';
my_model_sparse = CElegansModel(filename, settings);
my_model_sparse.plot_reconstruction_interactive(false);

my_model_sparse.AdaptiveDmdc_obj.plot_reconstruction(true,true,true,1);
title('With sparse signal subtracted')

% Now reset the data and redo the analysis
dat = importdata(filename);
my_model_sparse.dat = ...
    [dat.traces(1:end-1,:), dat.tracesDif]';
my_model_sparse.calc_AdaptiveDmdc();
my_model_sparse.plot_reconstruction_interactive(false);

my_model_sparse.AdaptiveDmdc_obj.plot_reconstruction(true,true,true,1);
title('With sparse signal still in the data')
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


%% Use both RPCA and ID_binary
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

disp('Does not help...')

% First calculate the baseline model
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'filter_window_dat',6,...
    'use_deriv',true,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'RPCA_reconstruction_residual_and_ID_binary';
my_model_both = CElegansModel(filename, settings);
my_model_both.plot_reconstruction_interactive(false);

fprintf('Overall model error is %f\n',...
    my_model_both.AdaptiveDmdc_obj.calc_reconstruction_error())


% Does it do any better if "drift" is subtracted out?
tmp_dat = my_model_both.L_global - mean(my_model_both.L_global,2);
[u, s, v] = svd(tmp_dat.');
figure;
% ind = 1:4;
ind = 4;
imagesc(real(u(:,ind)*s(ind,ind)*v(:,ind)')')

figure;
plot(real(u(:,ind)))
title('Temporal coefficient')

figure;
plot(real(v(:,ind)))
title('Spatial coefficients')
%==========================================================================


%% Look at neurons that seem to be strongly drifting
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat = importdata(filename);

% this_neuron = 108; % Unnamed, but seems to have slight activity
% this_neuron = 50; % Unnamed, but seems to have slight activity
% this_neuron = 100; % Unnamed, but seems to have slight activity
this_neuron = 101; % RIS
% this_neuron = 111; % Unnamed, but seems to have slight activity
% this_neuron = 121; % Unnamed, but has strong activity at the end
% this_neuron = 42; % Spikes!
% this_neuron = 45; % Spikes!
% this_neuron = 84; % Spikes!
% this_neuron = 1; % Very strong single event

this_dat = dat.traces(:,this_neuron);
% this_dat = dat.tracesDif(:,this_neuron);
this_dat = this_dat - mean(this_dat);


% Also look at the fft to see what we're cutting off
Fs = 1;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = length(this_dat);             % Length of signal
t = (0:L-1)*T;        % Time vector

Y = fft(this_dat);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
figure;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

% Look at the data and a filtered version
figure;
plot(this_dat, 'LineWidth',2)
hold on;

% Make up a low-pass filter
%   From Mathworks page

% lpFilt = designfilt('lowpassiir','FilterOrder',8, ...
%     'PassbandFrequency',10,'PassbandRipple',0.2, ...
%     'SampleRate',10e2);

% lpFilt = designfilt('lowpassiir','FilterOrder',12, ...
%     'HalfPowerFrequency',0.2,'DesignMethod','butter');

% Fstop = 0.1;
% Fpass = 0.05;
Apass = 0.1;
Astop = 20;
Fs = 1;
% Use the fft to make a slightly smarter filter
sz = length(P1);
noise_level = max(P1(round(length(P1)/2):end)) * 1.5;
noise_beginning = [];
while isempty(noise_beginning)
    noise_beginning = f(find(P1-noise_level>0,1,'last'));
    noise_level = noise_level * 0.95;
    fprintf('Reducing Noise level to %.5f\n',noise_level)
end
Fpass = noise_beginning;
Fstop = min(Fpass*1.5,0.499);

design_method = 'cheby2';
lpFilt = designfilt('lowpassiir', ...
  'PassbandFrequency',Fpass, 'StopbandFrequency',Fstop,...  
  'PassbandRipple',Apass,'StopbandAttenuation',Astop,...
  'SampleRate',Fs,...
  'DesignMethod','cheby2');
filter_dat = filtfilt(lpFilt, this_dat);
plot(filter_dat, 'LineWidth',2)

title(sprintf('Std:%.3f; Max:%.2f; Min:%.2f',...
    std(this_dat), max(this_dat), min(this_dat)))

% Also look at the filtered spectrum
Y = fft(filter_dat);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

figure;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of filter(X(t))')
xlabel('f (Hz)')
ylabel('|P1(f)|')

%==========================================================================


%% Use new 'smart' filter mode

filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

dat = importdata(filename);

% First calculate the baseline model
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'filter_window_dat',0,...
    'filter_window_global',10,...
    'filter_aggressiveness',1.1,... % New mode
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'lambda_sparse',0); % NO SPARSE CONTROLLER
settings.global_signal_mode = 'ID_binary_and_grad';
% settings.global_signal_mode = 'ID_binary_and_x_times_state';
% settings.global_signal_mode = 'RPCA_and_grad';
my_model_smart_filter = CElegansModel(filename, settings);
my_model_smart_filter.plot_reconstruction_interactive(false);

%==========================================================================


%% Plot reconstructions with only a partial control signal
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat = importdata(filename);
num_neurons = size(dat.traces,2);
num_slices = size(dat.traces,1);

% First the baseline settings
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',true,...
    ...'filter_window_dat',6,...
    'use_deriv',true,...
    'to_normalize_deriv',true);

% Calculate a model with only sparse signals
settings.global_signal_mode = 'None';
my_model_sparse = CElegansModel(filename, settings);
% my_model_sparse.plot_colored_reconstruction();

% Calculate a full model with both control signals
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_both = CElegansModel(filename, settings);

% Calculate a model with no sparse signal
settings.lambda_sparse = 0;
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_global = CElegansModel(filename, settings);
% my_model_global.plot_colored_reconstruction();

% Now use no sparse signal, but the same (filtered) data as my_model_sparse
dat.traces = my_model_sparse.dat';
settings.use_deriv = false;
settings.to_normalize_deriv = false;
my_model_global_filtered = CElegansModel(dat, settings);
warning('This last model is somewhat unstable...');
% my_model_global_filtered.plot_colored_reconstruction();

% Look at the histogram of neuron errors for each
global_dat = sum((my_model_global.dat - ...
    my_model_global.AdaptiveDmdc_obj.calc_reconstruction_control([],[],false)...
    ).^2, 2) / num_slices;
global_dat = global_dat(1:num_neurons,:);
global_dat_filt = sum((my_model_global_filtered.dat - ...
    my_model_global_filtered.AdaptiveDmdc_obj.calc_reconstruction_control([],[],false)...
    ).^2, 2) / num_slices;
global_dat_filt = global_dat_filt(1:num_neurons,:);
sparse_dat = sum((my_model_sparse.dat - ...
    my_model_sparse.AdaptiveDmdc_obj.calc_reconstruction_control([],[],false)...
    ).^2, 2) / num_slices;
sparse_dat = sparse_dat(1:num_neurons,:);
full_dat = sum((my_model_both.dat - ...
    my_model_both.AdaptiveDmdc_obj.calc_reconstruction_control([],[],false)...
    ).^2, 2) / num_slices;
full_dat = full_dat(1:num_neurons,:);

figure;
bin_width = 5e-3;

x_sz = [0,0.13];
y_sz = [0 100];
subplot(2,1,1)
h1 = histogram(full_dat,'BinWidth',bin_width);
title('Per-neuron error for the full model')
xlim(x_sz)
ylim(y_sz)
% h1 = histogram(full_dat);
% hold on
% figure
% h2 = histogram(global_dat(global_dat>min_error),'BinWidth',bin_width);
% h2 = histogram(global_dat);
subplot(2,1,2)
h2 = histogram(global_dat_filt,'BinWidth',bin_width);
title('Per-neuron error for global-only model (filtered data)')
xlim(x_sz)
ylim(y_sz)
% h2 = histogram(sparse_dat(sparse_dat>min_error),'BinWidth',bin_width);
% legend({'Full model','Model with only global signals'});

figure;
sparse_signal_sum = sum(my_model_both.control_signal(1:num_neurons,:).^2,2);
normalize = sum(my_model_global.dat(1:num_neurons,:).^2,2); % RAW data
fraction_sparse = sparse_signal_sum./normalize;
bin_width = 1e-3;
histogram(fraction_sparse, 'BinWidth',bin_width)
title('Per-neuron fraction of sparse control signal activity')

% More plots!
% Look at the reconstructions errors with only the global signal
% (unfiltered)

figure;
histogram(global_dat,'BinWidth',1e-2)
title('Per-neuron reconstruction error')

[global_sorted, global_ind] = sort(global_dat);
disp(global_ind(end-10:end))

%% Look at different norms: 3d space of L_inf, L2, and "outlier spread"
obj = my_model_global;
this_dat = obj.dat;
this_approx = ...
    obj.AdaptiveDmdc_obj.calc_reconstruction_control([],[],false);

dat_inf = zeros(num_neurons,1);
dat_L2 = zeros(num_neurons,1);
dat_spread = zeros(num_neurons,1);

for i = 1:num_neurons
    dat_diff = this_dat(i,:) - this_approx(i,:);
    dat_diff = dat_diff - mean(dat_diff);
%     dat_diff = dat_diff./std(dat_diff);
    
%     outlier_ind = find(isoutlier(dat_diff));
%     outlier_ind = outlier_ind(2:end-1);
%     sz = length(outlier_ind);
%     start_ind = 0;
%     end_ind = 0;
%     for i2 = 1:(sz-1)
%         if sz==1
%             break
%         end
%         % Check to make sure outliers have neighbors (more robust)
%         if start_ind==0 && ...
%                 outlier_ind(i2+1)==(outlier_ind(i2)+1)
%             start_ind = i2;
%         end
%         if end_ind==0 && ...
%                 outlier_ind(end-i2+1)==(outlier_ind(end-i2)+1)
%             end_ind = i2;
%         end
%         if start_ind>0 && end_ind>0
%             dat_spread(i) = outlier_ind(end-end_ind) - ...
%                 outlier_ind(start_ind);
%         end
%     end
    
    dat_diff = abs(dat_diff);
    dat_inf(i) = norm(dat_diff, Inf);
    dat_L2(i) = norm(dat_diff, 2);
%     dat_inf(i) = norm(dat_diff, Inf)/norm(dat_diff, 2);
%     dat_inf(i) = norm(dat_diff, Inf)/norm(this_approx(i,:), Inf);
%     dat_L2(i) = norm(dat_diff, 2)/trapz(this_approx(i,:));
    dat_spread(i) = max(this_dat(i,:))/norm(this_dat(i,:));
%     dat_spread(i) = max(this_dat(i,:))/std(this_dat(i,:));
%     dat_spread(i) = max(this_dat(i,:))/max(this_approx(i,:));
    % Top 10% and bottom 90% of errors
%     cutoff = quantile(dat_diff, 0.95);
%     dat_inf(i) = norm(dat_diff(dat_diff>=cutoff), 2);
%     dat_L2(i) = norm(dat_diff(dat_diff<cutoff), 2);
    
    fprintf('Inf norm:%.2f; L2 norm:%.1f; Spread:%d; max/std:%.2f\n',...
        dat_inf(i), dat_L2(i), dat_spread(i), max(this_dat(i,:))/std(this_dat(i,:)))
%     obj.plot_reconstruction_interactive(true,i);
%     pause
end
figure;
groups = {{[2, 15, 121]}, ...
    {[1, 101, 5, 62, 63, 64]},...
    {[42, 44, 90]},...
    {[72, 84, 76, 77, 128]},...
    {[44, 46, 83, 93, 53, 57, 38, 68, 59]}};
scatter3(dat_inf, dat_L2, dat_spread);
hold on
for i = 1:length(groups)
    ind = groups{i}{:};
    scatter3(dat_inf(ind), dat_L2(ind), dat_spread(ind),100,'*');
end
legend({'All data', 'Single events','A couple events','Transitions','Noisy state-labels','Clean State Labels'})
xlabel('Infinity Norm')
ylabel('L2 Norm')
zlabel('Spread of outliers')
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


%% Use hold-out cross-validation
filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';
% dat = importdata(filename);
% num_neurons = size(dat.traces,2);
% num_slices = size(dat.traces,1);

% First the baseline settings
ad_settings = struct('cross_val_fraction',0.15);
sparsity_goal = 0.6; %Default value
ad_settings.sparsity_goal = sparsity_goal;
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    ...'dmd_mode','func_DMDc',...
    'dmd_mode','sparse',...
    'add_constant_signal',false,...
    ...'filter_window_dat',6,...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'AdaptiveDmdc_settings',ad_settings);
settings.global_signal_mode = 'ID_binary_and_grad';

num_worms = 5;
err_train = zeros(100, num_worms);
err_test = zeros(1,num_worms);
for i=1:num_worms
    filename = sprintf(filename_template,i);
    my_model_crossval = CElegansModel(filename, settings);

    % Use crossvalidation functions
    err_train(:,i) = my_model_crossval.AdaptiveDmdc_obj.calc_baseline_error();
    err_test(i) = my_model_crossval.AdaptiveDmdc_obj.calc_test_error();
    
    % Print summary statistics
    fprintf('For worm %d:\n', i)
    fprintf('Mean of training errors x10^4: %.2f\n',...
        (1e4)*mean(err_train(:,i)))
    fprintf('Std of training errors x10^4: %.2f\n',(1e4)*std(err_train(:,i)))
    fprintf('Test error x10^4: %.2f\n',...
        (1e4)*err_test(i))
end

figure
plot(err_test,'or','LineWidth',2)
hold on
boxplot(err_train)
ylim([0, 1.1*max(err_test)])
title('Box plot of training errors vs. test data error')

%==========================================================================


%% Use external cross-validation object
filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';

% First the baseline settings
ad_settings = struct('cross_val_fraction',0.15);
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    ...'filter_window_dat',6,...
    'use_deriv',false,...
    'to_normalize_deriv',true,...
    'lambda_sparse',0,...
    'AdaptiveDmdc_settings',ad_settings);
settings.global_signal_mode = 'ID_binary_and_grad';

% num_worms = 5;
num_worms = 1;
err_train = zeros(100, num_worms);
err_test = zeros(1,num_worms);
all_cross_vals = cell(num_worms, 1);

for i=1:num_worms
    filename = sprintf(filename_template,i);
    my_model_crossval = CElegansModel(filename, settings);

    % Use crossvalidation functions
    err_train(:,i) = my_model_crossval.AdaptiveDmdc_obj.calc_baseline_error();
    err_test(i) = my_model_crossval.AdaptiveDmdc_obj.calc_test_error();
    
    % Use external cross validation object
    dmd_func = @(X1,X2,u,~) ...
        my_model_crossval.AdaptiveDmdc_obj.calc_one_step_error(X1,X2,u);
    num_folds = 10;
    num_trials = 20;
    cross_val_test_errors = zeros(num_folds, num_trials);
    cross_val_train_errors = zeros(num_folds, num_trials);
    for i2 = 1:num_trials
        settings2 = struct('num_test_columns',5*i2, ...
            'control_signal',my_model_crossval.control_signal,...
            'num_folds_to_test', 10);
        this_cross_val = CrossValidationDmd(...
            my_model_crossval.dat, dmd_func, settings2);
        cross_val_test_errors(:,i2) = this_cross_val.test_errors;
        cross_val_train_errors(:,i2) = this_cross_val.train_errors;
%         this_cross_val.plot_box();
    end
    % Plot these errors
    fig = figure;
    boxplot(cross_val_test_errors);
    test_ylim = ylim;
    hold on
    boxplot(cross_val_train_errors);
    all_limits = [test_ylim ylim];
    ylim([min(all_limits), max(all_limits)]);
    
    % Print summary statistics
    fprintf('For worm %d:\n', i)
    fprintf('Mean of training errors x10^4: %.2f\n',...
        (1e4)*mean(err_train(:,i)))
    fprintf('Std of training errors x10^4: %.2f\n',(1e4)*std(err_train(:,i)))
    fprintf('Test error x10^4: %.2f\n',...
        (1e4)*err_test(i))
end

figure
plot(err_test,'or','LineWidth',2)
hold on
boxplot(err_train)
ylim([0, 1.1*max(err_test)])
title('Box plot of training errors vs. test data error')





%==========================================================================


%% Add a couple 'spike-like' neurons with NO sparse controller
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

dat = importdata(filename);
num_neurons = size(dat.traces,2);

% First calculate the baseline model
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'lambda_sparse',0);
%     'augment_data',6);
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_residual = CElegansModel(filename, settings);

% Now add some interesting neurons in
%   UPDATE: The "DIVERGENCE" notes below were from a bug... actually adding
%   most of these does better! Adding in all of them does by far the best
%   relatively.

% All unnamed neurons, but quite cleanly spikey
%   Result: somewhat better; same w/derivs
neurons_of_interest = [2, 15, 121];

% Still all unnamed
%   Result: somewhat better; not as good w/derivs
% neurons_of_interest = [1, 5, 62:64, 101];
neurons_of_interest = [neurons_of_interest, [1, 5, 62:64, 101]];

% SMDXX
%   Result: Actually doesn't help on the L2 error, but looks better by eye
% neurons_of_interest = [42, 44, 90];
% neurons_of_interest = [neurons_of_interest, [42, 44, 90]]; % Goes unstable when paired with spikers
% neurons_of_interest = [42]; % Doesn't help
% neurons_of_interest = [44]; % Destabilizing a bit
% neurons_of_interest = [90]; % Doesn't help

% AVB, RIB, unnamed
%   Result: Helps a goood bit; similar with derivatives
neurons_of_interest = [72, 84, 76, 77, 128];

% Also add the derivatives
neurons_of_interest = [neurons_of_interest, neurons_of_interest+num_neurons];

% Make the actual model
ad_settings = my_model_residual.AdaptiveDmdc_settings;
ad_settings.x_indices(neurons_of_interest) = [];
settings.AdaptiveDmdc_settings = ad_settings;
my_model_residual2 = CElegansModel(filename, settings);

% Exploratory plots and headline error
fprintf('Error for base model is %f\n',...
    my_model_residual.AdaptiveDmdc_obj.calc_reconstruction_error())
fprintf('Error for augmented model is %f\n',...
    my_model_residual2.AdaptiveDmdc_obj.calc_reconstruction_error())

% my_model_residual.plot_reconstruction_interactive(false);
% title('Original model')
% my_model_residual2.plot_reconstruction_interactive(false);
% title('Augmented model')
my_model_residual2.plot_colored_reconstruction();
%==========================================================================


%% Use hold-out cross-validation as a function of percent
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

% First the baseline settings
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    ...'filter_window_dat',6,...
    'use_deriv',true,...
    'to_normalize_deriv',true...,...
    );%'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_and_grad';

num_runs = 10;
cross_val_window_size_percent = 0.7;

% all_percents = linspace(0.3, 0.05, num_runs);
all_percents = linspace(0.45, 0.15, num_runs);
err_train = zeros(200, num_runs);
if cross_val_window_size_percent == 1
    err_test = zeros(1,num_runs);
else
    err_test = zeros(size(err_train));
end
all_cross_vals = cell(num_runs, 1);

for i = 1:num_runs
    ad_settings = struct('hold_out_fraction',all_percents(i),...
        'cross_val_window_size_percent', cross_val_window_size_percent);
    settings.AdaptiveDmdc_settings = ad_settings;
    my_model_crossval = CElegansModel(filename, settings);

    % Use crossvalidation functions
    err_train(:,i) = my_model_crossval.AdaptiveDmdc_obj.calc_baseline_error();
    if cross_val_window_size_percent == 1
        err_test(i) = my_model_crossval.AdaptiveDmdc_obj.calc_test_error();
    else
        err_test(:,i) = my_model_crossval.AdaptiveDmdc_obj.calc_test_error();
    end
    
    % Print summary statistics
%     fprintf('For worm %d:\n', i)
%     fprintf('Mean of training errors x10^4: %.2f\n',...
%         (1e4)*mean(err_train(:,i)))
%     fprintf('Std of training errors x10^4: %.2f\n',(1e4)*std(err_train(:,i)))
%     fprintf('Test error x10^4: %.2f\n',...
%         (1e4)*err_test(i))
end

figure
if cross_val_window_size_percent == 1
    plot(err_test,'or','LineWidth',2)
else
    boxplot(err_test,'colors',[1 0 0])
end
hold on
boxplot(err_train)
ylim([0, 1.1*max(max([err_test;err_train]))])
xticklabels(round(all_percents,2))
xlabel('Percent of data in hold-out set')
ylabel('L2 error')
title('Box plot of training errors vs. test data error')

%==========================================================================


%% Use plots from paper plots to look at a subset of bar charts
all_figs = cell(5,1);

filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';
% settings = struct(...
%     'to_subtract_mean',false,...
%     'to_subtract_mean_sparse',false,...
%     'to_subtract_mean_global',false,...
%     'dmd_mode','func_DMDc');
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

%%
max_err_percent = 0.25;
only_named_neurons = false;
num_neurons = size(combined_dat_global,1);
for i=1:5
%     % Use the dynamic fixed point
%     [all_roles_dynamics{i,1}, all_roles_dynamics{i,2}] = ...
%         all_models{i}.calc_neuron_roles_in_transition(true, [], true);
%     % Just use centroid of a behavior
%     [all_roles_centroid{i,1}, all_roles_centroid{i,2}] = ...
%         all_models{i}.calc_neuron_roles_in_transition(true, [], false);
    % Global mode actuation
    [all_roles_global{i,1}, all_roles_global{i,2}] = ...
        all_models{i}.calc_neuron_roles_in_global_modes(...
        only_named_neurons, [], max_err_percent);
end

%---------------------------------------------
% Data preprocessing
%---------------------------------------------
% [ combined_dat_dynamic, all_labels_dynamic ] =...
%     combine_different_trials( all_roles_dynamics );
% [ combined_dat_centroid, all_labels_centroid ] =...
%     combine_different_trials( all_roles_centroid );
[ combined_dat_global, all_labels_global ] =...
    combine_different_trials( all_roles_global, ~only_named_neurons );

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
all_figs{3} = figure('DefaultAxesFontSize',14);
if only_named_neurons
    % Long names mean it was an ambiguous identification
    name_lengths = cellfun(@(x) length(x)<6, all_labels_global)';
    this_ind = find((sum(role_counts,2)>3).*name_lengths);
    b = bar(role_counts(this_ind,:), 'stacked');
    xticks(1:length(this_ind));
    xticklabels(all_labels_global(this_ind))
    xtickangle(90)
else
    b = bar(role_counts, 'stacked');
end
for i=1:length(b)
    b(i).FaceColor = my_cmap_3d(i,:);
end
legend([possible_roles(1:end-1); {'high error'}])
yticks(1:max(max((role_counts))))
ylabel('Number of times identified')
title('Neuron roles using global mode activation')

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


%% Use hold-out cross-validation as a function of truncation rank
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

% First the baseline settings
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true...,...
    );%'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_and_grad';

cross_val_window_size_percent = 0.7;

% all_ranks = 3:5:75;
all_ranks = 1:30;
num_runs = length(all_ranks);
err_train = zeros(200, num_runs);
if cross_val_window_size_percent == 1
    err_test = zeros(1,num_runs);
else
    err_test = zeros(size(err_train));
end
all_cross_vals = cell(num_runs, 1);

for i = 1:num_runs
    ad_settings = struct('hold_out_fraction',0.2,...
        'cross_val_window_size_percent', cross_val_window_size_percent,...
        'truncation_rank', all_ranks(i),...
        'truncation_rank_control', all_ranks(i));
    settings.AdaptiveDmdc_settings = ad_settings;
    my_model_crossval = CElegansModel(filename, settings);

    % Use crossvalidation functions
    err_train(:,i) = my_model_crossval.AdaptiveDmdc_obj.calc_baseline_error();
    if cross_val_window_size_percent == 1
        err_test(i) = my_model_crossval.AdaptiveDmdc_obj.calc_test_error();
    else
        err_test(:,i) = my_model_crossval.AdaptiveDmdc_obj.calc_test_error();
    end
end

figure
if cross_val_window_size_percent == 1
    plot(err_test,'or','LineWidth',2)
else
    boxplot(err_test,'colors',[1 0 0])
end
hold on
boxplot(err_train)
ylim([0, 1.1*max(max([err_test;err_train]))])
xticklabels(round(all_ranks,2))
xlabel('Truncation rank')
ylabel('L2 error')
title('Box plot of training errors vs. test data error')

%==========================================================================


%% Examine correlation coefficients of data and reconstructions
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

dat = my_model_errors.dat;
approx = my_model_errors.AdaptiveDmdc_obj.calc_reconstruction_control();
R = zeros(size(dat,1),1);
P = zeros(size(dat,1),1);
for i = 1:size(dat,1)
    [tmp_R, tmp_P] = corrcoef(dat(i,:), approx(i,:));
    R(i) = tmp_R(2,1); % Get off-diagonal element
    P(i) = tmp_P(2,1); % Get off-diagonal element
end

figure;
plot(R);
figure;
plot(P)

err = find(R(1:129)<0.5);
my_model_errors.plot_reconstruction_interactive(false, err);

good = find(R(1:129)>0.95);
my_model_errors.plot_reconstruction_interactive(false, good);
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


%% Look at offsetting the data by different amounts
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

% First the baseline settings
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'to_normalize_deriv',true,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_and_grad';

cross_val_window_size_percent = 0.7;

% all_ranks = 3:5:75;
all_offsets = 1:30;
num_runs = length(all_offsets);
err_train = zeros(200, num_runs);
if cross_val_window_size_percent == 1
    err_test = zeros(1,num_runs);
else
    err_test = zeros(size(err_train));
end
all_cross_vals = cell(num_runs, 1);

for i = 1:num_runs
    ad_settings = struct('hold_out_fraction',0.2,...
        'cross_val_window_size_percent', cross_val_window_size_percent,...
        'dmd_offset', all_offsets(i));
    settings.AdaptiveDmdc_settings = ad_settings;
    my_model_crossval = CElegansModel(filename, settings);

    % Use crossvalidation functions
    err_train(:,i) = my_model_crossval.AdaptiveDmdc_obj.calc_baseline_error();
    if cross_val_window_size_percent == 1
        err_test(i) = my_model_crossval.AdaptiveDmdc_obj.calc_test_error();
    else
        err_test(:,i) = my_model_crossval.AdaptiveDmdc_obj.calc_test_error();
    end
end

figure
if cross_val_window_size_percent == 1
    plot(err_test,'or','LineWidth',2)
else
    boxplot(err_test,'colors',[1 0 0])
end
hold on
boxplot(err_train)
ylim([0, 1.1*max(max([err_test;err_train]))])
xticklabels(round(all_offsets,2))
xlabel('DMD data offset')
ylabel('L2 error')
title('Box plot of training errors vs. test data error')

%==========================================================================


%% Try tls dmd
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
ad_settings = struct('what_to_do_dmd_explosion','project');
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'dmd_mode','tdmd',...
    'lambda_sparse',0,...
    'AdaptiveDmdc_settings',ad_settings);
settings.global_signal_mode = 'None';
my_model_tls = CElegansModel(filename, settings);

my_model_tls.plot_colored_reconstruction();
my_model_tls.plot_eigenvalues_and_frequencies(114)

%==========================================================================


%% Build models to look at each data set
filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary';
% settings.global_signal_mode = 'ID_binary_and_grad';

%---------------------------------------------
% Calculate 5 worms
%---------------------------------------------
all_models = cell(5,1);
for i=1:5
    filename = sprintf(filename_template,i);
%     if i==4
%         settings.lambda_sparse = 0.035; % Decided by looking at pareto front
%     else
%         settings.lambda_sparse = 0.05;
%     end
    all_models{i} = CElegansModel(filename,settings);
end


%==========================================================================


%% Look at Kato-type raw-ish (?) data
% filename_both = {...
%     'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\sevenStateColoring.mat',...
%     'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\TS20140715e_lite-1_punc-31_NLS3_2eggs_56um_1mMTet_basal_1080s.mat'};
% filename_both = {...
%     'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\sevenStateColoring.mat',...
%     'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\TS20140715f_lite-1_punc-31_NLS3_3eggs_56um_1mMTet_basal_1080s.mat'};
% filename_both = {...
%     'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\sevenStateColoring.mat',...
%     'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\TS20140905c_lite-1_punc-31_NLS3_AVHJ_0eggs_1mMTet_basal_1080s.mat'};
% filename_both = {...
%     'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\sevenStateColoring.mat',...
%     'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\TS20140926d_lite-1_punc-31_NLS3_RIV_2eggs_1mMTet_basal_1080s.mat'};
filename_both = {...
    'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\sevenStateColoring.mat',...
    'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\TS20141221b_THK178_lite-1_punc-31_NLS3_6eggs_1mMTet_basal_1080s.mat'};

settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    ...'lambda_sparse', 0,...
    'use_deriv',false,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary';

% First no derivative (note: they should be much better)
my_model_Kato_no_deriv = CElegansModel(filename_both, settings);

% First no derivative (note: they should be much better)
%   However, the derivatives are discontinuous!
settings.use_deriv = true;
my_model_Kato_with_deriv = CElegansModel(filename_both, settings);

% Basic plots
my_model_Kato_no_deriv.plot_reconstruction_interactive(false);
title('No derivative Kato model')
my_model_Kato_with_deriv.plot_reconstruction_interactive(false);
title('With derivative Kato model')



%==========================================================================


%% Look at data with Oxygen stimulus
folder_name = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\npr1_1_PreLet\AN20140730a_ZIM575_PreLet_6m_O2_21_s_1TF_47um_1330_\';
filename = [folder_name 'wbdataset.mat'];
% Calculate the model
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary';
my_model_stimulus = ...
    CElegansModel(filename, settings);

%==========================================================================


%% Reprise: Use dynamics from worm5 on worm3
% Note: learn the control signal from the worm though
filename5 = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
filename3 = '../../Zimmer_data/WildType_adult/simplewt3/wbdataset.mat';

% Get first model and error
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'lambda_sparse', 0);
settings.global_signal_mode = 'ID_binary';

my_model5 = CElegansModel(filename5, settings);
fprintf('Reconstruction error for worm 5: %f\n',...
    my_model5.AdaptiveDmdc_obj.calc_reconstruction_error());
% my_model5.AdaptiveDmdc_obj.plot_reconstruction(true);

% Get second model and initial error
my_model3 = CElegansModel(filename3, settings);
fprintf('Reconstruction error for worm 3: %f\n',...
    my_model3.AdaptiveDmdc_obj.calc_reconstruction_error());
my_model3.AdaptiveDmdc_obj.plot_reconstruction(true);

% Get the overlapping neurons
names5 = my_model5.AdaptiveDmdc_obj.get_names([], false, false);
names3 = my_model3.AdaptiveDmdc_obj.get_names([], false, false);
% Indices are different for worms 3 and 5
[ind3] = ismember(names3, names5);
ind3 = logical(ind3.*(~strcmp(names3, '')));
[ind5] = ismember(names5, names3);
ind5 = logical(ind5.*(~strcmp(names5, '')));

% Truncate the data and redo the worms
dat_struct5 = importdata(filename5);
if length(ind5) > size(dat_struct5.traces,2)
    % Then we're using derivatives too, but the indices should be shorter
    ind5 = ind5(1:(length(ind5)/2));
    dat_struct5.tracesDif(:,~ind5) = [];
end
dat_struct5.traces(:,~ind5) = [];
for i={'ID','ID2','ID3'}
    tmp = dat_struct5.(i{1});
    tmp(~ind5) = [];
    dat_struct5.(i{1}) = tmp;
end
% settings.lambda_global = 0.01;
% settings.lambda_sparse = 0.07;
my_model5_truncate = CElegansModel(dat_struct5, settings);
my_model5_truncate.plot_reconstruction_interactive(true);

dat_struct3 = importdata(filename3);
if length(ind3) > size(dat_struct3.traces,2)
    % Then we're using derivatives too, but the indices should be shorter
    ind3 = ind3(1:(length(ind3)/2));
    dat_struct3.tracesDif(:,~ind3) = [];
end
dat_struct3.traces(:,~ind3) = [];
for i={'ID','ID2','ID3'}
    tmp = dat_struct3.(i{1});
    tmp(~ind3) = [];
    dat_struct3.(i{1}) = tmp;
end
my_model3_truncate = CElegansModel(dat_struct3, settings);
my_model3_truncate.plot_reconstruction_interactive(true);
title('Truncated model with original dynamics')

% Get errors and apply worm5 dynamics to worm3
fprintf('Reconstruction error for truncated worm 3: %f\n',...
    my_model3_truncate.AdaptiveDmdc_obj.calc_reconstruction_error());
fprintf('Reconstruction error for truncated worm 5: %f\n',...
    my_model5_truncate.AdaptiveDmdc_obj.calc_reconstruction_error());

my_model3_truncate.AdaptiveDmdc_obj.A_original = ...
    my_model5_truncate.AdaptiveDmdc_obj.A_original;
my_model3_truncate.AdaptiveDmdc_obj.A_separate = ...
    my_model5_truncate.AdaptiveDmdc_obj.A_separate;
my_model3_truncate.plot_reconstruction_interactive(true);
title("Reconstruction using alternate dynamics")

fprintf('Reconstruction error for worm 3 data and worm 5 A matrix: %f\n',...
    my_model3_truncate.AdaptiveDmdc_obj.calc_reconstruction_error());
%==========================================================================


%% Reprise: Examine correlation coefficients of data and reconstructions
% filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
filename = {...
    'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\sevenStateColoring.mat',...
    'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\TS20141221b_THK178_lite-1_punc-31_NLS3_6eggs_1mMTet_basal_1080s.mat'};

% First the settings and model
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'use_deriv',true,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary_and_grad';
my_model_errors = CElegansModel(filename, settings);

% Also get a no-sparse model
settings.lambda_sparse = 0;
my_model_global_only = CElegansModel(filename, settings);

% Get correlation matrices
[R_full, A_full] = my_model_errors.calc_correlation_matrix();
[R_global, A_global] = my_model_global_only.calc_correlation_matrix();

% Plot matrix and diagonals
% figure;
% imagesc(A);
% title('Full correlation matrix')
% figure;
% imagesc(A(1:n2,1:n2) - A((n+1):(3*n2),1:n2))
% title('Missing Correlation between data and reconstruction')
% colorbar
figure;
histogram(R_full, 'BinWidth', 0.05)
hold on
histogram(R_global, 'BinWidth', 0.05)
title('Correlation of diagonals')

%==========================================================================





