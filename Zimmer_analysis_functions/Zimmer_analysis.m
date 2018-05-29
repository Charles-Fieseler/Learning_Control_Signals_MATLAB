

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
lambda_global = 0.004; % rank=1
% lambda_global = 0.005; % rank=2
% lambda_global = 0.0055; % rank=3
settings = struct('lambda_global',lambda_global);
my_model2 = CElegansModel(filename, settings);

my_model2.plot_colored_fixed_point();

% Plot all individually labeled behaviors (as clouds)
my_model2.plot_colored_fixed_point('FWD');
my_model2.plot_colored_fixed_point('REVSUS');

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
    'lambda_sparse', 0.035);
my_model = CElegansModel(filename, settings);

dat_with_control = my_model.dat_with_control;
target = 'C:\Users\charl\Documents\Current_work\Zimmer_nn_predict\dat_with_control';
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

%% Plot pareto fronts of the sparse signal
% Note: learn the control signal from the worm though
filename5 = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

% Get model and initial error
settings = struct('to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false);
settings.global_signal_mode = 'ID';
my_model5 = CElegansModel(filename5, settings);
fprintf('Reconstruction error for worm 5: %f\n',...
    my_model5.AdaptiveDmdc_obj.calc_reconstruction_error());
% my_model5.AdaptiveDmdc_obj.plot_reconstruction(true);

% Calculate pareto front with different types of global mode calculations
lambda_vec = linspace(0.03,0.07,50);
global_signal_mode = {'ID_simple','ID_and_offset'};%'ID',
for i = 1:length(global_signal_mode)
    my_model5.calc_pareto_front(lambda_vec, global_signal_mode{i}, (i>1));
end

my_model5.plot_pareto_front();

%==========================================================================









