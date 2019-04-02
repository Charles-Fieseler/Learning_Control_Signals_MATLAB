

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
dat_struct = importdata(filename);

settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'filter_window_dat', 0,...
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

% Plot the RPCA learned signal
ind = my_model_residual.control_signals_metadata{'RPCA_one_step_residual',:}{:};
figure;
plot(my_model_residual.control_signal(ind,:)');

% Just look at the residual in PCA space
X1 = my_model_residual.dat(:,1:end-1);
X2 = my_model_residual.dat(:,2:end);
this_dat = X2 - (X2/X1)*X1; %Naive DMD residual 

[~, ~, ~, proj3d] = plotSVD(this_dat);
plot_colored(proj3d, dat_struct.SevenStates(1:size(proj3d,2)),...
    dat_struct.SevenStatesKey);

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


%% Optimized DMD (real data)
filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
% dat_struct = importdata(filename);

settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    ...'dmd_mode','func_DMDc',...
    'dmd_mode','optdmd',...
    'add_constant_signal',false,...
    'filter_window_dat',0,...
    'use_deriv',false,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary';
my_model_opt = CElegansModel(filename, settings);

%==========================================================================


%% Use cubic spline to interpolate data and derivatives
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);

t = dat_struct.timeVectorSeconds;
dat = dat_struct.traces;
dat = CElegansModel.flat_filter(dat,3);
pp = spline(t, dat');

% pd is the polynomial degree of the polynomial segments,
% so for a spline created by spline, we have pd=3.
pd = pp.order - 1; 
D = diag(pd:-1:1,1);
% given deriv as the order of the derivative to compute
% just loop
pprime = pp;
pprime.coefs = pp.coefs*D;

% Sample plot
figure;
% t_interp = linspace(0,t(end),length(t)*2);
t_interp = t;
plot(t, dat(:,46))
hold on
smooth_dat = ppval(pp,t_interp);
plot(t_interp, smooth_dat(46,:))
title('Data')

figure
plot(t(1:end-1), dat_struct.tracesDif(:,46))
hold on
smooth_deriv = ppval(pprime,t_interp);
plot(t_interp, smooth_deriv(46,:))
% plot(t_interp(1:end-1),diff(smooth_dat(46,:)))
title('Derivatives')

% Now make a model
dat_struct.traces = smooth_dat';
dat_struct.tracesDif = smooth_deriv';

settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'filter_window_dat',0,...
    'use_deriv',true,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary';
my_model_smooth = CElegansModel(dat_struct, settings);

my_model_smooth.plot_colored_reconstruction();
disp(my_model_smooth.AdaptiveDmdc_obj.calc_reconstruction_error())
%==========================================================================


%% Try to get stable no-derivative models
close all

% filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
foldername = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\npr1_1_PreLet\';
filename = [foldername ...
    ...'AN20140730a_ZIM575_PreLet_6m_O2_21_s_1TF_47um_1330_\'...
    ...'AN20140730b_ZIM575_PreLet_6m_O2_21_s_1TF_47um_1600_\'...
    ...'AN20140730c_ZIM575_PreLet_6m_O2_21_s_1TF_47um_1600_\'...
    ...'AN20140730e_ZIM575_PreLet_6m_O2_21_s_1TF_47um_1840_\'...
    ...'AN20140730g_ZIM575_PreLet_6m_O2_21_s_1TF_47um_1840_\'...
    ...'AN20140730i_ZIM575_PreLet_6m_O2_21_s_1TF_47um_2015_\'...
    ...'AN20140807b_ZIM575_PreLet_6m_O2_21_s_1mMTF_47um_1440_\'...
    ...'AN20140807d_ZIM575_PreLet_6m_O2_21_s_1mMTF_47um_1540_\'...
    ...'AN20150910a_ZIM1027_1mMTF_O2_21_s_47um_1610_PreLet_\'...
    'AN20150917a_ZIM1027_1mMTF_O2_21_s_47um_1345_PreLet_\'...
    'wbdataset.mat'];
dat_struct = importdata(filename);

ad_settings = struct('sparsity_goal', 0.4);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    ...'dmd_mode','sparse',...
    'add_constant_signal',false,...
    'filter_window_dat',0,...
    ...'AdaptiveDmdc_settings', ad_settings,...
    ...'lambda_sparse',0,...
    ...'filter_aggressiveness', 1.01,...
    'use_deriv',false,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary';
my_model_smooth = CElegansModel(dat_struct, settings);

% my_model_smooth.plot_colored_reconstruction();
my_model_smooth.plot_reconstruction_interactive(false);
disp(my_model_smooth.AdaptiveDmdc_obj.calc_reconstruction_error())

fprintf('Truncation rank (control): %d (%d)\n',...
    my_model_smooth.AdaptiveDmdc_obj.truncation_rank,...
    my_model_smooth.AdaptiveDmdc_obj.truncation_rank_control);
%==========================================================================


%% Perturb state labels and check model statistics
filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
dat_struct = importdata(filename);

ad_settings = struct(...
    'hold_out_fraction', 0.15,...
    'cross_val_window_size_percent', 0.7);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'filter_window_dat',0,...
    ...'lambda_sparse', 0,... % For a quick look at the sensitivity
    'AdaptiveDmdc_settings', ad_settings,...
    'use_deriv',false,...
    'to_normalize_deriv',true);
settings.global_signal_mode = 'ID_binary';

n = 10;
all_models = cell(n, 1);

max_perturb = 10;
num_perturb_single_u = 5;
all_u = perturb_state_labels(dat_struct.SevenStates, [],...
    max_perturb, num_perturb_single_u, n);

err_train = zeros(200, n);
err_test = zeros(200, n);

for i = 1:n
    dat_struct.SevenStates = all_u{i};
    all_models{i} = CElegansModel(dat_struct, settings);
    err_train(:, i) = all_models{i}.AdaptiveDmdc_obj.calc_baseline_error();
    err_test(:, i) = all_models{i}.AdaptiveDmdc_obj.calc_test_error();
end

figure
boxplot(err_test,'Colors',[1 0 0])
hold on
boxplot(err_train)
ylim([0, 1.1*max(max([err_test;err_train]))])
title(sprintf('Training errors vs. test data error (max_perturb=%d; num_perturb_single_u=%d)',...
    max_perturb, num_perturb_single_u))

%==========================================================================


%% Ignoring some neurons: autocorrelation
filename = '../../Zimmer_data/WildType_adult/simplewt4/wbdataset.mat';
dat_struct = importdata(filename);
dat = dat_struct.traces;
n = size(dat,2);

for i = 1:n
    f = figure;
    subplot(2,1,1)
    plot(dat(:,i))
    title(sprintf('Neuron %d (%s)',...
        i, my_model_filter.AdaptiveDmdc_obj.get_names(i)))
    subplot(2,1,2)
    acf(dat(:,i), 100);
    pause
    close(f)
end

%==========================================================================


%% Ignoring some neurons: autocorrelation within a behavior
% filename = '../../Zimmer_data/WildType_adult/simplewt4/wbdataset.mat';
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);
dat = dat_struct.traces;

% Use model as filter
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'lambda_sparse', 0,...
    'use_deriv',false);

% Don't bother with state labels
settings.global_signal_mode = 'None';
my_model_filter = CElegansModel(filename, settings);

which_label = 'REVSUS';
[~, ~, ~, ind] = my_model_filter.get_control_signal_during_label(which_label);
interesting_neurons = {'SMBDL', 'SMBDR', 'AVAL', 'AVAR'};
interesting_neurons = find(contains(...
    my_model_filter.AdaptiveDmdc_obj.get_names(), interesting_neurons));
n = length(interesting_neurons);

% for i = 1:n
%     this_n = interesting_neurons(i);
%     
%     figure;
%     subplot(2,1,1)
%     plot(ind, dat(ind,this_n))
%     title(sprintf('Neuron %d (%s) during %s',...
%         i, my_model_filter.AdaptiveDmdc_obj.get_names(this_n), which_label))
%     subplot(2,1,2)
%     acf(dat(ind,this_n), 100);
% end

% SMD subset
ind = 1000:1600;
this_n = 125;
figure;
subplot(2,1,1)
plot(ind, dat(ind,this_n))
title(sprintf('Neuron %d (%s) during %s',...
    i, my_model_filter.AdaptiveDmdc_obj.get_names(this_n), 'SLOW'))
subplot(2,1,2)
acf(dat(ind,this_n), 600);
% AVA subset
ind = 700:1100;
this_n = 46;
figure;
subplot(2,1,1)
plot(ind, dat(ind,this_n))
title(sprintf('Neuron %d (%s) during %s',...
    i, my_model_filter.AdaptiveDmdc_obj.get_names(this_n), 'REVSUS'))
subplot(2,1,2)
acf(dat(ind,this_n), 200);
%==========================================================================


%% Ignoring neurons: test autocorrelation threshold
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);
dat = dat_struct.traces;

% Use model as filter
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    ...'lambda_sparse', 0,...
    'autocorrelation_noise_threshold', 0.5,...
    'use_deriv',false);

settings.global_signal_mode = 'ID_binary';
my_model_prune = CElegansModel(filename, settings);

% Easy measure of quality
my_model_prune.plot_colored_reconstruction();

%==========================================================================


%% Adding neurons to controller: O2 sensory
folder_name = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\npr1_1_PreLet\AN20140730a_ZIM575_PreLet_6m_O2_21_s_1TF_47um_1330_\';
filename = [folder_name 'wbdataset.mat'];

% First calculate the baseline model
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'use_deriv',false,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary';

% Get the baseline model
my_model_sensory = CElegansModel(filename, settings);

% Find the neurons to add
neurons_of_interest_labels = {'AQR', 'URXL', 'URXR', 'BAGL', 'BAGR'};
neurons_of_interest = find(contains(...
    my_model_sensory.AdaptiveDmdc_obj.get_names([], [], false, false),...
    neurons_of_interest_labels));

% Make the sensory-aware model
ad_settings = my_model_sensory.AdaptiveDmdc_settings;
ad_settings.x_indices(neurons_of_interest) = [];
settings.AdaptiveDmdc_settings = ad_settings;
my_model_sensory2 = CElegansModel(filename, settings);

% General plots
my_model_sensory.plot_reconstruction_interactive(false);
title('Original model')
my_model_sensory2.plot_reconstruction_interactive(true);
title('O2 sensory neurons added')

%==========================================================================


%% Adding neurons to controller: use new input method
folder_name = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\npr1_1_PreLet\AN20140730a_ZIM575_PreLet_6m_O2_21_s_1TF_47um_1330_\';
filename = [folder_name 'wbdataset.mat'];

% First calculate the baseline model
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal', false,...
    'use_deriv',false,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary';

% Get the baseline model
my_model_sensory = CElegansModel(filename, settings);

% Make the sensory-aware model (NEW)
settings.designated_controller_channels = ...
    {'O2_sensory', {'AQR', 'URX', 'BAG'}};
my_model_sensory2 = CElegansModel(filename, settings);

% General plots
my_model_sensory.plot_reconstruction_interactive(false);
title('Original model')
my_model_sensory2.plot_reconstruction_interactive(true);
title('O2 sensory neurons added')

%==========================================================================


%% Fix the SMB names in the original Zimmer structs
% Note that I renamed them to 'old' by hand
filename_template = '../../Zimmer_data/WildType_adult/simplewt%d/wbdataset';
filename_template_export = [filename_template '.mat'];
filename_template_import = [filename_template '_old.mat'];

fields_to_check = {'ID', 'ID2', 'ID3'};

for i = 1:5
    dat = importdata(sprintf(filename_template_import, i));
    
    for i2 = 1:length(fields_to_check)
        this_array = dat.(fields_to_check{i2});
        for i3 = 1:length(dat.ID)
            str = this_array{i3};
            if ischar(str) && contains(str, 'SMB')
                this_array{i3} = strrep(str, 'SMB', 'SMD');
            end
        end
        
        dat.(fields_to_check{i2}) = this_array;
    end
    save(sprintf(filename_template_export, i), 'dat');
end


%==========================================================================


%% Compare to a no-dynamics fit (no sparse)
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'dmd_mode','func_DMDc',...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary';

% First get a baseline model (no sparse signals)
my_model_base = CElegansModel(filename, settings);

% Second, get a no-dynamics model
settings.dmd_mode = 'no_dynamics';
my_model_bu = CElegansModel(filename, settings);

% Plot
all_figs = cell(6,1);

all_figs{1} = my_model_bu.plot_colored_reconstruction();

interesting_neurons = 1:my_model_bu.original_sz(1);
fig_hist = figure('DefaultAxesFontSize', 14);
subplot(2,1,1)
my_model_bu.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,30])
xlim([0,1])
title('Correlation coefficients for the no-dynamics model')
subplot(2,1,2)
my_model_base.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,30])
xlim([0,1])
title('Correlation coefficients for the base model')

all_figs{2} = fig_hist;

all_figs{3} = my_model_bu.plot_reconstruction_interactive(false, 'AVAR');
title('AVAR reconstruction for the no-dynamics model')
all_figs{4} = my_model_base.plot_reconstruction_interactive(false, 'AVAR');
title('AVAR reconstruction for the base model')
all_figs{5} = my_model_bu.plot_reconstruction_interactive(false, 'VA0');
title('VA (motor neuron) reconstruction for the no-dynamics model')
all_figs{6} = my_model_base.plot_reconstruction_interactive(false, 'VA0');
title('VA (motor neuron) reconstruction for the base model')

% Save
to_save = false;
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\scratch_figures\';
all_fnames = {'PCA_no_dynamics_no_sparse',...
    'histogram_model_vs_no_dynamics_no_sparse',...
    'AVAR_no_dynamics',...
    'AVAR_base_model',...
    'VA_no_dynamics',...
    'VA_base_model'};
if to_save
    pause
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = [foldername all_fnames{i}];
        this_fig = all_figs{i};
        saveas(this_fig, fname, 'png');
    end
end

%==========================================================================


%% Compare to a no-dynamics fit (with sparse)
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'dmd_mode','func_DMDc');
settings.global_signal_mode = 'ID_binary';

% First get a baseline model (no sparse signals)
my_model_base = CElegansModel(filename, settings);

% Second, get a no-dynamics model
settings.dmd_mode = 'no_dynamics';
my_model_bu = CElegansModel(filename, settings);

% Plot
my_model_bu.plot_colored_reconstruction();
my_model_bu.plot_reconstruction_interactive();

figure
subplot(2,1,1)
all_corr_bu = my_model_bu.calc_correlation_matrix();
histogram(all_corr_bu, 'BinWidth', 0.05);
xlim([0,1])
ylim([0,25])
title('Correlation coefficients for the no-dynamics model (+sparse)')
subplot(2,1,2)
all_corr_base = my_model_base.calc_correlation_matrix();
histogram(all_corr_base, 'BinWidth', 0.05);
xlim([0,1])
ylim([0,25])
title('Correlation coefficients for the base model (+sparse)')

names = my_model_bu.get_names();
ind = ~cellfun(@isempty, names);
figure
subplot(2,1,1)
histogram(all_corr_bu(ind), 'BinWidth', 0.05);
xlim([0,1])
ylim([0,10])
title('Correlation coefficients for the no-dynamics model (named only) (+sparse)')
subplot(2,1,2)
histogram(all_corr_base(ind), 'BinWidth', 0.05);
xlim([0,1])
ylim([0,10])
title('Correlation coefficients for the base model (named only) (+sparse)')

%==========================================================================


%% Compare to a no-dynamics fit (with sparse; enforce diagonal B_sparse)
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_separate_sparse_from_data', false,...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'enforce_diagonal_sparse_B', true,...
    'offset_control_signal', true,...
    'dmd_mode','sparse');
settings.global_signal_mode = 'ID_binary';

% First get a baseline model (no sparse signals)
my_model_base = CElegansModel(filename, settings);

% Second, get a no-dynamics model
settings.dmd_mode = 'no_dynamics_sparse';
ad_settings = struct('sparsity_goal', 0.999);
settings.AdaptiveDmdc_settings = ad_settings;
my_model_bu = CElegansModel(filename, settings);

% Plot
all_figs = cell(4,1);

all_figs{1} = my_model_base.plot_colored_data();
all_figs{2} = my_model_base.plot_colored_reconstruction();
title('Reconstruction with base model')
all_figs{3} = my_model_bu.plot_colored_reconstruction();
title('Reconstruction with no dynamics (only control)')

% interesting_neurons = {'VA'};
interesting_neurons = 1:my_model_base.original_sz(1);
fig_hist = figure('DefaultAxesFontSize', 14);
subplot(2,1,1)
my_model_bu.plot_correlation_histogram(false, interesting_neurons, fig_hist);
xlim([0,1])
ylim([0,35])
title('Correlation coefficients for the no-dynamics model (+sparse)')
subplot(2,1,2)
my_model_base.plot_correlation_histogram(false, interesting_neurons, fig_hist);
xlim([0,1])
ylim([0,35])
title('Correlation coefficients for the base model (+sparse)')

all_figs{4} = fig_hist;

% Save
to_save = true;
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\scratch_figures\';
my_viewpoint = [0, 80];
all_fnames = {'PCA_data',...
    'PCA_constrained_model',...
    'PCA_constrained_no_dynamics',...
    'histogram_model_vs_no_dynamics'};
if to_save
    pause
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = [foldername all_fnames{i}];
        this_fig = all_figs{i};
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Model with reversal B-matrix entries removed
all_neurons = {'AVA','AVE','RIM','AIB'};
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_separate_sparse_from_data', false,...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'enforce_zero_entries', { {{all_neurons,'ID_binary'}} },...
    'enforce_diagonal_sparse_B', false,...
    'lambda_sparse', 0,...
    'offset_control_signal', true,...
    'dmd_mode','sparse');
settings.global_signal_mode = 'ID_binary';

my_model_no_REV = CElegansModel(filename, settings);

% Plot
fig = my_model_no_REV.plot_correlation_histogram(false, all_neurons);
title('Histogram of correlations with no reversal neurons')
xlim([-0.1, 1])
ylim([0, 20])

% Save
to_save = false;
if to_save
    foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\scratch_figures\';
    fname = [foldername 'histogram_no_reversal_controllers'];
    pause
    saveas(fig, fname, 'png');
end
%==========================================================================


%% Model with some B-matrix entries removed (systematic)
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_separate_sparse_from_data', false,...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'enforce_diagonal_sparse_B', false,...
    'lambda_sparse', 0,...
    'offset_control_signal', true,...
    'dmd_mode','sparse');
settings.global_signal_mode = 'ID_binary';

% Loop through
all_neurons = {'AVA','AVE','RIM','AIB'};
all_figs = cell(2*length(all_neurons), 1);
enforce_zero_entries =  { {{},'ID_binary'} };
all_models = cell(length(all_neurons),1);

for i = 1:length(all_neurons)
    this_set = setdiff(all_neurons, all_neurons{i});
    enforce_zero_entries{1}{1} = this_set;
    settings.enforce_zero_entries = enforce_zero_entries;
    all_models{i} = CElegansModel(filename, settings);
    
    all_figs{2*i-1} = ...
        all_models{i}.plot_correlation_histogram(false, all_neurons);
    title(sprintf('Correlations for model with control on only %s',...
        all_neurons{i}))
    xlim([-0.1, 1])
    ylim([0, 20])
    
    all_figs{2*i} = ...
        all_models{i}.plot_reconstruction_interactive(false, 'AVAR');
    title(sprintf('Reconstruction of AVAR with control on only %s',...
        all_neurons{i}))
    
    drawnow
end
    
% Save
to_save = false;
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\scratch_figures\';
if to_save
    pause
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        if mod(i, 2) == 1
            fname = sprintf('%shistogram_only_%s_rev_controllers',...
                foldername, all_neurons{ceil(i/2)});
        else
            fname = sprintf('%sreconstruction_only_%s_rev_controllers',...
                foldername, all_neurons{ceil(i/2)});
        end
        this_fig = all_figs{i};
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Use only gradients as controllers; compare to no-dynamics
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    ...'augment_data', 20,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','sparse',...
    'dmd_mode','func_DMDc',...
    ...'dmd_mode', 'tdmd',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_grad_only';

% First get a baseline model (no sparse signals)
my_model_base = CElegansModel(filename, settings);

% Second, get a no-dynamics model
settings.dmd_mode = 'no_dynamics';
my_model_bu = CElegansModel(filename, settings);

% Plot things
% my_model_base.plot_reconstruction_interactive();
% my_model_bu.plot_reconstruction_interactive();

all_figs = cell(3,1);

all_figs{1} = my_model_base.plot_colored_reconstruction();
title('Reconstruction for the base model')
all_figs{2} = my_model_bu.plot_colored_reconstruction();
title('Reconstruction for the no-dynamics model')

interesting_neurons = 1:my_model_bu.original_sz(1);
fig_hist = figure('DefaultAxesFontSize', 14);
subplot(2,1,1)
my_model_bu.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,40])
xlim([0,1])
title('Correlation coefficients for the no-dynamics model')
subplot(2,1,2)
my_model_base.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,40])
xlim([0,1])
title('Correlation coefficients for the base model')

all_figs{3} = fig_hist;

% Separate plot
% settings = struct('to_subtract_diagonal',true);
% compare_obj = compare_connectome(my_model_base.AdaptiveDmdc_obj, settings);
% compare_obj.plot_imagesc()

% Save
to_save = false;
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\scratch_figures\';
all_fnames = {'PCA_base_model_no_sparse_gradient',...
    'PCA_no_dynamics_no_sparse_gradient',...
    'histogram_model_vs_no_dynamics_no_sparse_gradient'};
if to_save
    pause
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = [foldername all_fnames{i}];
        this_fig = all_figs{i};
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Add neurons to the controller; nothing else
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
designated_controller_channels = {...
    'Reversal_neurons', {'AIB'},...
    'Turning_neurons', {'SMD', 'RIV'},...
    'Sleeping_neurons', {'RIS', 'ALA'}...
    'Forward_neurons', {'AVB'},... {'RIB'},...
    };
% designated_controller_channels = {...
%     ...'Reversal_neurons', {'AIB'},...
%     'Turning_neurons', {'SMD'},..., 'RIV'},...
%     ...'Sleeping_neurons', {'RIS', 'ALA'}...
%     ...'Forward_neurons', {'AVB'},... {'RIB'},...
%     };
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'designated_controller_channels', {designated_controller_channels},...
    ...'augment_data', 20,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','sparse',...
    'dmd_mode','func_DMDc',...
    ...'dmd_mode','tdmd',...
    'lambda_sparse',0);
settings.global_signal_mode = 'None';

% First get a baseline model (no sparse signals)
my_model_base = CElegansModel(filename, settings);

% Second, get a no-dynamics model
settings.dmd_mode = 'no_dynamics';
my_model_bu = CElegansModel(filename, settings);

% Plot things
% my_model_base.plot_reconstruction_interactive();
% my_model_bu.plot_reconstruction_interactive();

all_figs = cell(3,1);

% all_figs{1} = my_model_base.plot_colored_reconstruction();
% title('Reconstruction for the base model')
% all_figs{2} = my_model_bu.plot_colored_reconstruction();
% title('Reconstruction for the no-dynamics model')

interesting_neurons = 1:my_model_bu.dat_sz(1);
fig_hist = figure('DefaultAxesFontSize', 14);
subplot(2,1,1)
my_model_bu.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,40])
xlim([-0.2,1])
title('Correlation coefficients for the no-dynamics model')
subplot(2,1,2)
my_model_base.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,40])
xlim([-0.2,1])
title('Correlation coefficients for the model with neuron controllers')

all_figs{3} = fig_hist;



%==========================================================================


%% Use only 3 positive transitions as controllers; compare to no-dynamics
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    ...'augment_data', 20,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','sparse',...
    'dmd_mode','func_DMDc',...
    ...'dmd_mode', 'tdmd',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model (no sparse signals)
my_model_base = CElegansModel(filename, settings);

% Second, get a no-dynamics model
settings.dmd_mode = 'no_dynamics';
my_model_bu = CElegansModel(filename, settings);

% Plot things
my_model_base.plot_reconstruction_interactive();
% my_model_bu.plot_reconstruction_interactive();

all_figs = cell(3,1);

all_figs{1} = my_model_base.plot_colored_reconstruction();
title('Reconstruction for the base model')
% all_figs{2} = my_model_bu.plot_colored_reconstruction();
% title('Reconstruction for the no-dynamics model')

interesting_neurons = 1:my_model_bu.original_sz(1);
fig_hist = figure('DefaultAxesFontSize', 14);
subplot(2,1,1)
my_model_bu.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,40])
xlim([0,1])
title('Correlation coefficients for the no-dynamics model')
subplot(2,1,2)
my_model_base.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,40])
xlim([0,1])
title('Correlation coefficients for the base model')

all_figs{3} = fig_hist;

% Separate plot
% settings = struct('to_subtract_diagonal',true);
% compare_obj = compare_connectome(my_model_base.AdaptiveDmdc_obj, settings);
% compare_obj.plot_imagesc()

% Save
to_save = false;
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\scratch_figures\';
all_fnames = {'PCA_base_model_no_sparse_gradient',...
    'PCA_no_dynamics_no_sparse_gradient',...
    'histogram_model_vs_no_dynamics_no_sparse_gradient'};
if to_save
    pause
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = [foldername all_fnames{i}];
        this_fig = all_figs{i};
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Use all 7 of the positive transitions as controllers; compare to no-dynamics
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',true,...
    'use_deriv',false,...
    ...'augment_data', 20,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','sparse',...
    'dmd_mode','func_DMDc',...
    ...'dmd_mode', 'tdmd',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0,...
    'global_signal_subset', {{'REV', 'FWD', 'SLOW', 'DT', 'VT'}});
settings.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model (no sparse signals)
my_model_base = CElegansModel(filename, settings);

% Second, get a no-dynamics model
settings.dmd_mode = 'no_dynamics';
my_model_bu = CElegansModel(filename, settings);

% Plot things
my_model_base.plot_reconstruction_interactive();
% my_model_bu.plot_reconstruction_interactive();

all_figs = cell(3,1);

all_figs{1} = my_model_base.plot_colored_reconstruction();
title('Reconstruction for the base model')
% all_figs{2} = my_model_bu.plot_colored_reconstruction();
% title('Reconstruction for the no-dynamics model')

interesting_neurons = 1:my_model_bu.original_sz(1);
fig_hist = figure('DefaultAxesFontSize', 14);
subplot(2,1,1)
my_model_bu.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,40])
xlim([0,1])
title('Correlation coefficients for the no-dynamics model')
subplot(2,1,2)
my_model_base.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,40])
xlim([0,1])
title('Correlation coefficients for the base model')

all_figs{3} = fig_hist;

% Separate plot
% settings = struct('to_subtract_diagonal',true);
% compare_obj = compare_connectome(my_model_base.AdaptiveDmdc_obj, settings);
% compare_obj.plot_imagesc()

% Save
to_save = false;
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\scratch_figures\';
all_fnames = {'PCA_base_model_no_sparse_gradient',...
    'PCA_no_dynamics_no_sparse_gradient',...
    'histogram_model_vs_no_dynamics_no_sparse_gradient'};
if to_save
    pause
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = [foldername all_fnames{i}];
        this_fig = all_figs{i};
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Use only 3 positive transition SPIKES as controllers; compare to no-dynamics
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','sparse',...
    'dmd_mode','func_DMDc',...
    ...'dmd_mode', 'tdmd',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model (no sparse signals)
my_model_base = CElegansModel(filename, settings);

% Second, get a no-dynamics model
settings.dmd_mode = 'no_dynamics';
my_model_bu = CElegansModel(filename, settings);

% Plot things
my_model_base.plot_reconstruction_interactive();
% my_model_bu.plot_reconstruction_interactive();

all_figs = cell(3,1);

all_figs{1} = my_model_base.plot_colored_reconstruction();
title('Reconstruction for the base model')
% all_figs{2} = my_model_bu.plot_colored_reconstruction();
% title('Reconstruction for the no-dynamics model')

interesting_neurons = 1:my_model_bu.original_sz(1);
fig_hist = figure('DefaultAxesFontSize', 14);
subplot(2,1,1)
my_model_bu.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,40])
xlim([0,1])
title('Correlation coefficients for the no-dynamics model')
subplot(2,1,2)
my_model_base.plot_correlation_histogram(false, interesting_neurons, fig_hist);
ylim([0,40])
xlim([0,1])
title('Correlation coefficients for the base model')

all_figs{3} = fig_hist;

% Separate plot
% settings = struct('to_subtract_diagonal',true);
% compare_obj = compare_connectome(my_model_base.AdaptiveDmdc_obj, settings);
% compare_obj.plot_imagesc()

% Save
to_save = false;
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\scratch_figures\';
all_fnames = {'PCA_base_model_no_sparse_gradient',...
    'PCA_no_dynamics_no_sparse_gradient',...
    'histogram_model_vs_no_dynamics_no_sparse_gradient'};
if to_save
    pause
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        fname = [foldername all_fnames{i}];
        this_fig = all_figs{i};
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Use only 3 positive transitions as controllers and neurons
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
designated_controller_channels = {...
    ...'Turning_neurons', {'SMD', 'RIV'},...
    ...'Turning_neurons', {'RIV'},...
    'Sleeping_neurons', {'RIS', 'ALA'},...
    'Sensory_neurons', {'ASK', 'AVF'}, ...
    ...'Speed_neurons', {'RIB'},...
    'Unnamed_sensory_neurons', [1 5 15 64 121 122],...
    };
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',true,...
    'use_deriv',false,...
    'designated_controller_channels', {designated_controller_channels},...
    ...'augment_data', 3,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 0,...
    ...'dmd_mode','sparse',...
    'dmd_mode','func_DMDc',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'None';

% First get a baseline model (no sparse signals)
my_model_3_kicks = CElegansModel(filename, settings);

% Plot things
% my_model_3_kicks.plot_reconstruction_interactive();
my_model_3_kicks.plot_reconstruction_interactive(true, 'AVBL');

%==========================================================================


%% Use only 3 positive transitions; try to predict these kicks
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    ...'dmd_mode','func_DMDc',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model (no sparse signals)
my_model_3_kicks = CElegansModel(filename, settings);

% Plot things
% my_model_base.plot_reconstruction_interactive();

% How well are the control signals predicted by a linear function?
x = my_model_3_kicks.AdaptiveDmdc_obj.x_len;
B_prime = my_model_3_kicks.AdaptiveDmdc_obj.A_original((x+1):end,:);
figure
imagesc(B_prime(:,1:x))
colorbar
xlabel('Neuron number')
ylabel('Control signal index')
title('Predicting the control signal from data');

%---------------------------------------------
% Linear reconstruction from the model
%---------------------------------------------
which_ctr = 3;
ctr = my_model_3_kicks.control_signal(which_ctr,:);
ctr_reconstruct = zeros(size(ctr));
ctr_reconstruct(1) = ctr(1);
for i = 2:length(ctr)
    prev_tmp = my_model_3_kicks.dat_with_control(:,i-1);
    prev_tmp(x+1) = ctr_reconstruct(i-1); % Replace only this control signal
    ctr_reconstruct(i) = B_prime(1,:)*prev_tmp;
end

figure
ctr = my_model_3_kicks.control_signal(1,:);
plot(ctr)
hold on
plot(ctr_reconstruct, '--', 'Linewidth',2)
legend({'True control signal','Reconstructed control signal'})


%---------------------------------------------
% Linear reconstruction from a direct fit to DATA
%---------------------------------------------
U2 = my_model_3_kicks.control_signal(:,2:end);
X1 = my_model_3_kicks.dat(:,1:end-1);
B_prime2 = U2/X1;

ctr = my_model_3_kicks.control_signal(which_ctr,:);
ctr_reconstruct = zeros(size(ctr));
ctr_reconstruct(1) = ctr(1);
for i = 2:length(ctr)
    ctr_reconstruct(i) = B_prime2(which_ctr,:) * X1(:,i-1);
end

figure
ctr = my_model_3_kicks.control_signal(which_ctr,:);
plot(ctr)
hold on
plot(ctr_reconstruct, 'Linewidth',2)
title(sprintf('Reconstruction of control signal %d',which_ctr))
legend({'True','Reconstructed'})

%---------------------------------------------
% LASSO from a direct fit to DATA
%---------------------------------------------
U2 = my_model_3_kicks.control_signal(:,2:end);
X1 = my_model_3_kicks.dat(:,1:end-1);
disp('Fitting lasso models...')
all_intercepts = zeros(size(U2,1),1);
which_fit = 8;
for i = size(U2,1):-1:1
%     B_prime_lasso(i,:) = U2(i,:)/X1;
    [all_fits, fit_info] = lasso(X1', U2(i,:), 'NumLambda',10);
    all_intercepts(i) = fit_info.Intercept(which_fit);
    B_prime_lasso(i,:) = all_fits(:,which_fit); % Which fit = determined by eye
%     B_prime_lasso(i,:) = lasso(X1', zeros(size(X1,2),1));
end

ctr = my_model_3_kicks.control_signal(which_ctr,:);
ctr_reconstruct = zeros(size(ctr));
ctr_reconstruct(1) = ctr(1);
for i = 2:length(ctr)
    ctr_reconstruct(i) = B_prime_lasso(which_ctr,:) * X1(:,i-1);
end
ctr_reconstruct = ctr_reconstruct + all_intercepts(which_ctr);

figure
ctr = my_model_3_kicks.control_signal(which_ctr,:);
plot(ctr)
hold on
plot(ctr_reconstruct, 'Linewidth',2)
title(sprintf('Sparse reconstruction of control signal %d',which_ctr))
legend({'True','Reconstructed'})

%---------------------------------------------
% Linear reconstruction from a direct fit to RECONSTRUCTED data
%---------------------------------------------
U2 = my_model_3_kicks.control_signal(:,2:end);
X1 = my_model_3_kicks.dat(:,1:end-1);
% X1 = my_model_3_kicks.AdaptiveDmdc_obj.reset_threshold
B_prime3 = U2/X1;

which_ctr = 1;
ctr = my_model_3_kicks.control_signal(which_ctr,:);
ctr_reconstruct = zeros(size(ctr));
ctr_reconstruct(1) = ctr(1);
for i = 2:length(ctr)
    ctr_reconstruct(i) = B_prime3(which_ctr,:) * ...
        my_model_3_kicks.dat(:,i-1);
end

figure
ctr = my_model_3_kicks.control_signal(which_ctr,:);
plot(ctr)
hold on
plot(ctr_reconstruct, 'Linewidth',2)
title(sprintf('Reconstruction of control signal %d',which_ctr))
legend({'True','Reconstructed'})


% Also plot the matrix
figure
% imagesc(B_prime2);
imagesc(B_prime_lasso);
colorbar

names = my_model_3_kicks.get_names();
xticks(1:x)
xticklabels(names)
xtickangle(90)
yticks(1:5)
ind = contains(my_model_3_kicks.state_labels_key, {'REV', 'DT', 'VT'});
yticklabels(my_model_3_kicks.state_labels_key(ind))
%==========================================================================


%% Reconstruct the 3 transition signals using neurons; add those in as controllers
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    ...'dmd_mode','func_DMDc',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model (no sparse signals)
my_model_3_kicks = CElegansModel(filename, settings);

%---------------------------------------------
% Use lasso to get encoding neurons
%---------------------------------------------

% Get lasso models
U2 = my_model_3_kicks.control_signal(:,2:end);
X1 = my_model_3_kicks.dat(:,1:end-1);
disp('Fitting lasso models...')
all_intercepts = zeros(size(U2,1),1);
which_fit = 19;
for i = size(U2,1):-1:1
    [all_fits, fit_info] = lasso(X1', U2(i,:), 'NumLambda',20);
    all_intercepts(i) = fit_info.Intercept(which_fit);
    B_prime_lasso(i,:) = all_fits(:,which_fit); % Which fit = determined by eye
end

sparsity_pattern = (B_prime_lasso==0);
% disp('Redo-ing fit with lasso sparsity mask')
% [n, m] = size(B_prime_lasso);
% cvx_begin
%     variable B(n, m)
%     minimize( norm(B*X1 - U2, 2) )
%     B(sparsity_pattern) == 0 %#ok<NOPTS,EQEFF>
% cvx_end

to_plot = true;
if to_plot
    % Get reconstructed signal and plot
    ctr = my_model_3_kicks.control_signal(which_ctr,:);
    ctr_reconstruct_raw = zeros(size(ctr));
    ctr_reconstruct_raw(1) = ctr(1);
    ctr_reconstruct_cvx = ctr_reconstruct_raw;
    for i = 2:length(ctr)
        ctr_reconstruct_raw(i) = B_prime_lasso(which_ctr,:) * X1(:,i-1);
        ctr_reconstruct_cvx(i) = B(which_ctr,:) * X1(:,i-1);
    end
    ctr_reconstruct_raw = ctr_reconstruct_raw + all_intercepts(which_ctr);

    figure
    ctr = my_model_3_kicks.control_signal(which_ctr,:);
    plot(ctr)
    hold on
    plot(ctr_reconstruct_raw, 'Linewidth',2)
    plot(ctr_reconstruct_cvx, 'Linewidth',2)
    title(sprintf('Sparse reconstruction of control signal %d',which_ctr))
    legend({'True','Reconstructed (lasso)','Reconstructed (cvx)'})

    % Also plot the matrix
    figure
    imagesc(B_prime_lasso);
    colorbar

    names = my_model_3_kicks.get_names();
    xticks(1:x)
    xticklabels(names)
    xtickangle(90)
    ind = contains(my_model_3_kicks.state_labels_key, {'REV', 'DT', 'VT'});
    yticks(1:length(ind))
    yticklabels(my_model_3_kicks.state_labels_key(ind))
    title('Matrix for reconstructing the control signal from data')
end

%---------------------------------------------
% Add "encoding neurons" back into model
%---------------------------------------------

designated_controller_channels = {...
    'Sensory_unnamed', [1 5 15],...
    'Sensory', {'RIS'},...
    'DT_neurons', find(sparsity_pattern(1,:)==0),...
    'VT_neurons', find(sparsity_pattern(2,:)==0),...
    'REV1_neurons', find(sparsity_pattern(3,:)==0),...
    'REV2_neurons', find(sparsity_pattern(4,:)==0),...
    'REVSUS_neurons', find(sparsity_pattern(5,:)==0)   };
settings.designated_controller_channels = designated_controller_channels;
my_model_encoding = CElegansModel(filename, settings);

my_model_encoding.plot_colored_reconstruction();
% my_model_encoding.plot_reconstruction_interactive(true, 'AVBL');


%==========================================================================


%% Same as above but with sensory and without trivial neurons
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    ...'dmd_mode','func_DMDc',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model (no sparse signals)
my_model_3_kicks = CElegansModel(filename, settings);

%---------------------------------------------
% Use lasso to get encoding neurons
%---------------------------------------------
trivial_neurons = {...
    'AVA', 'AVE', 'RIM', 'AIB',... % Reversal
    ...'SMD'... % Turning
    }; 
ind = 1:my_model_3_kicks.dat_sz(1);
ind(my_model_3_kicks.name2ind(trivial_neurons)) = [];

% Get lasso models
U2 = my_model_3_kicks.control_signal(:,2:end);
X1 = my_model_3_kicks.dat(ind,1:end-1);
disp('Fitting lasso models...')
all_intercepts = zeros(size(U2,1),1);
B_prime_lasso = zeros(size(U2*X1'));
which_fit = 18;
for i = size(U2,1):-1:1
    [all_fits, fit_info] = lasso(X1', U2(i,:), 'NumLambda',20);
    all_intercepts(i) = fit_info.Intercept(which_fit);
    B_prime_lasso(i,:) = all_fits(:,which_fit); % Which fit = determined by eye
end

sparsity_pattern = (B_prime_lasso==0);
% disp('Redo-ing fit with lasso sparsity mask')
% [n, m] = size(B_prime_lasso);
% cvx_begin
%     variable B(n, m)
%     minimize( norm(B*X1 - U2, 2) )
%     B(sparsity_pattern) == 0 %#ok<NOPTS,EQEFF>
% cvx_end

to_plot = true;
if to_plot
    which_ctr = 5;
    % Get reconstructed signal and plot
    ctr = my_model_3_kicks.control_signal(which_ctr,:);
    ctr_reconstruct_raw = zeros(size(ctr));
    ctr_reconstruct_raw(1) = ctr(1);
%     ctr_reconstruct_cvx = ctr_reconstruct_raw;
    for i = 2:length(ctr)
        ctr_reconstruct_raw(i) = B_prime_lasso(which_ctr,:) * X1(:,i-1);
%         ctr_reconstruct_cvx(i) = B(which_ctr,:) * X1(:,i-1);
    end
    ctr_reconstruct_raw = ctr_reconstruct_raw + all_intercepts(which_ctr);

    figure
    ctr = my_model_3_kicks.control_signal(which_ctr,:);
    plot(ctr)
    hold on
    plot(ctr_reconstruct_raw, 'Linewidth',2)
%     plot(ctr_reconstruct_cvx, 'Linewidth',2)
    title(sprintf('Sparse reconstruction of control signal %d',which_ctr))
    legend({'True','Reconstructed (lasso)','Reconstructed (cvx)'})

    % Also plot the matrix
    figure
    imagesc(B_prime_lasso);
    colorbar

    names = my_model_3_kicks.get_names(ind);
    xticks(1:length(ind))
    xticklabels(names)
    xtickangle(90)
    key_ind = contains(my_model_3_kicks.state_labels_key, {'REV', 'DT', 'VT'});
    yticks(1:length(key_ind))
    yticklabels(my_model_3_kicks.state_labels_key(key_ind))
    title('Matrix for reconstructing the control signal from data')
end

%---------------------------------------------
% Add "encoding neurons" back into model
%---------------------------------------------
designated_controller_channels = {...
    'Sensory_unnamed', [1 5 15],...
    'Sensory', {'RIS'},...
    'DT_neurons', find(sparsity_pattern(1,:)==0),...
    'VT_neurons', find(sparsity_pattern(2,:)==0),...
    'REV1_neurons', find(sparsity_pattern(3,:)==0),...
    'REV2_neurons', find(sparsity_pattern(4,:)==0),...
    'REVSUS_neurons', find(sparsity_pattern(5,:)==0)   };
settings.designated_controller_channels = designated_controller_channels;
my_model_encoding = CElegansModel(filename, settings);

my_model_encoding.plot_colored_reconstruction();
% my_model_encoding.plot_reconstruction_interactive(true, 'AVBL');


%==========================================================================


%% Same as above but with sensory and looping to remove trivial neurons
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    'dmd_mode','func_DMDc',...
    'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model (no sparse signals)
my_model_3_kicks = CElegansModel(filename, settings);

%---------------------------------------------
% Use lasso to get encoding neurons
%---------------------------------------------
sz = 50;
all_models = cell(sz,1);
sensory_neurons = [1 5 15];
trivial_neurons = sensory_neurons;
for i_trivial = 1:sz
    % Get lasso models
    U2 = my_model_3_kicks.control_signal(:,2:end);
    X1 = my_model_3_kicks.dat(:,1:end-1);
    X1(trivial_neurons,:) = 0;
    disp('Fitting lasso models...')
    all_intercepts = zeros(size(U2,1),1);
    B_prime_lasso = zeros(size(U2*X1'));
    which_fit = 19;
    for i = size(U2,1):-1:1
        [all_fits, fit_info] = lasso(X1', U2(i,:), 'NumLambda',20);
        all_intercepts(i) = fit_info.Intercept(which_fit);
        B_prime_lasso(i,:) = all_fits(:,which_fit); % Which fit = determined by eye
    end

    sparsity_pattern = (B_prime_lasso==0);

    to_plot = false;
    if to_plot
        which_ctr = 5;
        % Get reconstructed signal and plot
        ctr = my_model_3_kicks.control_signal(which_ctr,:);
        ctr_reconstruct_raw = zeros(size(ctr));
        ctr_reconstruct_raw(1) = ctr(1);
    %     ctr_reconstruct_cvx = ctr_reconstruct_raw;
        for i = 2:length(ctr)
            ctr_reconstruct_raw(i) = B_prime_lasso(which_ctr,:) * X1(:,i-1);
    %         ctr_reconstruct_cvx(i) = B(which_ctr,:) * X1(:,i-1);
        end
        ctr_reconstruct_raw = ctr_reconstruct_raw + all_intercepts(which_ctr);

        figure
        ctr = my_model_3_kicks.control_signal(which_ctr,:);
        plot(ctr)
        hold on
        plot(ctr_reconstruct_raw, 'Linewidth',2)
    %     plot(ctr_reconstruct_cvx, 'Linewidth',2)
        title(sprintf('Sparse reconstruction of control signal %d',which_ctr))
        legend({'True','Reconstructed (lasso)','Reconstructed (cvx)'})

        % Also plot the matrix
        figure
        imagesc(B_prime_lasso);
        colorbar

        names = my_model_3_kicks.get_names(ind);
        xticks(1:length(ind))
        xticklabels(names)
        xtickangle(90)
        key_ind = contains(my_model_3_kicks.state_labels_key, {'REV', 'DT', 'VT'});
        yticks(1:length(key_ind))
        yticklabels(my_model_3_kicks.state_labels_key(key_ind))
        title('Matrix for reconstructing the control signal from data')
    end

    %---------------------------------------------
    % Add "encoding neurons" back into model
    %---------------------------------------------
    designated_controller_channels = {...
        'Sensory_unnamed', sensory_neurons,...
        ...'Sensory', {'RIS'},...
        'DT_neurons', find(sparsity_pattern(1,:)==0),...
        'VT_neurons', find(sparsity_pattern(2,:)==0),...
        'REV1_neurons', find(sparsity_pattern(3,:)==0),...
        'REV2_neurons', find(sparsity_pattern(4,:)==0),...
        'REVSUS_neurons', find(sparsity_pattern(5,:)==0)   };
    settings.designated_controller_channels = designated_controller_channels;
    settings.global_signal_mode = 'None';
    all_models{i_trivial} = CElegansModel(filename, settings);
    
    %---------------------------------------------
    % Designate "trivial" neurons for next loop
    %---------------------------------------------
    [~, max_ind] = max(sum(abs(B_prime_lasso),1));
    fprintf('Removing neuron %d (name=%s)\n', ...
        max_ind, my_model_3_kicks.get_names(max_ind));
    trivial_neurons = [trivial_neurons max_ind]; %#ok<AGROW>

end

%---------------------------------------------
% Boxplot of all the correlations
%---------------------------------------------
figure
all_corr = [];
all_groups = [];
all_ctr_sz = zeros(sz,1);
for i = 1:sz
    this_dat = real(all_models{i}.calc_correlation_matrix().');
    all_corr = [all_corr this_dat];
    all_groups = [all_groups (i-1)*ones(size(this_dat))];
    all_ctr_sz(i) = size(all_models{i}.control_signal,1);
end

all_names = my_model_3_kicks.get_names(trivial_neurons(4:end));
all_names = [{'All present'} all_names(1:end-1)];
for i = 1:sz
    if isempty(all_names{i})
        all_names{i} = num2str(trivial_neurons(i));
    end
end

boxplot(all_corr, all_groups)
xticklabels(all_names)
xlabel('Neurons removed from controller')
xtickangle(90)

figure
plot(all_ctr_sz)
title('Size of control signal across models')

%==========================================================================


%% Using PCA to identify "unexpected" encoding neurons
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
ad_settings = struct(...
    'oscillation_threshold', 0.01);
    
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    'dmd_mode','func_DMDc',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';
original_sensory_neurons = ...
    [1 5 15 121 ... % unnamed 
    60 ... % ALA
    55 56 ... % ASK
    101 ... % RIS
    ];
designated_controller_channels = {...
    'Sensory', original_sensory_neurons};
settings.designated_controller_channels = designated_controller_channels;

% First get a baseline model
my_model_3_kicks = CElegansModel(filename, settings);

%---------------------------------------------
% Use PCA to get unexpected neurons; hope they encode something!
%---------------------------------------------
num_PCA_modes = 3;
PCA_threshold = 5e-2;
% Get PCA modes; note that these will not be the original numbering because
% some sensory neurons have already been removed
[U, S, V] = plotSVD(my_model_3_kicks.dat);

all_encoding_neurons = cell(num_PCA_modes,1);
% Must be below threshold on ALL these PCA modes
for i = 1:num_PCA_modes
    u = U(:,i);
    all_encoding_neurons{i} = find( abs(u)<PCA_threshold );
    if i>1
        encoding_neurons = ...
            intersect(encoding_neurons, all_encoding_neurons{i});
    else
        encoding_neurons = all_encoding_neurons{1};
    end
end

fprintf('Adding these neurons into the controller:\n')
disp(my_model_3_kicks.get_names(encoding_neurons))

% Fix indexing for the originally imported data
for i = 1:length(original_sensory_neurons)
    encoding_neurons = encoding_neurons + ...
        double( encoding_neurons>=original_sensory_neurons(i) );
end

designated_controller_channels = {...
    'Sensory_unnamed', original_sensory_neurons};
designated_controller_channels = [ designated_controller_channels ...
    'encoding_neurons_all_modes', encoding_neurons' ];

to_plot = false;
if to_plot
    which_ctr = 5;
    % Get reconstructed signal and plot
    ctr = my_model_3_kicks.control_signal(which_ctr,:);
    ctr_reconstruct_raw = zeros(size(ctr));
    ctr_reconstruct_raw(1) = ctr(1);
%     ctr_reconstruct_cvx = ctr_reconstruct_raw;
    for i = 2:length(ctr)
        ctr_reconstruct_raw(i) = B_prime_lasso(which_ctr,:) * X1(:,i-1);
%         ctr_reconstruct_cvx(i) = B(which_ctr,:) * X1(:,i-1);
    end
    ctr_reconstruct_raw = ctr_reconstruct_raw + all_intercepts(which_ctr);

    figure
    ctr = my_model_3_kicks.control_signal(which_ctr,:);
    plot(ctr)
    hold on
    plot(ctr_reconstruct_raw, 'Linewidth',2)
%     plot(ctr_reconstruct_cvx, 'Linewidth',2)
    title(sprintf('Sparse reconstruction of control signal %d',which_ctr))
    legend({'True','Reconstructed (lasso)','Reconstructed (cvx)'})

    % Also plot the matrix
    figure
    imagesc(B_prime_lasso);
    colorbar

    names = my_model_3_kicks.get_names(ind);
    xticks(1:length(ind))
    xticklabels(names)
    xtickangle(90)
    key_ind = contains(my_model_3_kicks.state_labels_key, {'REV', 'DT', 'VT'});
    yticks(1:length(key_ind))
    yticklabels(my_model_3_kicks.state_labels_key(key_ind))
    title('Matrix for reconstructing the control signal from data')
end

%---------------------------------------------
% Add "encoding neurons" into model
%---------------------------------------------

settings.global_signal_mode = 'None';
settings.designated_controller_channels = designated_controller_channels;
my_model_encoding = CElegansModel(filename, settings);

my_model_encoding.plot_colored_reconstruction();
my_model_encoding.plot_reconstruction_interactive();
%==========================================================================


%% Identify "unique information" neurons via single-neuron reconstructions
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);
ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    'dmd_mode','func_DMDc',...
    'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'None';

my_model_bu = CElegansModel(filename, settings);

%---------------------------------------------
% Reconstruct neurons individually
%---------------------------------------------
% other_neurons = 1:size(dat_struct.traces,2);
other_neurons = 1:108; % For use with autocorrelation_noise_threshold=0.3 (worm 5)
all_corr = zeros(size(other_neurons));
all_residuals = zeros(length(other_neurons), size(dat_struct.traces,1));
for i = other_neurons
    this_controller_set = other_neurons;
    this_controller_set(i) = [];
    designated_controller_channels = {'Other_neurons', this_controller_set};

    settings.designated_controller_channels = designated_controller_channels;

    my_model_1_neuron = CElegansModel(filename, settings);
    
    all_corr(i) = my_model_1_neuron.calc_correlation_matrix();
    all_residuals(i,:) = ...
        my_model_1_neuron.AdaptiveDmdc_obj.calc_reconstruction_error('residual_vector');
end

[sort_corr, sort_ind] = sort(all_corr);
all_names = my_model_bu.get_names();

to_plot = false;
if to_plot
    figure
    plot(sort_corr)
    title('Correlation coefficients for reconstructions using all other neurons')

    all_names = my_model_bu.get_names();
    xticks(other_neurons)
    xticklabels(all_names(sort_ind))
    xlabel('Neuron')
    xtickangle(90)
end

%---------------------------------------------
% Check the "unique information" in a neuron
%---------------------------------------------

% Check the quality of the reconstructions in 2 ways
all_chi_p = zeros(size(other_neurons));
all_acf_res = zeros(size(other_neurons));
all_acf_dat = zeros(size(other_neurons));
for i = other_neurons
    this_dat = all_residuals(i,:);
    [~, all_chi_p(i)] = chi2gof(this_dat);
    all_acf_res(i) = mean(acf(this_dat', 5, false));
    
    all_acf_dat(i) = mean(acf(my_model_bu.dat(i,:)', 1, false));
end

if to_plot
    figure
    plot(all_chi_p(sort_ind), 'o')
    hold on
    plot(1 - all_acf_res(sort_ind), '*')
    title('Analysis of residuals')
    legend({'Chi squared p values', '1 - Mean of 5-step autocorrelations'})
    xticks(other_neurons)
    xticklabels(all_names(sort_ind))
    xlabel('Neuron')
    xtickangle(90)
    
    figure
    plot(all_chi_p(sort_ind), all_acf_res(sort_ind), 'o')
    ylabel('Autocorrelations')
    xlabel('Chi squared value')
    title('Analysis of residuals')
   
    figure
    all_corr_ratios = all_corr./all_acf_dat;
    [sort_corr_ratio, sort_ind_ratio] = sort(all_corr_ratios);
    plot(sort_corr_ratio)
    xticks(other_neurons)
    xticklabels(all_names(sort_ind_ratio))
    xlabel('Neuron')
    xtickangle(90)
    title('Ratio of reconstruction correlations to autocorrelation')
    
    % Look at the ratio of signal variance to total variance in each neuron
    [~, dat_signal, dat_noise] = calc_snr(my_model_bu.dat);
    snr_vec = var(dat_signal-mean(dat_signal,2), 0, 2) ./...
        var(my_model_bu.dat-mean(my_model_bu.dat,2), 0, 2);
    figure;
    plot(snr_vec(sort_ind))
    hold on
    plot(sort_corr)
    legend({'Percentage of signal in each neuron','Reconstruction-data correlation'})
    xticks(other_neurons)
    xticklabels(all_names(sort_ind))
    xlabel('Neuron')
    xtickangle(90)
    
    % Bar chart showing snr and best reconstruction
    reconstructed_signal = min(...
        [snr_vec(sort_ind) sort_corr'], [], 2);
    other_signal = sort_corr' - snr_vec(sort_ind);
    ind = other_signal>0;
    correlated_noise = zeros(size(snr_vec));
    correlated_noise(ind) = other_signal(ind);
    independent_signal = zeros(size(snr_vec));
    independent_signal(~ind) = -other_signal(~ind);
    independent_noise = 1 - correlated_noise - reconstructed_signal - independent_signal;
    figure
    bar([reconstructed_signal, correlated_noise,...
        independent_signal, independent_noise], 'stacked')
    legend({'Reconstructable signal', 'Reconstructable (correlated) noise',...
        'Independent signal', 'Independent noise'})
    xticks(other_neurons)
    xticklabels(all_names(sort_ind))
    xlabel('Neuron')
    xtickangle(90)
    
    
    % Simpler bar chart ignoring correlated noise
    [~, sort_snr_ind] = sort(snr_vec);
    reconstructed_signal = min(...
        [snr_vec(sort_snr_ind) all_corr(sort_snr_ind)'], [], 2);
    other_signal = all_corr(sort_snr_ind)' - snr_vec(sort_snr_ind);
    ind = other_signal>0;
    independent_signal = zeros(size(snr_vec));
    independent_signal(~ind) = -other_signal(~ind);
    figure('DefaultAxesFontSize',14)
    independent_noise = 1 - reconstructed_signal - independent_signal;
    bar([reconstructed_signal,...
        independent_signal,...
        independent_noise], 'stacked')
    legend({'Reconstructable signal',...
        'Independent signal', 'Independent noise'})
    xticks(other_neurons)
    xticklabels(all_names(sort_snr_ind))
    xlabel('Neuron')
    xtickangle(90)
end

% Plot different numbers of PC reconstructions
to_save_plots = true;
if to_save_plots
    fname = 'variance_capture_%d_PCs';
    num_PCs_vec = [3 4 5 6 10 20 35];
%     num_PCs_vec = [3 4 5];
    [~, dat_signal_baseline] = calc_snr(my_model_bu.dat);
    snr_vec_baseline = ...
        var(dat_signal_baseline-mean(dat_signal_baseline,2), 0, 2) ./...
        var(my_model_bu.dat-mean(my_model_bu.dat,2), 0, 2);
    for i = 1:length(num_PCs_vec)
        num_PCs = num_PCs_vec(i);
        % Look at the ratio of signal variance to total variance in each neuron
        [~, dat_signal, dat_noise] = calc_snr(my_model_bu.dat, num_PCs);
        snr_vec = var(dat_signal-mean(dat_signal,2), 0, 2) ./...
            var(my_model_bu.dat-mean(my_model_bu.dat,2), 0, 2);
        
        % Simpler bar chart with baseline of total signal level
        [sort_snr, sort_snr_ind] = sort(snr_vec);
        other_signal = snr_vec_baseline(sort_snr_ind) - sort_snr;
        other_signal = other_signal.*(other_signal>0);
        fig = figure('DefaultAxesFontSize',14, ...
            'units','normalized','outerposition',[0 0 1 1]);
        bar([sort_snr,...
            other_signal], 'stacked')
        legend({'Signal in this number of PCs',...
            'Total signal'}, 'Location', 'southeast')
        xticks(other_neurons)
        xticklabels(all_names(sort_snr_ind))
        xlabel('Neuron')
        xtickangle(90)
        title(sprintf('Fraction of variance per neuron for %d PCs',...
            num_PCs))
        drawnow
        
        saveas(fig, sprintf(fname, num_PCs), 'png')
    end
end

%---------------------------------------------
% Add in several that aren't reconstructed by X number of PCs
%---------------------------------------------
num_PCs = 3;
% num_neurons_to_add = 62; % RIS for PC=3
num_neurons_to_add = 80;
% Look at the ratio of signal variance to total variance in each neuron
[~, dat_signal, dat_noise] = calc_snr(my_model_bu.dat, num_PCs);
snr_vec = var(dat_signal-mean(dat_signal,2), 0, 2) ./...
    var(my_model_bu.dat-mean(my_model_bu.dat,2), 0, 2);

% Simpler bar chart with baseline of total signal level
[~, sort_snr_ind] = sort(snr_vec);

designated_controller_channels = {...
    'Unreconstructed_neurons', sort_snr_ind(1:num_neurons_to_add)};
settings.designated_controller_channels = designated_controller_channels;
settings.filter_window_global = 1;
my_model_encoding = CElegansModel(filename, settings);

my_model_encoding.plot_colored_reconstruction();
my_model_encoding.plot_reconstruction_interactive(false);

%---------------------------------------------
% Add in low reconstructed neurons and some hand-picked ones
%---------------------------------------------
num_PCs = 3;
num_neurons_to_add = 30;
% Look at the ratio of signal variance to total variance in each neuron
[~, dat_signal, dat_noise] = calc_snr(my_model_bu.dat, num_PCs);
snr_vec = var(dat_signal-mean(dat_signal,2), 0, 2) ./...
    var(my_model_bu.dat-mean(my_model_bu.dat,2), 0, 2);

% Simpler bar chart with baseline of total signal level
[~, sort_snr_ind] = sort(snr_vec);

ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','func_DMDc',...
    'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'None';
designated_controller_channels = {...
    'Unreconstructed_neurons', sort_snr_ind(1:num_neurons_to_add)',...
    'Suspicious_neurons', [108],...
    'Sensory_neurons', {'ASK'},...
    'Unique_signal',{'SMD', 'AVF', 'RIS', 'RIV'}};
settings.designated_controller_channels = designated_controller_channels;
my_model_encoding = CElegansModel(filename, settings);

my_model_encoding.plot_colored_reconstruction();
my_model_encoding.plot_reconstruction_interactive(false);
%==========================================================================


%% Look at controllability subspace of the transition kicks model
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);
ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    'dmd_mode','func_DMDc',...
    'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

my_model_base = CElegansModel(filename, settings);


% Get the rank of the controllability subspace
signal_rank = rank(my_model_base.dat);
n = my_model_base.dat_sz(1);
A = my_model_base.AdaptiveDmdc_obj.A_separate(1:n, 1:n);
B = my_model_base.AdaptiveDmdc_obj.A_separate(1:n, (n+1):end);
ctr_mat = ctrb(A, B);
controllability_rank = rank(ctr_mat);

fprintf('Signal rank: %d\nControllability subspace rank: %d\n',...
    signal_rank, controllability_rank)

% Get the rank of the reconstruction
reconstruction_rank = rank(...
    my_model_base.AdaptiveDmdc_obj.calc_reconstruction_control());
fprintf('Signal rank: %d\nReconstructed signal rank: %d\n',...
    signal_rank, reconstruction_rank)

% Use MATLAB state space models
my_dss = dss(A, B, eye(size(A)), 0, eye(size(A)), 1);

r = 10;
[~, baldat] = hsvd(my_dss);
reduced_dss = balred(my_dss, r, baldat);
ctr_gramian = gram(reduced_dss, 'c');

A2 = reduced_dss.A;
B2 = reduced_dss.B;

% The desired state will be a step in the direction of an svd mode in the
% data
% x1 = 0.1*my_model_base.L_sparse_modes(:,2);
x1 = my_model_base.dat(:,5);
% x0 = zeros(size(x1));
x0 = my_model_base.dat(:,1);
% Put these desired states in the reduced model basis
x1 = baldat.u(:,1:r)'*x1;
x0 = baldat.u(:,1:r)'*x0;

t0 = 0;
t1 = 5; % time horizon
tspan = linspace(t0, t1, 1000);

% From: https://en.wikipedia.org/wiki/Minimum_energy_control
u = @(t) -B2'*exp(A2'.*(t1-t))/ctr_gramian*(exp(A2.*(t1-t))*x0-x1);

figure
u_t = zeros(size(B2,2), length(tspan));
for i = 1:length(tspan)
    u_t(:,i) = u(tspan(i));
end
imagesc(real(u_t))
colorbar
%==========================================================================


%% Dynamic importance metric
% i.e. how much better does a single neuron reconstruct each other neuron
% when compared to a null model
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);
ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    'dmd_mode','func_DMDc',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'None';

all_neurons = size(dat_struct.traces,2);
all_corr = zeros(all_neurons);
all_corr_null = zeros(all_neurons);
for i = 1:all_neurons
    % Get models
    designated_controller_channels = {'This_neuron', i};
    settings.designated_controller_channels = ...
        designated_controller_channels;
    settings.dmd_mode = 'func_DMDc';
    my_model_1_neuron = CElegansModel(filename, settings);
    settings.dmd_mode = 'no_dynamics';
    my_model_1_neuron_null = CElegansModel(filename, settings);

    % Get correlations and save them
    ind = 1:all_neurons;
    ind(i) = [];
    all_corr(i, ind) = my_model_1_neuron.calc_correlation_matrix();
    all_corr_null(i, ind) = my_model_1_neuron_null.calc_correlation_matrix();
end

% Plot these matrices
figure;
imagesc(real(all_corr))
title('All correlations in dynamic model')
xlabel('Controller neuron')
ylabel('Controlled neurons')

figure;
imagesc(real(all_corr_null))
title('All correlations in null model')
xlabel('Controller neuron')
ylabel('Controlled neurons')

% Calculate an improvement score (strictly positive)
% dynamic_improvement = (1+real(all_corr))./(1+real(all_corr_null)) - 1;
% figure;
% imagesc(dynamic_improvement)
% title('Dynamic improvement (unbounded)')
% colorbar

dynamic_improvement = real(all_corr)-abs(all_corr_null);
% dynamic_improvement = real(all_corr)-real(all_corr_null);
figure;
imagesc(dynamic_improvement)
title('Dynamic improvement (subtraction)')
colorbar
xlabel('Controller neuron')
ylabel('Controlled neurons')
xticks(1:all_neurons)
xticklabels(all_names(sort_ind))
xtickangle(90)
yticks(1:all_neurons)
yticklabels(all_names(sort_ind))

% figure;
% boxplot(dynamic_improvement)
% title('Dynamic improvement (subtraction)')
% xlabel('Controller neuron')
all_names = my_model_bu.get_names();

dyn_mean = mean(dynamic_improvement,1);
dyn_median = median(dynamic_improvement,1);
dyn_max = max(dynamic_improvement(sort_ind,:),[],1);

% [~, sort_ind] = sort(dyn_mean, 'descend');
[~, sort_ind] = sort(dyn_max, 'descend');
figure
plot(dyn_mean(sort_ind))
hold on
plot(dyn_median(sort_ind))
plot(dyn_max(sort_ind))
title('Dynamic improvement (subtraction)')
legend({'Mean', 'Median', 'Max'})
xlabel('Controller neuron')
xticks(1:all_neurons)
xticklabels(all_names(sort_ind))
xtickangle(90)


%---------------------------------------------
% Create a new model with large dynamic importance controllers
%---------------------------------------------
num_neurons = 30;
original_sensory_neurons = ...
    [1 5 15 64 121 122 ... % unnamed 
    ...60 ... % ALA
    ...55 56 ... % ASK
    ...101 ... % RIS
    ];
designated_controller_channels = ...
    {'High_importance_neurons', sort_ind(1:num_neurons),...
    'Sensory_neurons', original_sensory_neurons,...
    'Turning_neurons', 'SMD'};
settings.designated_controller_channels = ...
    designated_controller_channels;
settings.dmd_mode = 'func_DMDc';
settings.filter_window_dat = 1;
settings.filter_window_global = 1;
my_model_important_neurons = CElegansModel(filename, settings);

my_model_important_neurons.plot_colored_reconstruction();
my_model_important_neurons.plot_reconstruction_interactive(false);

%==========================================================================


%% Greedy algorithm for adding neurons via their dynamic importance
% i.e. how much better does a single neuron reconstruct each other neuron
% when compared to a null model
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);
ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    'dmd_mode','func_DMDc',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'None';

my_model_base = CElegansModel(filename, settings);

if exist('previous_controllers', 'var')
    error('Do you really want to delete the previous model?')
end

num_models = 10;
all_corr = cell(num_models,1);
all_corr_null = cell(num_models,1);
dyn_func = @(x,y) real(x) - max(real(y),0);

previous_controllers = {'original_controllers', [1 5 15]};
sz = size(dat_struct.traces, 2);
all_neurons = 1:sz;

% for i_model = 1:num_models
% for i_model = 11:15
for i_model = 13:15
    
    % Get the list of non-controller neurons with original indices
    candidates = 1:sz;
    all_ind = [previous_controllers{2:2:length(previous_controllers)}];
    candidates(all_ind) = [];

    % Build the dynamic importance metric
    this_corr = zeros(sz);
    this_corr_null = zeros(sz);
    for i = 1:length(candidates)
        test_neuron = candidates(i);
        % Get models
        designated_controller_channels = [previous_controllers...
            'Test_neuron', test_neuron];
        settings.designated_controller_channels = ...
            designated_controller_channels;
        settings.dmd_mode = 'func_DMDc';
        my_model_1_neuron = CElegansModel(filename, settings);
        settings.dmd_mode = 'no_dynamics';
        my_model_1_neuron_null = CElegansModel(filename, settings);

        % Get correlations and save them
        ind = candidates;
        ind(i) = [];
        this_corr(test_neuron, ind) = ...
            my_model_1_neuron.calc_correlation_matrix();
        this_corr_null(test_neuron, ind) = ...
            my_model_1_neuron_null.calc_correlation_matrix();
    end
    
%     all_corr{i_model} = this_corr;
%     all_corr_null{i_model} = this_corr_null;
    all_corr = [all_corr; this_corr]; %#ok<AGROW>
    all_corr_null = [all_corr_null; this_corr_null]; %#ok<AGROW>
    
    % Use the dynamic metric to add a single 'best' neuron to the next model
    this_dyn = dyn_func(this_corr, this_corr_null);
    [this_improvement, next_neuron] = max(mean(this_dyn, 2));
    fprintf('Adding neuron %d (%s); mean improvement is %.2f\n',...
        next_neuron, my_model_base.get_names(next_neuron), this_improvement)
    assert(~ismember(next_neuron,...
        [previous_controllers{2:2:length(previous_controllers)}]),...
        'Attempting to re-add the same controller neuron; aborting')
    previous_controllers = [previous_controllers,...
        {sprintf('Best_neuron_round_%d', i_model), next_neuron}]; %#ok<AGROW>
end

% Calculate an improvement score (strictly positive)
all_names = my_model_base.get_names();

dyn_mean = mean(dynamic_improvement,2);
dyn_median = median(dynamic_improvement,2);
dyn_max = max(dynamic_improvement,[],2);

% [~, sort_ind] = sort(dyn_mean, 'descend');
[~, sort_ind] = sort(dyn_median, 'descend');
% [~, sort_ind] = sort(dyn_max, 'descend');

dynamic_improvement = dyn_func(this_corr, this_corr_null);
% dynamic_improvement = real(all_corr)-real(all_corr_null);
figure;
imagesc(dynamic_improvement(sort_ind, sort_ind))
title('Dynamic improvement (subtraction)')
colorbar
xlabel('Controller neuron')
ylabel('Controlled neurons')
xticks(all_neurons)
xticklabels(all_names(sort_ind))
xtickangle(90)
yticks(all_neurons)
yticklabels(all_names(sort_ind))

figure
plot(dyn_mean(sort_ind))
hold on
plot(dyn_median(sort_ind))
plot(dyn_max(sort_ind))
title('Dynamic improvement (subtraction)')
legend({'Mean', 'Median', 'Max'})
xlabel('Controller neuron')
xticks(all_neurons)
xticklabels(all_names(sort_ind))
xtickangle(90)

%---------------------------------------------
% Create a new model with large dynamic importance controllers
%---------------------------------------------
settings.designated_controller_channels = previous_controllers;
settings.dmd_mode = 'func_DMDc';
% settings.filter_window_dat = 1;
% settings.filter_window_global = 1;
my_model_important_neurons = CElegansModel(filename, settings);

my_model_important_neurons.plot_colored_reconstruction();
my_model_important_neurons.plot_reconstruction_interactive(false);

%==========================================================================


%% UPDATED Greedy algorithm for adding neurons via their dynamic importance
% i.e. how much better does a single neuron reconstruct each other neuron
% when compared to a null model
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);
ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    ...'augment_data', 5,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    'dmd_mode','func_DMDc',...
    ...'autocorrelation_noise_threshold', 0.2,...
    'autocorrelation_noise_threshold', 0.7,...
    'lambda_sparse',0);
settings.global_signal_mode = 'None';

my_model_base = CElegansModel(filename, settings);

if exist('previous_controllers', 'var')
    error('Do you really want to delete the previous model?')
end

num_models = 10;
all_corr = cell(num_models,1);
[~, data_corr_mat] = my_model_base.calc_correlation_matrix();
dyn_func = @(x,y) real(x) - max(real(y),0);

previous_controllers = {'original_controllers', [1 5 15]};
sz = my_model_base.dat_sz(1);
all_neurons = 1:sz;
data_corr_mat = data_corr_mat(all_neurons,all_neurons);

for i_model = 1:num_models
% for i_model = 2:num_models
% for i_model = 11:15
    
    % Get the list of non-controller neurons with original indices
    candidates = 1:sz;
    all_ind = [previous_controllers{2:2:length(previous_controllers)}];
    candidates(all_ind) = [];
    
    % Get the base model for this round
    settings.designated_controller_channels = previous_controllers;
    my_model_base_next = CElegansModel(filename, settings);
    previous_correlations = zeros(1, sz);
    previous_correlations(candidates) = ...
        my_model_base_next.calc_correlation_matrix()';
    previous_correlations = repmat(previous_correlations, [sz, 1]);
    
    % Build the dynamic importance metric
    this_corr = zeros(sz);
    data_correlations = zeros(sz);
    for i = 1:length(candidates)
        test_neuron = candidates(i);
        % Get models
        designated_controller_channels = [previous_controllers...
            'Test_neuron', test_neuron];
        settings.designated_controller_channels = ...
            designated_controller_channels;
        my_model_1_neuron = CElegansModel(filename, settings);

        % Get correlations and save them
        ind = candidates;
        ind(i) = [];
        this_corr(test_neuron, ind) = ...
            my_model_1_neuron.calc_correlation_matrix();
        data_correlations(test_neuron, ind) = ...
            data_corr_mat(test_neuron, ind);
    end
    
    all_corr{i_model} = this_corr;
    % Two parts to the null:
    %   The model correlations without the new control neuron
    %   Plus the correlation of the new control neuron with the data
    this_corr_null = data_correlations + previous_correlations;
    
    % Use the dynamic metric to add a single 'best' neuron to the next model
    this_dyn = dyn_func(this_corr, this_corr_null);
    [this_improvement, next_neuron] = max(mean(this_dyn, 2));
    fprintf('Adding neuron %d (%s); mean improvement is %.2f\n',...
        next_neuron, my_model_base.get_names(next_neuron), this_improvement)
    assert(~ismember(next_neuron,...
        [previous_controllers{2:2:length(previous_controllers)}]),...
        'Attempting to re-add the same controller neuron; aborting')
    previous_controllers = [previous_controllers,...
        {sprintf('Best_neuron_round_%d', i_model), next_neuron}]; %#ok<AGROW>
    
end

% Calculate an improvement score (strictly positive)
all_names = my_model_base.get_names();

dynamic_improvement = dyn_func(this_corr, this_corr_null);
dyn_mean = mean(dynamic_improvement,2);
dyn_median = median(dynamic_improvement,2);
dyn_max = max(dynamic_improvement,[],2);

% [~, sort_ind] = sort(dyn_mean, 'descend');
[~, sort_ind] = sort(dyn_median, 'descend');
% [~, sort_ind] = sort(dyn_max, 'descend');

figure;
imagesc(dynamic_improvement(sort_ind, sort_ind))
title('Dynamic improvement (subtraction)')
colorbar
xlabel('Controller neuron')
ylabel('Controlled neurons')
xticks(all_neurons)
xticklabels(all_names(sort_ind))
xtickangle(90)
yticks(all_neurons)
yticklabels(all_names(sort_ind))

figure
plot(dyn_mean(sort_ind))
hold on
plot(dyn_median(sort_ind))
plot(dyn_max(sort_ind))
title('Dynamic improvement (subtraction)')
legend({'Mean', 'Median', 'Max'})
xlabel('Controller neuron')
xticks(all_neurons)
xticklabels(all_names(sort_ind))
xtickangle(90)

%---------------------------------------------
% Create a new model with large dynamic importance controllers
%---------------------------------------------
settings.designated_controller_channels = previous_controllers;
settings.dmd_mode = 'func_DMDc';
% settings.filter_window_dat = 1;
% settings.filter_window_global = 1;
my_model_important_neurons = CElegansModel(filename, settings);

my_model_important_neurons.plot_colored_reconstruction();
my_model_important_neurons.plot_reconstruction_interactive(false);

%==========================================================================


%% Iteratively solve directly for control signals
% Overall solver settings
to_use_L1 = false;
num_iter = 20;
r_ctr = 2;
to_use_model_U = false;
tspan = 1:1000;
to_solve_PCA_basis = false;

% Create model
filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
dat_struct = importdata(filename);

settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    ...'augment_data', 20,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','func_DMDc',...
    'autocorrelation_noise_threshold', 0.6,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model as a preprocessor
my_model_base = CElegansModel(filename, settings);

% Then use the model
X1 = my_model_base.dat(:,tspan(1:end-1));
X2 = my_model_base.dat(:,tspan(2:end));
% X1 = dat_struct.traces(tspan(1:end-1),:)';
% X2 = dat_struct.traces(tspan(2:end),:)';
% X1 = X1 - mean(X1,2);
if to_solve_PCA_basis
    % Truncate the data first for speed purposes
    [r, d, ~, U_X1] = optimal_truncation(X1);
    r = round(r/4);
    U_X1 = U_X1(:,1:r);
    X1 = U_X1'*X1;
    % X2 = U_X1'*(X2-mean(X2,2));
    X2 = U_X1'*X2;
    % Regularize the modes so the 1st isn't dominant
    X1 = X1./sqrt(d(1:r)+1e-1);
    X2 = X2./sqrt(d(1:r)+1e-1);
end

[n, m] = size(X1);

all_err = zeros(num_iter,2);
if to_use_model_U
    U = my_model_base.control_signal(1:r_ctr,1:m);
else
%     U = rand(r_ctr, m);
    % Initialize with the errors from a naive DMD fit
    [~, ~, U] = svd(real(X2 - (X2/X1)*X1));
    U = U(:,1:r_ctr)';
end
if to_use_L1
    % lambda = 1e15; % The U matrix has entries on the order of 1e-16...
%     lambda = 1e-2;
    which_fit = 9;
    all_U = cell(num_iter, 1);
else
    lambda = 1e-2;
    sparsity_pattern = false(size(U));
    all_U = cell(num_iter, 1);
end

tstart = tic;
for i = 1:num_iter
    fprintf('Iteration %d/%d\n', i, num_iter)
    % Step 1:
    %   Get A and B matrix, given U
%     [~, ~, B, ~, Uhat, ~, ~, ~, ~, ~, A] = ...
%         func_DMDc(X1, X2, full(U), r, r_ctr);
    AB = X2/[X1; full(U)];
    A = AB(:,1:n);
    B = AB(:,(n+1):end);
    
    all_err(i, 1) = norm(A*X1 + B*U - X2, 2);
    
    % Step 2:
    %   Use cvx to get A and U, given B
    %   NEVERMIND, just solve for U; otherwise takes too long
    %   Also try to sparsify U
    if to_use_L1
%         cvx_begin
%             variable A(n,n)
%             variable U(r_ctr, m)
%             minimize( norm(A*X1 + B*U - X2, 2) + lambda*norm(U,1) )
%         cvx_end
        res = X2 - A*X1;
        all_fits = zeros(m,5);
        all_intercepts = zeros(m,1);
        this_U = zeros(size(U));
%         for i2 = 1:(m-1)
        for i2 = 1:m
%             [all_fits, fit_info] = lasso(B, res(:,i2+1),...
%                 'NumLambda', 10);
            [all_fits, fit_info] = lasso(B, res(:,i2),...
                'NumLambda', 10);
            this_U(:, i2) = all_fits(:,which_fit); % Which fit = determined by eye
            all_intercepts(i2) = fit_info.Intercept(which_fit);
        end
        all_U{i} = this_U + all_intercepts';
        U = this_U;
        fprintf('Number of nonzero control signals: %d\n', nnz(U))
        
    else
        % Use sequential LS
%         cvx_begin
%             variable A(n,n)
%             variable U(r_ctr, m)
% %             minimize( norm(A*X1 + B*U - X2, 2) + lambda*norm(U,1) )
%             minimize( norm(A*X1 + B*U - X2, 2) )
%             U(sparsity_pattern) == 0 %#ok<NOPTS,EQEFF>
%         cvx_end
        
        U = B\(X2 - A*X1);
        U(sparsity_pattern) = 0;
        fprintf('Number of nonzero control signals: %d\n', nnz(U))
        % Get rid of bottom 10% of nonzeros
        tmp = reshape(abs(U), [m*r_ctr, 1]);
        threshold = quantile(tmp(tmp>0),0.1);
        
%         threshold = median(median(abs(U)));
        sparsity_pattern = abs(U)<threshold;
        all_U{i} = U;
    end
    
    all_err(i, 2) = norm(A*X1 + B*U - X2, 2);
end
toc(tstart)

figure
plot(all_err(:,1))
hold on
plot(all_err(:,2))
title('Errors for alternating minimization')
legend({'Initial (L2) step', 'Second (cvx) step'})
xlabel('Iteration')
ylabel('L2 error')

approx_X1 = zeros(size(X1));
approx_X1 = [X1(:,1) approx_X1];
if ~to_use_L1
% U = all_U{19};
    U = all_U{end};
end
AB = X2/[X1; full(U)];
A = AB(:,1:n);
B = AB(:,(n+1):end);
for i = 1:m
    approx_X1(:,i+1) = A*approx_X1(:,i) + B*U(:,i);
end
plot_2imagesc_colorbar(X1, approx_X1, '2 1');

figure;
i = 2;
plot(X1(i,:));
hold on; 
plot(approx_X1(i,:))

plot_2imagesc_colorbar(U, B, '2 1', 'Control signal', 'B matrix');

%==========================================================================


%% AUGMENT; Use only 3 positive transitions; try to predict these kicks
% i.e. just projecting the data onto the control signals
filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
ad_settings = struct(...
    'oscillation_threshold', 0.01);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'AdaptiveDmdc_settings', ad_settings,...
    'augment_data', 25,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    'dmd_mode','func_DMDc',...
    'global_signal_subset', {{'DT','VT','REV'}},...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model (no sparse signals)
my_model_3_kicks = CElegansModel(filename, settings);

%---------------------------------------------
% LASSO from a direct fit to DATA
%---------------------------------------------
U2 = my_model_3_kicks.control_signal(:,2:end);
X1 = my_model_3_kicks.dat(:,1:end-1);
disp('Fitting lasso models...')
all_intercepts = zeros(size(U2,1),1);
B_prime_lasso = zeros(size(U2,1), size(X1,1));
which_fit = 8;
for i = 1:size(U2,1)
%     B_prime_lasso(i,:) = U2(i,:)/X1;
    [all_fits, fit_info] = lasso(X1', U2(i,:), 'NumLambda',10);
    all_intercepts(i) = fit_info.Intercept(which_fit);
    B_prime_lasso(i,:) = all_fits(:,which_fit); % Which fit = determined by eye
%     B_prime_lasso(i,:) = lasso(X1', zeros(size(X1,2),1));
end

ctr = my_model_3_kicks.control_signal(which_ctr,:);
ctr_reconstruct = zeros(size(ctr));
ctr_reconstruct(1) = ctr(1);
for i = 2:length(ctr)
    ctr_reconstruct(i) = B_prime_lasso(which_ctr,:) * X1(:,i-1);
end
ctr_reconstruct = ctr_reconstruct + all_intercepts(which_ctr);

figure
ctr = my_model_3_kicks.control_signal(which_ctr,:);
plot(ctr)
hold on
plot(ctr_reconstruct, 'Linewidth',2)
title(sprintf('Sparse reconstruction of control signal %d',which_ctr))
legend({'True','Reconstructed'})

%---------------------------------------------
% Linear reconstruction from a direct fit to RECONSTRUCTED data
%---------------------------------------------
U2 = my_model_3_kicks.control_signal(:,2:end);
X1 = my_model_3_kicks.dat(:,1:end-1);
% X1 = my_model_3_kicks.AdaptiveDmdc_obj.reset_threshold
B_prime3 = U2/X1;

which_ctr = 1;
ctr = my_model_3_kicks.control_signal(which_ctr,:);
ctr_reconstruct = zeros(size(ctr));
ctr_reconstruct(1) = ctr(1);
for i = 2:length(ctr)
    ctr_reconstruct(i) = B_prime3(which_ctr,:) * ...
        my_model_3_kicks.dat(:,i-1);
end

figure
ctr = my_model_3_kicks.control_signal(which_ctr,:);
plot(ctr)
hold on
plot(ctr_reconstruct, 'Linewidth',2)
title(sprintf('Reconstruction of control signal %d',which_ctr))
legend({'True','Reconstructed'})


% Also plot the matrix
figure
% imagesc(B_prime2);
imagesc(B_prime_lasso);
colorbar

names = my_model_3_kicks.get_names();
xticks(1:x)
xticklabels(names)
xtickangle(90)
yticks(1:5)
ind = contains(my_model_3_kicks.state_labels_key, {'REV', 'DT', 'VT'});
yticklabels(my_model_3_kicks.state_labels_key(ind))

% Unroll one of the controllers
sz = size(B_prime_lasso);
fig = figure;
i = 1;
dat_unroll1 = reshape(B_prime_lasso(i,:), ...
    [sz(2)/settings.augment_data, settings.augment_data]);
imagesc(dat_unroll1)
colormap(cmap_white_zero(dat_unroll1));
colorbar
title(sprintf('Unrolled predictors for control signal %d', i))
yticks(1:x)
yticklabels(names)
xlabel('Number of delay frames')
[~, im1] = fig.Children.Children;
im1.ButtonDownFcn = @(x,y) ...
    my_model_3_kicks.callback_plotter_reconstruction(x,y);

% Unroll the predictors of a single neuron
fig = figure;
% this_neuron = 'AVAL';
% this_neuron = 'SMDDL';
this_neuron = 'RIML';
i = my_model_3_kicks.name2ind(this_neuron);
i = i(1);
A = my_model_3_kicks.AdaptiveDmdc_obj.A_separate(:,1:sz(2));
aug = settings.augment_data;
dat_unroll2 = real(reshape(A(i,:), [sz(2)/aug, aug]));
imagesc(dat_unroll2)
colormap(cmap_white_zero(dat_unroll2));
colorbar
title(sprintf('Unrolled predictors for neuron %s', this_neuron))
yticks(1:x)
yticklabels(names(1:n))
xlabel('Number of delay frames')
[~, im1] = fig.Children.Children;
im1.ButtonDownFcn = @(x,y) ...
    my_model_3_kicks.callback_plotter_reconstruction(x,y);
                
% Unroll all neurons and look at summary statistics
n = my_model_3_kicks.original_sz(1);
aug = settings.augment_data;
all_unroll = zeros(n, aug, n);
A = my_model_3_kicks.AdaptiveDmdc_obj.A_separate(:,1:sz(2));
for i = 1:my_model_3_kicks.original_sz(1)
    all_unroll(:,:,i) = real(reshape(A(i,:), [sz(2)/aug, aug]));
end
pos_unroll = all_unroll.*(all_unroll>0);
unroll_per_neuron_pos = reshape(mean(pos_unroll,1),[aug,n]);
figure;
waterfall(unroll_per_neuron_pos')
title('Actuation plot for positive weights')
xlabel('Time delay')
ylabel('Neuron being actuated')
plot_std_fill(unroll_per_neuron_pos, 2);
title('Positive actuation averaged over all neurons')

figure;
neg_unroll = all_unroll.*(all_unroll<0);
unroll_per_neuron_neg = reshape(mean(neg_unroll,1),[aug,n]);
waterfall(-unroll_per_neuron_neg')
title('Flipped actuation plot for negative weights')
xlabel('Time delay')
ylabel('Neuron being actuated')

plot_std_fill(unroll_per_neuron_neg, 2);
title('Negative actuation averaged over all neurons')
%==========================================================================


%% Havok-like identification of control signals
filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'augment_data', 25,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    ...'dmd_mode','truncated_DMDc',... % Keeps the predictions of the controller
    'dmd_mode','func_DMDc',...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'None';

% Get a no-controller model
my_model_U = CElegansModel(filename, settings);




%==========================================================================


%% Use the new fast sparse algorithm
filename = '../../Zimmer_data/WildType_adult/simplewt1/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'augment_data', 6,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','sparse_fast',...
    'global_signal_subset', {{'DT','VT','REV'}},...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

tstart = tic;
my_model_sparse = CElegansModel(filename, settings);
toc(tstart);

% my_model_sparse.plot_reconstruction_interactive(false);
my_model_sparse.plot_matrix_A();
my_model_sparse.plot_unrolled_matrix('waterfall');
%==========================================================================


%% Try to cluster presynaptic weights
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'augment_data', 9,...
    ...'filter_window_global', 0,...
    'filter_window_dat', 1,...
    'dmd_mode','sparse_fast',...
    'global_signal_subset', {{'DT','VT','REV'}},...
    ...'autocorrelation_noise_threshold', 0.3,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

my_model_cluster = CElegansModel(filename, settings);

% Get the unrolled matrix into the proper format
my_model_cluster.calc_unrolled_matrix();
dat = my_model_cluster.A_unroll;
n = size(dat,1);
dat = reshape(permute(dat, [1 3 2]), [n*n, size(dat,2)]);

% Get names as from-to pair
names_original = my_model_cluster.get_names([], true);
names_original = names_original(1:n)';

tmp = cell(n*n, 1);
for i = 1:n
    ind = ((i-1)*n+1):(i*n);
    tmp(ind) = cellfun(@(x) [x '_to_' names_original{i}], names_original,...
        'UniformOutput',false);
end
names = tmp;

name_pairs = cell(n*n, 2);
for i = 1:n
    ind = ((i-1)*n+1):(i*n);
    name_pairs(ind,1) = names_original;
    name_pairs(ind,2) = names_original(i);
end

use_only_named = true;
if use_only_named
    ind = cellfun(@(x) isequal(regexp(x, '[0-9]+'),1),name_pairs);
    ind = ~any(ind, 2);
    dat = dat(ind, :);
    names = names(ind);
    name_pairs = name_pairs(ind,:);
end

% threshold = 3e-1;
threshold = 1e-2;
significant_connection_ind = sum(abs(dat),2) > threshold;
dat = dat(significant_connection_ind,:);
names = names(significant_connection_ind);
name_pairs = name_pairs(significant_connection_ind,:);
fprintf('Using %d significant connections\n', ...
    length(find(significant_connection_ind)))

% Get distances and cluster
which_metric = 'corr';
disp('Calculating distances...')
switch which_metric
    case 'dtw'
        % Dynamic Time Warping distance
        all_dist = zeros(n,n);
        for i = 1:n
            for i2 = (i+1):n
                all_dist(i,i2) = dtw(dat(i,:), dat(i2,:));
                all_dist(i2,i) = all_dist(i,i2);
            end
        end
        
        all_dist_dtw = all_dist; % Save an extra because this takes a while

    case 'corr'
        % cross-correlation
        all_dist = squareform(pdist(dat, 'correlation'));

    case 'L2'
        % Euclidean distance
        all_dist = squareform(pdist(dat, 'euclidean'));

    otherwise
        error('Unrecognized metric')
end
% Cluster using linkage and plot
disp('Creating linkages...')
Z = linkage(all_dist);

disp('Clustering...')
max_clust = 400;
c = cluster(Z, 'maxclust', max_clust);

[~, ~, all_ind] = cluster_and_imagesc(all_dist, c, names, []);

%---------------------------------------------
% Visualize graphs for each cluster
%---------------------------------------------

% Look at centroids of clusters AND the graph
%   (only the ones that have more than one member)
all_centroids = [];
all_graphs = {};
for i = 1:length(all_ind)
    if length(all_ind{i})<=5
        continue
    end
    this_dat = dat(all_ind{i},:);
    all_centroids = [all_centroids; mean(this_dat,1)]; %#ok<AGROW>
    nodes1 = name_pairs(all_ind{i},1);
    nodes2 = name_pairs(all_ind{i},2);
    this_graph = digraph(nodes1, nodes2);
    all_graphs = [all_graphs; {this_graph}]; %#ok<AGROW>
    fig = figure('DefaultAxesFontSize',14);
    subplot(2,1,1)
    plot_std_fill(this_dat',2, [], [], fig);
    title(sprintf('Centroid of cluster %d (%d traces)', i, length(all_ind{i})))
    subplot(2,1,2)
    plot(this_graph, 'LineWidth',2)
    drawnow
%     pause
end
%==========================================================================


%% Reprise: Iteratively solve directly for control signals
% Overall solver settings
to_use_L1 = false;
num_iter = 80;
iter_removal_fraction = 0.05;
iter_re_up_fraction = 0;%0.01;
r_ctr = 15;
to_use_model_U = false;
to_threshold_total_U = false;
only_positive_U = true;
tspan = 1:3021;
% tspan = 1:500;
to_solve_PCA_basis = false;

rng(13);

% Create model
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);

% TEST
dat_struct.traces = smoothdata(dat_struct.traces, 'gaussian', 3);
all_max = max(dat_struct.traces, [], 1);
all_max = max(all_max, 1);
dat_struct.traces = dat_struct.traces ./ all_max;
% dat_struct.traces = dat_struct.traces.^2;
warning('USING SQUARED DATA')

settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    'designated_controller_channels', {{'sensory', 1}},...
    ...'augment_data', 5,...
    'filter_window_dat', 1,...
    'dmd_mode','func_DMDc',...
    'autocorrelation_noise_threshold', 0.6,...
    'lambda_sparse',0);
settings.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model as a preprocessor
my_model_base = CElegansModel(dat_struct, settings);

% Then use the model
X1 = my_model_base.dat(:,tspan(1:end-2));
X2 = my_model_base.dat(:,tspan(2:end-1));
X3 = my_model_base.dat(:,tspan(3:end));

[n, m] = size(X1);
S_left = diag(ones(m-1,1), 1);
S_right = diag(ones(m-1,1), -1);

all_err = zeros(num_iter,2);
if to_use_model_U
    U = my_model_base.control_signal(1:r_ctr,1:m);
else
    if ~only_positive_U
        % Initialize with the errors from a naive DMD fit
        [~, ~, U0] = svd(real(X2 - (X2/X1)*X1));
        U = U0(:,1:r_ctr)';
    else
        % Initialize with non-negative factorization
        [~, U0] = nnmf(real(X2 - (X2/X1)*X1), r_ctr);
        U = 2*U0(1:r_ctr,:);
    end
end

sparsity_pattern = false(size(U));
all_U = cell(num_iter, 1);

if only_positive_U
    sparse_func = @(x) x;
else
    sparse_func = @(x) abs(x);
end

tstart = tic;
for i = 1:num_iter
    fprintf('Iteration %d/%d\n', i, num_iter)
    % Step 1:
    %   Get A and B matrix, given U
    AB = X2/[X1; full(U)];
    A = AB(:,1:n);
    B = AB(:,(n+1):end);
    
    all_err(i, 1) = norm(A*X1 + B*U - X2, 2);
    
    % Step 2:
    %   Use cvx to get A and U, given B
    %   NEVERMIND, just solve for U; otherwise takes too long
    %   Also try to sparsify U
    U_raw = B\(X2 - A*X1);
    U = U_raw;
    if to_threshold_total_U
        % Normalize U by the actual effect it has on the data, i.e. include B
        U_effective = zeros(size(U));
        for i2 = 1:r_ctr
            U_effective(i2,:) = sum(abs(B(:,i2))*U(i2,:), 1);
        end
        U_effective_nonsparse = U_effective;
        U_effective_nonsparse(~sparsity_pattern) = 0;
        U_effective(sparsity_pattern) = 0;
        % Get rid of bottom 5% of BU nonzeros
        tmp = reshape(sparse_func(U_effective), [m*r_ctr, 1]);
        threshold = quantile(tmp(tmp>0),iter_removal_fraction);
        sparsity_pattern = sparse_func(U_effective) < threshold;

        % Add entries back in for the top 1% of the ignored matrix, if they
        % are above the median of the entries that are left
        tmp2 = reshape(sparse_func(U_effective_nonsparse), [m*r_ctr, 1]);
        threshold_top = max([quantile(tmp2(tmp2>0),1-iter_re_up_fraction), median(tmp)]);
        re_up_pattern = sparse_func(U_effective_nonsparse) > threshold_top;
    %     U(re_up_pattern) = U_effective_nonsparse(re_up_pattern);
    else
        % Threshold per row of U
        U_nonsparse = U;
        U(sparsity_pattern) = 0;
        U_nonsparse(~sparsity_pattern) = 0;
        re_up_pattern = false(size(sparsity_pattern));
        for i2 = 1:r_ctr
            tmp = U(i2,:);
            threshold = quantile(tmp(tmp>0),iter_removal_fraction);
            sparsity_pattern(i2,:) = sparse_func(tmp) < threshold;
            
            tmp2 = U_nonsparse(i2,:);
            threshold_top = max([quantile(tmp2(tmp2>0),1-iter_re_up_fraction), median(tmp)]);
            re_up_pattern(i2,:) = sparse_func(tmp2) > threshold_top;
        end
        
    end
    
    sparsity_pattern = (sparsity_pattern - re_up_pattern)==1;
    U(sparsity_pattern) = 0;
    % Smooth the controller slightly; increasing factor is heuristic
    U = 1.02*smoothdata(U, 2, 'gaussian', [1 0]);
    all_U{i} = U;
    
    fprintf('Number of nonzero control signals: %d\n', nnz(U))
    fprintf('Number of signals added back in: %d\n', nnz(re_up_pattern))
    
    all_err(i, 2) = norm(A*X1 + B*U - X2, 2);
end
toc(tstart)

% Visualizing "non-trivial" benefit of the control signal and dynamics
all_errs_no_A = zeros(num_iter, 1);
all_errs_Ai = zeros(num_iter, 1);
all_errs_model = zeros(num_iter, 1);
all_nnz = zeros(num_iter, 1);
% all_errs_A = all_errs_Ai;
% all_errs_A_mat = zeros(num_iter, n, m);
err_persistence = norm(X2 - X1(:,1));
deltaX = X2 - X1;
for i = 1:num_iter
%     fprintf('Iteration %d/%d\n', i, num_iter)
    Ui = all_U{i};
    all_errs_no_A(i) = norm((X2/Ui)*Ui-X2);
    all_errs_Ai(i) = norm((deltaX/Ui)*Ui-deltaX);
    all_nnz(i) = nnz(Ui);
end
figure('DefaultAxesFontSize', 16);
subplot(2,1,1)
plot(all_errs_no_A, 'LineWidth', 2);
hold on
plot(all_err, 'LineWidth', 2);
line([0 num_iter], [err_persistence err_persistence],...
    'Color', 'k', 'LineWidth', 2)
legend({'Only BU', 'Full model', 'Persistence model'})
title('Error: no A matrix')
subplot(2,1,2)
dynamic_improvement = all_errs_Ai./all_err(:,1);
plot(dynamic_improvement, 'LineWidth', 2)
hold on
[~, best_U] = max(dynamic_improvement(round(num_iter/5):end));
best_U = best_U + round(num_iter/5) - 1;
plot(best_U, dynamic_improvement(best_U), 'ro', 'LineWidth', 2)
title('Error: improvement over trivial dynamics')

% Plot control signal
U = all_U{best_U};
plot_2imagesc_colorbar(U, B, '2 1', 'Control signal', 'B matrix');

ind = my_model_base.state_labels_ind(1:size(U,2));
key = my_model_base.state_labels_key;
i = 1;
plot_colored(U(i, :), ind, key, 'plot');
title(sprintf('Control signal %d', i))

% Learn a new model with this controller
settings2 = settings;
settings2.augment_data = 0;
this_ctr = [smoothdata(5*U,2,'movmean',2), zeros(r_ctr,2)];
settings2.custom_control_signal = this_ctr(:, 1:end-settings2.augment_data);
settings2.global_signal_mode = 'None';
% settings2.dmd_mode = 'naive';
settings2.dmd_mode = 'func_DMDc';
% settings2.dmd_mode = 'sparse_fast';
% settings2.filter_window_dat = 3;
% settings2.filter_window_global = 10; % Does NOT affect custom signals
my_model_learned = CElegansModel(dat_struct, settings2);

my_model_learned.plot_reconstruction_interactive(false);
%% Some additional error plots

% Look at the cross-validation errors
ad_settings = struct(...
    'hold_out_fraction',0.2,...
    'cross_val_window_size_percent', 0.8);
settings2.AdaptiveDmdc_settings = ad_settings;
my_model_crossval = CElegansModel(dat_struct, settings2);

err_train = my_model_crossval.AdaptiveDmdc_obj.calc_baseline_error();
err_test = my_model_crossval.AdaptiveDmdc_obj.calc_test_error();

all_figs{1} = figure('DefaultAxesFontSize',14);
boxplot(err_test,'colors',[1 0 0])
hold on
boxplot(err_train)
ylim([0, 1.1*max([err_test;err_train])])
ylabel('L2 error')
title('Training and Test Data Reconstruction')

% Cross validation across control signals
err_train = zeros(200, num_iter);
err_test = zeros(200, num_iter);
ad_settings = struct(...
    'hold_out_fraction',0.2,...
    'cross_val_window_size_percent', 0.8);
settings2.AdaptiveDmdc_settings = ad_settings;
for i = 1:num_iter
    fprintf('Iteration %d/%d\n', i, num_iter)
    Ui = all_U{i};
    this_ctr = [smoothdata(5*Ui,2,'movmean',2), zeros(r_ctr,2)];
    settings2.custom_control_signal = this_ctr(:, 1:end-settings2.augment_data);
    my_model_crossval = CElegansModel(dat_struct, settings2);
    
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
%% Visualize the path control signals with increasing rank

max_rank = 30;
all_D = zeros(max_rank,1);
all_W = cell(max_rank, 1);
all_U = cell(max_rank, 1);
res = real(X2 - (X2/X1)*X1);
for i = 1:max_rank
    [all_W{i}, all_U{i}, all_D(i)] = nnmf(res, i);
end
figure;
plot(all_D)
title('All rms residuals')
xlabel('Rank')

all_names = cell(max_rank, 1);
all_names_vec = {};
tol = 1e-1;
x_to_plot = [];
for i = 1:5
    all_names{i} = cell(1, i);
    this_W = all_W{i};
    for i2 = 1:i
%         this_ind = find(isoutlier(this_W(:, i2), 'ThresholdFactor',3));
        x = this_W(:, i2);
        [~, this_ind] = maxk(x, 4);
%         this_ind = find( x > 3*median(x(x>tol)) );
        all_names{i}{:,i2} = my_model_learned.get_names(this_ind, true);
        all_names_vec = [all_names_vec strjoin(all_names{i}{:,i2}, ';')];
        
        % For next plot
        x_to_plot = [x_to_plot, ...
            [i; i2-i/2]]; %#ok<AGROW>
        
%         figure;
%         plot(x)
%         hold on
%         plot(this_ind, x(this_ind), 'o')
%         drawnow
%         pause
    end
end

figure;
plot(x_to_plot(1,:), x_to_plot(2,:), 'o')
hold on
textfit(x_to_plot(1,:), x_to_plot(2,:), all_names_vec)
%==========================================================================


%% Visualization of sparsified learned controllers as a function of rank
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);

% TEST
dat_struct.traces = smoothdata(dat_struct.traces, 'gaussian', 3);
% all_max = max(dat_struct.traces, [], 1);
% all_max = max(all_max, 1);
% dat_struct.traces = dat_struct.traces ./ all_max;
% dat_struct.traces = dat_struct.traces.^2;
warning('USING SQUARED DATA')

settings_base = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'add_constant_signal',false,...
    'use_deriv',false,...
    ...'designated_controller_channels', {{'sensory', 1}},...
    ...'augment_data', 5,...
    'filter_window_dat', 1,...
    'dmd_mode','func_DMDc',...
    'autocorrelation_noise_threshold', 0.6,...
    'lambda_sparse',0);
settings_base.global_signal_mode = 'ID_binary_transitions';

% First get a baseline model as a preprocessor
my_model_base = CElegansModel(dat_struct, settings_base);

which_sparsity = 60; % Manually choose the sparsity for now
num_iter = 101;
max_rank = 20;
all_U1 = cell(max_rank, 1);
all_A1 = cell(max_rank, 1);
all_B1 = cell(max_rank, 1);

% Build the sparse signals
settings = struct('num_iter', num_iter);
for i = 1:max_rank
    settings.r_ctr = i;
    [U, A, B] = sparse_residual_analysis(my_model_base, settings);
    all_U1{i} = U{which_sparsity};
    all_A1{i} = A{which_sparsity};
    all_B1{i} = B{which_sparsity};
end
%% Plot autocorrelation as sparsity increases for a given rank
if ~iscell(U)
%     which_rank = max_rank;
    which_rank = 10;
    % i.e. if I do things out of order
    [U, A, B] = sparse_residual_analysis(my_model_base, ...
        struct('num_iter', num_iter, 'r_ctr', which_rank));
end
r_ctr = size(U{1},1);

all_acf = zeros(num_iter, r_ctr);
all_peaks = zeros(num_iter, r_ctr);
all_nnz = zeros(num_iter, r_ctr);
all_blocks = zeros(num_iter, r_ctr);
% bw = 10;
% reference_dist = @(x, dat) exp(-x./mean(dat))./mean(dat);

for i = 1:num_iter
    for i2 = 1:r_ctr
        dat = U{i}(i2,:)';
        all_acf(i, i2) = acf(dat, 1, false);
%         all_peaks(i, i2) = length(findpeaks(dat, ...
%             'MinPeakProminence', median(dat(dat>0))));
%         all_peaks(i, i2) = length(findpeaks(dat));
%         [all_starts, all_ends] = calc_contiguous_blocks(logical(dat));
%         all_gaps = all_starts(2:end) - all_ends(1:end-1);
% 
%         [b_counts, b_edges] = histcounts(all_gaps, 'BinWidth', bw);
%         x = smoothdata(b_edges, 'movmean',[0 1]);
%         x = x(1:end-1); % Bin mid-points
%         tmp = reference_dist(x, all_gaps);
%         all_peaks(i, i2) = KLDiv(b_counts, tmp);
        all_peaks(i, i2) = -sum(dat(dat>0).*log2(dat(dat>0))); % Shannon

%         this_dat = dat - min(dat(dat>0)); % Shifted exponential
%         this_dat = this_dat(this_dat>0);
%         [all_starts, all_ends] = calc_contiguous_blocks(logical(dat));
%         this_dat = all_ends - all_starts + 1;
%         
%         pd = fitdist(this_dat', 'exponential');
%         all_peaks(i, i2) = pd.negloglik;
%         all_peaks(i, i2) = SampEn(dat, 3, 1e-3);
%         all_peaks(i, i2) = PermEn(dat', 5);
%         all_acf(i, i2) = mean(acf(dat, 2, false));
        all_nnz(i, i2) = nnz(dat);
        all_blocks(i, i2) = length(this_dat);
    end
end
fig = figure;
% waterfall(all_acf')
imagesc(all_acf')
% plot(all_acf)
i = 1;
findchangepts(all_peaks(:,i), 'Statistic', 'linear')

figure;
% this_dat = (log(all_nnz').*all_blocks'+2*all_peaks')./all_nnz'; % BIC version
this_dat = (all_peaks')./all_blocks'; % nll version
imagesc(this_dat);colorbar
title('Log likelihood or BIC version')

i = 1;
figure;
plot(all_acf(:,i))
hold on;
y = all_nnz(:,i)./all_peaks(:,i);
plot(y ./ max(y))
legend({'Autocorrelation', 'nnz/peaks'})
title(sprintf('Conrol signal %d', i))

i = 4;
x = (1:num_iter)';
figure;plot(all_peaks(:,i) - all_peaks(1,i)*0.95.^(x-1)); hold on;plot(x,0*x)
[~, max_ind] = max(all_acf(:,i));
title(sprintf('Difference between actual and modeled peaks; acf max is at %d',...
    max_ind));

i = 10;
figure;plot(U{50}(i,:));hold on;plot(U{80}(i,:));plot(U{90}(i,:))

% i = 19;
U_best = [];
acf_threshold = 0.7;
to_plot = false;
U_best_ind = zeros(r_ctr, 1);
for i = 1:r_ctr
    [max_acf, i2] = max(all_acf(:,i));
    if max_acf > acf_threshold
        U_best_ind(i) = i2;
        U_best = [U_best; U{i2}(i,:)]; %#ok<AGROW>
        [~, this_neuron] = max(abs(B{i2}(:,i)));
        fprintf('Adding signal %d with input onto %s (sparsity: %d)\n', ...
            i, my_model_base.get_names(this_neuron, true), i2)
    end
    if to_plot
        plot_colored(U{i2}(i,:), color_ind, key, 'plot');
        [~, this_neuron] = max(abs(B{i2}(:,i)));
        title(sprintf('Max acf (%.3f) for signal %d (%s) at sparsity %d',...
            max_acf, i, my_model_base.get_names(this_neuron, true), i2))
        drawnow
        pause
    end
end

% Make a model with these signals
settings2 = settings_base;
% settings2.filter_window_dat = 0;
settings2.global_signal_mode = 'None';
% settings2.augment_data = 5;
% settings2.dmd_mode = 'func_dmdc';
settings2.dmd_mode = 'sparse_fast';
settings2.designated_controller_channels = {'sensory', 1};
% this_ctr = [smoothdata(U_best,2,'gaussian',2), zeros(size(U_best,1),1)];
% this_ctr = [zeros(size(U_best,1),1), smoothdata(U_best,2,'movmean',2)];
this_ctr = [zeros(size(U_best,1),1), U_best];
settings2.custom_control_signal = this_ctr ./ max(this_ctr, [], 2);

best_model = CElegansModel(dat_struct, settings2);
best_model.plot_reconstruction_interactive(false);

color_ind = best_model.state_labels_ind(1:end-1);
%% Do a probability distribution comparison
% First, amplitude (vs. exponential)
bw = 1e-2;
reference_dist = @(x, dat) exp(-x./mean(dat));

[b_counts, b_edges] = histcounts(dat(dat>0), 'BinWidth', bw);
figure;histogram('BinEdges', b_edges, 'BinCounts', b_counts,...
    'Normalization','probability')
hold on;
x = smoothdata(b_edges, 'movmean',[0 1]);
x = x(1:end-1); % Bin mid-points
tmp = reference_dist(x, dat(dat>0));
% [b_counts2, b_edges2] = histcounts(tmp, 'BinWidth', bw);
% ind = (b_edges2+bw)>min(b_edges);
histogram('BinEdges', b_edges, 'BinCounts', tmp,...
    'Normalization','probability')

% Second, inter-event interval (also vs. exponential)
bw = 10;

[all_starts, all_ends] = calc_contiguous_blocks(logical(dat));
all_gaps = all_starts(2:end) - all_ends(1:end-1);

[b_counts, b_edges] = histcounts(all_gaps, 'BinWidth', bw);
figure;histogram('BinEdges', b_edges, 'BinCounts', b_counts,...
    'Normalization','probability')
hold on;
x = smoothdata(b_edges, 'movmean',[0 1]);
x = x(1:end-1); % Bin mid-points
tmp = reference_dist(x, all_gaps);
histogram('BinEdges', b_edges, 'BinCounts', tmp,...
    'Normalization','probability')

kl = KLDiv(b_counts, tmp);
title(sprintf('KL-divergence: %.2f', kl))

% Third: negloglik; back to amplitudes
reference_dist = 'exponential';
this_dat = dat - min(dat(dat>0));
this_dat = this_dat(this_dat>0);
pd = fitdist(this_dat, reference_dist);

figure;
histogram(pd.random([100,1]), 'Normalization', 'probability', 'BinWidth',5e-2);hold on
histogram(this_dat, 'Normalization', 'probability', 'BinWidth',5e-2);
%% Register the lines for later plotting across ranks
all_names = cell(max_rank, 1);
registered_lines = {};
registered_names = {};
num_steps_to_plot = 10;
for i = 1:max_rank
    all_names{i} = cell(1, i);
    A = all_A1{i};
    An = A^(num_steps_to_plot-3);
    B = all_B1{i};
    U = all_U1{i};
    for i2 = 1:i
        x = abs(B(:, i2));
%         [~, this_ind] = maxk(x, 4);
%         thresh = max(x)/2;
        [this_maxk, this_ind] = maxk(x, 2);
        if this_maxk(1)/this_maxk(2) > 2
            this_ind = this_ind(1);
        end
%         to_keep_ind = (x(this_ind) > thresh);
%         this_ind = this_ind(to_keep_ind);
        
        all_names{i}{:,i2} = my_model_base.get_names(this_ind, true);
        if ischar(all_names{i}{:,i2})
            all_names{i}{:,i2} = {all_names{i}{:,i2}};
        end
        % Names for registered lines graphing
        other_ind = true(size(x));
        other_ind(this_ind) = false;
        signal_name = sort(all_names{i}{:,i2});
        found_registration = false;
        this_x = i;
        Bi2Ui2 = B(this_ind,i2)*U(i2,:);
        this_y = norm(A(other_ind, :)*An*A(:, this_ind)*Bi2Ui2, 'fro');
        this_z = norm(A*An*A(:, this_ind)*Bi2Ui2, 'fro');
        this_y2 = this_y/norm(Bi2Ui2, 'fro');
%         BU = B(:,i2)*U(i2,:);
%         ABU = A*BU(:, 1:end-1);
%         BU = BU(:, 2:end);
%         tmp1 = ABU.*(ABU>0) - BU.*(BU>0);
%         tmp2 = ABU.*(ABU<0) - BU.*(BU<0);
%         this_y2 = norm(tmp1.*(tmp1>0) + tmp2.*(tmp2>0), 'fro') / norm(BU, 'fro');
%         this_y2 = norm(A(other_ind, :)*B(:,i2)*U(i2,:), 'fro') / ...
%             norm(B(:,i2)*U(i2,:), 'fro');
%         this_y2 = norm(A(other_ind, this_ind)*Bi2Ui2, 'fro') / ...
%             norm(Bi2Ui2, 'fro');
%         this_y2 = norm(A*B(:,i2)*U(i2,:), 'fro') / ...
%             norm(B(:,i2)*U(i2,:), 'fro');
%         [all_starts, all_ends] = calc_contiguous_blocks(...
%             logical(U(i2,:)), 1, 12);
%         [all_starts, all_ends] = calc_contiguous_blocks(...
%             logical(U(i2,:)), 1, 3);
%         this_y2 = mean(all_ends-all_starts);
%         this_y2 = mean(all_starts(2:end) - all_ends(1:end-1));
%         other_ind = true(size(x));
%         other_ind(this_ind) = false;
%         Bi2Ui2 = B(this_ind,i2)*U(i2,:);
%         this_y2 = norm(A(other_ind, :)*An*A(:, this_ind)*Bi2Ui2, 'fro');
%         this_y2 = this_y2/norm(Bi2Ui2, 'fro');
%         figure;histogram(A_nondiag*B(:,i2));
%         title(signal_name)
%         pause
        % Second feature: percent activity on rest of network
%         ABi = abs(A_nondiag*B(:,i2));
%         this_y2 = median(ABi(ABi>median(ABi)));
%         this_y2 = quantile(ABi, 0.99);
        
        this_dat = [this_x; this_y; this_z; i2; this_y2];
        for i4 = 1:length(registered_names)
            % Attach them to a previous line if it exists
            if isequal(registered_names{i4}, signal_name)
                registered_lines{i4} = [registered_lines{i4} ...
                    this_dat]; %#ok<SAGROW>
                found_registration = true;
                break
            end
        end
        if ~found_registration
            if isempty(registered_names)
                registered_names = {signal_name};
            else
                registered_names = [registered_names {signal_name}]; %#ok<AGROW>
            end
            registered_lines = [registered_lines {this_dat} ]; %#ok<AGROW>
        end
    end
end
%% Plot Metrics to distinguish the signals (cleaned up)
% Plot
opt = {'FontSize',14};

% figure('DefaultAxesFontSize', 16);hold on
% n = length(registered_lines);
% which_y = 2;
% text_xy = zeros(n, 2);
% text_names = cell(n, 1);
% for i = 1:n
%     xy = registered_lines{i};
%     if size(xy,2) > 1
%         plot(xy(1,:), xy(which_y,:), 'LineWidth',2)
%     else
%         plot(xy(1,:), xy(which_y,:), 'o')
%     end
%     text_xy(i, :) = [xy(1,1), xy(which_y,1)+0.02];
%     text_names{i} = strjoin(registered_names{i}, ';');
% end
% [clust_ind, centroids] = kmeans(text_xy(:,2), 3);
% [~, noise_ind] = min(centroids);
% text_names(clust_ind==noise_ind) = {''};
% textfit(text_xy(:,1), text_xy(:,2), text_names, opt{:})
% title(sprintf('Control signals weighted by %d step importance (not including self)',...
%     num_steps_to_plot))
% title('Control signals weighted by importance (not including self)')
% ylabel(sprintf('norm(activity) after %d steps',...
%     num_steps_to_plot))
% xlabel('Rank')

figure('DefaultAxesFontSize', 16);hold on
n = length(registered_lines);
text_xy = zeros(n, 2);
text_names = cell(n, 1);
for i = 1:length(registered_lines)
    xy = registered_lines{i};
    if any(xy(1,:)==max_rank) % Used for U_best acf cutoff
        if U_best_ind(xy(4, xy(1,:)==max_rank)) == 0
            continue
        end
    else
        continue
    end
    if size(xy,2) > 1
        plot(xy(1,:), xy(5,:), 'LineWidth',2)
    else
        plot(xy(1,:), xy(5,:), 'o')
    end
    text_xy(i, :) = [xy(1,1), xy(5,1)+0.02];
    text_names{i} = strjoin(registered_names{i}, ';');
end
textfit(text_xy(:,1), text_xy(:,2), text_names, opt{:})
title(sprintf('Relative importance of %d step importance (INCLUDING self)',...
    num_steps_to_plot))
xlabel('Rank')

% figure('DefaultAxesFontSize', 16);hold on
% for i = 1:length(registered_lines)
%     xy = registered_lines{i};
%     if size(xy,2) > 1
%         plot(xy(1,:), xy(3,:), 'LineWidth',2)
%     else
%         plot(xy(1,:), xy(3,:), 'o')
%     end
%     text_xy(i, :) = [xy(1,1), xy(3,1)+0.02];
%     text_names{i} = strjoin(registered_names{i}, ';');
% end
% textfit(text_xy(:,1), text_xy(:,2), text_names, opt{:})
% title(sprintf('Control signals weighted by %d step importance (INCLUDING self)',...
%     num_steps_to_plot))
% xlabel('Rank')

% Make a model using the "good" signals
settings2 = settings_base;
settings2.augment_data = 0;
r_ctr = 15;
U = all_U1{r_ctr};
% which_signals = {...
%     {'SMDVL'; 'SMDVR'}, {'AVAR'; 'AVAL'}, {'AVFL'; 'AVFR'},...
%     {'RIBL'; 'RIBR'}, {'SMDDL'; 'SMDDR'}};
% which_signals = {...
%     {'SMDVL'; 'SMDVR'}, {'AVAR'; 'AVAL'}, {'SMDDL'; 'SMDDR'}};
which_signals = {...
    {'SMDVR'}, {'AVAR'; 'AVAL'}, {'SMDDL'}};
this_r_ctr = length(which_signals);
U_ind = zeros(this_r_ctr,1);
this_ctr = zeros(this_r_ctr,size(U,2));
for i = 1:length(which_signals)
    name_ind = cellfun(@(x) isequal(sort(which_signals{i}),x), registered_names);
    this_line = registered_lines{name_ind};
    U_ind = this_line(4, this_line(1,:)==r_ctr); % The original U space
    this_ctr(i,:) = U(U_ind,:);
end
% this_ctr = [smoothdata(5*this_ctr,2,'movmean',2), zeros(this_r_ctr,2)];
this_ctr = [smoothdata(5*this_ctr,2,'gaussian',2), zeros(this_r_ctr,2)];
settings2.custom_control_signal = this_ctr(:, 1:end-settings2.augment_data);
settings2.global_signal_mode = 'None';
% settings2.augment_data = 5;
% settings2.dmd_mode = 'naive';
settings2.dmd_mode = 'func_DMDc';
% settings2.dmd_mode = 'sparse_fast';
% settings2.filter_window_dat = 3;
my_model_subset = CElegansModel(dat_struct, settings2);
% my_model_subset.plot_colored_reconstruction();
% my_model_subset.plot_reconstruction_interactive(false);
%% Plotting (using reconsruction goodness, not just magnitude of signals)
all_names = cell(max_rank, 1);
registered_lines = {};
registered_names = {};
num_steps_to_plot = 50;
num_reconstructions = 10;

% Model settings
settings2 = settings_base;
settings2.augment_data = 0;
settings2.global_signal_mode = 'None';
settings2.dmd_mode = 'func_DMDc';
settings_naive = settings2;

% First get a baseline model as a preprocessor
my_model_naive = CElegansModel(dat_struct, settings_naive);
recon_func_no_ctr = @(x,t)...
    my_model_naive.AdaptiveDmdc_obj.calc_reconstruction_original(x,t);
num_neurons = my_model_naive.original_sz(1);
% corr_f = @(x,y) diag(1 - squareform(pdist([x; y], 'correlation')), num_neurons);
corr_f = @(x,y,i) diag(squareform(pdist([x(i,:); y(i,:)], 'correlation')), length(i));

for i = 1:15
    all_names{i} = cell(1, i);
    U = all_U1{i};
    u = size(U,1);
    % Build a model for this U
    this_ctr = [smoothdata(2*U,2,'gaussian',2), zeros(i,2)];
    settings2.custom_control_signal = this_ctr(:, 1:end);
    this_model = CElegansModel(dat_struct, settings2);
    recon_func = @(x,t,w)...
        this_model.AdaptiveDmdc_obj.calc_reconstruction_control(x,t,w);
    clear this_recon
    B = this_model.AdaptiveDmdc_obj.A_original(:, end-u+1:end);
    for i2 = 1:i
        % Names for registered lines graphing
        x = abs(B(:, i2));
        [~, this_ind] = maxk(x, 2);
        all_names{i}{:,i2} = this_model.get_names(this_ind, true);
        if ischar(all_names{i}{:,i2})
            all_names{i}{:,i2} = {all_names{i}{:,i2}};
        end
        signal_name = sort(all_names{i}{:,i2});
        found_registration = false;
        % Get the reconstruction (plot-able) variables
        this_x = i;
        all_starts = calc_contiguous_blocks(...
            logical(this_ctr(i2,:)), 4, 1);
        n = this_model.dat_sz(2);
        all_starts = all_starts(all_starts < (n-num_steps_to_plot));
        
        this_order = randperm(length(all_starts));
        all_y = zeros(num_reconstructions,1);
        all_z = zeros(num_reconstructions,1);
        all_y2 = zeros(num_reconstructions,1);
        for i3 = 1:min([num_reconstructions, length(all_starts)])
            this_start = all_starts(this_order(i3));
            tspan = this_start:(this_start+num_steps_to_plot-1);
            x = this_model.dat(:, tspan);
            this_recon = recon_func(x(:,1), tspan, i2);
            this_recon_no_ctr = recon_func_no_ctr(x(:,1), tspan);
            % features: L2 error and correlation
%             all_y(i3) = norm(x-this_recon,'fro') / norm(x,'fro');
            all_y(i3) = mean(corr_f(x, this_recon_no_ctr, this_ind)) / ...
                mean(corr_f(x, this_recon, this_ind));
            all_z(i3) = norm(x-this_recon_no_ctr,'fro') / norm(x,'fro');
            all_y2(i3) = norm(x(this_ind,:)-this_recon(this_ind,:),'fro')...
                .../ norm(x(this_ind,:),'fro');
                / norm(x(this_ind,:)-this_recon_no_ctr(this_ind,:),'fro');
%             j = this_ind(1);
%             figure;plot(x(j,:));hold on;
%             plot(this_recon(j,:)); plot(this_recon_no_ctr(j,:))
%             legend({'Data', 'Reconstruction with control', 'Naive reconstruction'})
        end
        
        this_y = mean(all_y);
        this_z = mean(all_z);
        this_y2 = mean(all_y2);
        
%         dat_to_add = [this_x; this_y; this_z; i2];
        dat_to_add = [this_x; this_y; this_z; i2; this_y2];
        % Attach them to a previous line if it exists
        for i4 = 1:length(registered_names)
            if isequal(registered_names{i4}, signal_name)
                registered_lines{i4} = [registered_lines{i4} ...
                    dat_to_add]; %#ok<SAGROW>
                found_registration = true;
                break
            end
        end
        if ~found_registration
            if isempty(registered_names)
                registered_names = {signal_name};
            else
                registered_names = [registered_names {signal_name}]; %#ok<AGROW>
            end
            registered_lines = [registered_lines ...
                {dat_to_add} ]; %#ok<AGROW>
        end
    end
end

% Plot
opt = {'FontSize',14};

figure('DefaultAxesFontSize', 16);hold on
n = length(registered_lines);
which_y = 2;
text_xy = zeros(n, 2);
text_names = cell(n, 1);
for i = 1:n
    xy = registered_lines{i};
    if size(xy,2) > 1
        plot(xy(1,:), xy(which_y,:), 'LineWidth',2)
    else
        plot(xy(1,:), xy(which_y,:), 'o')
    end
    text_xy(i, :) = [xy(1,1), xy(which_y,1)+0.001];
    text_names{i} = strjoin(registered_names{i}, ';');
end
[clust_ind, centroids] = kmeans(text_xy(:,2), 3);
[~, noise_ind] = min(centroids);
text_names(clust_ind==noise_ind) = {''};
textfit(text_xy(:,1), text_xy(:,2), text_names, opt{:})
title(sprintf('Control signals weighted by %d step importance (not including self)',...
    num_steps_to_plot))
% title('Control signals weighted by importance (not including self)')
% ylabel(sprintf('norm(activity) after %d steps',...
%     num_steps_to_plot))
title('Magnitude of reconstruction L2 error')
xlabel('Rank')

figure('DefaultAxesFontSize', 16);hold on
n = length(registered_lines);
text_xy = zeros(n, 2);
text_names = cell(n, 1);
for i = 1:length(registered_lines)
    xy = registered_lines{i};
    if size(xy,2) > 1
        plot(xy(1,:), xy(5,:), 'LineWidth',2)
    else
        plot(xy(1,:), xy(5,:), 'o')
    end
    text_xy(i, :) = [xy(1,1), xy(5,1)+0.02];
    text_names{i} = strjoin(registered_names{i}, ';');
end
textfit(text_xy(:,1), text_xy(:,2), text_names, opt{:})
% title(sprintf('Relative importance of %d step importance (INCLUDING self)',...
%     num_steps_to_plot))
title('Ratio of reconstruction error to naive error')
% xlabel('Rank')

figure('DefaultAxesFontSize', 16);hold on
for i = 1:length(registered_lines)
    xy = registered_lines{i};
    if size(xy,2) > 1
        plot(xy(1,:), xy(3,:), 'LineWidth',2)
    else
        plot(xy(1,:), xy(3,:), 'o')
    end
    text_xy(i, :) = [xy(1,1), xy(3,1)+0.02];
    text_names{i} = strjoin(registered_names{i}, ';');
end
textfit(text_xy(:,1), text_xy(:,2), text_names, opt{:})
title('STD of reconstruction L2 error')
xlabel('Rank')
%% Make a model using the "good" signals
settings2 = settings_base;
settings2.augment_data = 0;
r_ctr = 15;
U = all_U1{r_ctr};
% which_signals = {...
%     {'SMDVL'; 'SMDVR'}, {'AVAR'; 'AVAL'}, {'AVFL'; 'AVFR'},...
%     {'RIBL'; 'RIBR'}, {'SMDDL'; 'SMDDR'}};
which_signals = {...
    {'SMDVL'; 'SMDVR'}, {'AVAR'; 'AVAL'}, {'SMDDL'; 'SMDDR'}};
this_r_ctr = length(which_signals);
U_ind = zeros(this_r_ctr,1);
this_ctr = zeros(this_r_ctr,size(U,2));
for i = 1:length(which_signals)
    name_ind = cellfun(@(x) isequal(sort(which_signals{i}),x), registered_names);
    this_line = registered_lines{name_ind};
    U_ind = this_line(4, this_line(1,:)==r_ctr); % The original U space
    this_ctr(i,:) = U(U_ind,:);
end
% this_ctr = [smoothdata(5*this_ctr,2,'movmean',2), zeros(this_r_ctr,2)];
this_ctr = [smoothdata(5*this_ctr,2,'gaussian',2), zeros(this_r_ctr,2)];
settings2.custom_control_signal = this_ctr(:, 1:end-settings2.augment_data);
settings2.global_signal_mode = 'None';
% settings2.augment_data = 5;
% settings2.dmd_mode = 'naive';
settings2.dmd_mode = 'func_DMDc';
% settings2.dmd_mode = 'sparse_fast';
% settings2.filter_window_dat = 3;
my_model_subset = CElegansModel(dat_struct, settings2);
my_model_subset.plot_colored_reconstruction();
% my_model_subset.plot_reconstruction_interactive(false);
%% Cross-validation plots for adding signals

% both model settings
settings2 = settings_base;
ad_settings = struct(...
    'hold_out_fraction',0.5,...
    'cross_val_window_size_percent', 0.6);
settings2.AdaptiveDmdc_settings = ad_settings;
settings2.global_signal_mode = 'None';
settings2.dmd_mode = 'func_DMDc';
r_ctr = 14;
U = all_U1{r_ctr};

% Set up list of control signals
rev_ctr = {'AVAR'; 'AVAL'};
% dt_ctr = {'SMDDL'; 'SMDDR'};
dt_ctr = {'SMDDL'}; % Not sure why both names aren't picked up...
% vt_ctr = {'SMDVL'; 'SMDVR'};
vt_ctr = {'SMDVR'};
fwd_ctr = {'RIBL'; 'RIBR'};
% all_signals = {...
%     {},...
%     { rev_ctr },...
%     { rev_ctr, dt_ctr },...
%     { vt_ctr, rev_ctr, dt_ctr },...
%     { vt_ctr, rev_ctr, dt_ctr, fwd_ctr } };
all_signals = {...
    {},...
    { rev_ctr },...
    { vt_ctr, rev_ctr, dt_ctr }, ... 
    { vt_ctr, rev_ctr, dt_ctr, fwd_ctr } };

% Box plots
num_iter = length(all_signals);
all_models = cell(num_iter, 1);
to_return_matrix = true;
test_length = 400;
if ~to_return_matrix
    err_train = zeros(200, num_iter);
    err_test = zeros(200, num_iter);
else
    err_train = zeros(num_neurons, num_iter);
    err_test = zeros(num_neurons, num_iter);
    err_total = zeros(num_neurons, num_iter);
end
ad_settings = struct(...
    'hold_out_fraction',0.2,...
    'cross_val_window_size_percent', 0.8);
settings2.AdaptiveDmdc_settings = ad_settings;
leg_str = cell(num_iter, 1);
leg_str(:) = {''};
error_func = @(x,y) corr(x', y');
for i = 1:num_iter
    fprintf('Iteration %d/%d\n', i, num_iter)
    % Get the signals
    which_signals = all_signals{i};
    this_r_ctr = length(which_signals);
    this_ctr = zeros(this_r_ctr,size(U,2));
    for i2 = 1:length(which_signals)
        name_ind = cellfun(@(x) isequal(sort(which_signals{i2}),x), registered_names);
        this_line = registered_lines{name_ind};
        U_ind = this_line(4, this_line(1,:)==r_ctr); % The original U space
        this_ctr(i2,:) = U(U_ind,:);
        
        this_name = strjoin(registered_names{name_ind}, ';');
        if isempty(leg_str{i})
            leg_str{i} = this_name;
        else
            leg_str{i} = [leg_str{i} '+' this_name];
        end
    end
    this_ctr = [smoothdata(5*this_ctr,2,'gaussian',2), zeros(this_r_ctr,2)];
    settings2.custom_control_signal = this_ctr(:, 1:end);
    all_models{i} = CElegansModel(dat_struct, settings2);
    
    % Use crossvalidation functions
    fprintf('Calculating error terms...\n')
    tmp_train = all_models{i}.AdaptiveDmdc_obj.calc_baseline_error(...
        [], test_length, to_return_matrix, error_func);
    tmp_test = all_models{i}.AdaptiveDmdc_obj.calc_test_error(...
        [], to_return_matrix, error_func);
    if to_return_matrix
%         err_train(:,i) = mean(tmp_train, 1);
%         err_test(:,i) = mean(tmp_test, 1);
        err_train(:,i) = mean(tmp_train, 2);
        err_test(:,i) = mean(tmp_test, 2);
        err_total(:,i) = all_models{i}.calc_correlation_matrix();
    else
        err_train(:,i) = tmp_train;
        err_test(:,i) = tmp_test;
    end
end
all_figs{1} = figure('DefaultAxesFontSize',14);
boxplot(err_test,'colors',[1 0 0])
hold on
boxplot(err_train)
ylim([0, 1.1*max(max([err_test;err_train]))])
ylabel('L2 error')
xlabel('Number of control signals')
xticklabels(leg_str)
xtickangle(60)
title('Training and Test Data Reconstruction')

figure;
[improvement, sort_ind] = sort(err_total(:,4)-err_total(:,3));
these_names = all_models{2}.get_names(sort_ind, true);
plot(improvement, 'o')
xticks(1:length(sort_ind))
xticklabels(these_names)
xtickangle(60)
%% Correlation between learned signals and "real" ones
dat = [my_model_base.control_signal;...
    my_model_subset.control_signal];

all_dist = squareform(pdist(dat, 'jaccard'));

figure;
imagesc(all_dist)
colorbar
%% Plotting (scratch)

all_names = cell(max_rank, 1);
all_names_vec = {};
registered_lines = {};
registered_names = {};
tol = 1e-1;
x_to_plot = [];
num_steps_to_plot = 15;
for i = 1:20
    all_names{i} = cell(1, i);
    A = all_A1{i};
    An = A^(num_steps_to_plot-3);
    B = all_B1{i};
    U = all_U1{i};
    for i2 = 1:i
%         this_ind = find(isoutlier(this_W(:, i2), 'ThresholdFactor',3));
        x = abs(B(:, i2));
%         [~, this_ind] = maxk(x, 4);
%         thresh = max(x)/2;
        [~, this_ind] = maxk(x, 2);
%         to_keep_ind = (x(this_ind) > thresh);
%         this_ind = this_ind(to_keep_ind);
        
        all_names{i}{:,i2} = my_model_base.get_names(this_ind, true);
        if ischar(all_names{i}{:,i2})
            all_names{i}{:,i2} = {all_names{i}{:,i2}};
        end
        % Names for point graphing
%         all_names_vec = [all_names_vec strjoin(all_names{i}{:,i2}, ';')]; %#ok<AGROW>
        % Names for registered lines graphing
        other_ind = true(size(x));
        other_ind(this_ind) = false;
        signal_name = sort(all_names{i}{:,i2});
        found_registration = false;
        this_x = i;
        this_y = norm(A(other_ind, :)*An*A(:, this_ind)*B(this_ind,i2)*U(i2,:));
        for i4 = 1:length(registered_names)
            % Attach them to a previous line if it exists
%             this_overlap = intersect(registered_names{i4}, signal_name);
%             if ~isempty(this_overlap)
            if isequal(registered_names{i4}, signal_name)
                registered_lines{i4} = [registered_lines{i4} ...
                    [this_x; this_y]]; %#ok<SAGROW>
                found_registration = true;
                break
            end
        end
        if ~found_registration
            if isempty(registered_names)
                registered_names = {signal_name};
            else
                registered_names = [registered_names {signal_name}]; %#ok<AGROW>
            end
            registered_lines = [registered_lines ...
                {[this_x; this_y]} ]; %#ok<AGROW>
        end
        
        % Next plot: weight by "downstream effect"
%         x_to_plot = [x_to_plot, ...
%             [i; i2-i/2]]; %#ok<AGROW>
%         ABU1 = A(:, this_ind)*B(this_ind,i2)*U(i2,:);
%         other_ind = true(size(x));
%         other_ind(this_ind) = false;
%         ABU2 = A(other_ind, this_ind)*B(this_ind,i2)*U(i2,:);
%         ABU3 = A(other_ind, :)*(A^7)*A(:, this_ind)*B(this_ind,i2)*U(i2,:);
%         x_to_plot = [x_to_plot, ...
%             [i; norm(ABU1); norm(ABU2); norm(ABU3)]]; %#ok<AGROW>
    end
end

% figure;
% plot(x_to_plot(1,:), x_to_plot(2,:), '.')
% hold on
% textfit(x_to_plot(1,:), x_to_plot(2,:), all_names_vec)
% title('Signals weighted by total effect')
% 
% figure;
% plot(x_to_plot(1,:), x_to_plot(3,:), '.')
% hold on
% textfit(x_to_plot(1,:), x_to_plot(3,:), all_names_vec)
% title('Signals weighted by other-neuron effect')
% 
% figure;
% plot(x_to_plot(1,:), x_to_plot(4,:), '.')
% hold on
% textfit(x_to_plot(1,:), x_to_plot(4,:), all_names_vec)
% title('Signals weighted by other-neuron effect 5 steps away')

figure;hold on
n = length(registered_lines);
text_xy = zeros(n, 2);
text_names = cell(n, 1);
for i = 1:length(registered_lines)
    xy = registered_lines{i};
    plot(xy(1,:), xy(2,:), 'LineWidth',2)
    text_xy(i, :) = [xy(1,1), xy(2,1)];
    text_names{i} = strjoin(registered_names{i}, ';');
end
textfit(text_xy(:,1), text_xy(:,2), text_names)
    
% 
% ind = my_model_base.state_labels_ind;
% key = my_model_base.state_labels_key;
% i = 3;
% i2 = 2;
% plot_colored(all_U1{i}(i2,:), ind(1:end-1), key, 'plot');
% title(sprintf('Control signal %d for rank-%d model', i2, i))

%==========================================================================

%% Scratch for above

% Build data to plot
% X1 = my_model_base.dat(:,1:end-1);
% X2 = my_model_base.dat(:,2:end);
% 
% all_names = cell(max_rank, 1);
% registered_lines = {};
% registered_names = {};
% num_steps_to_plot = 3;
% for i = 1:max_rank
%     all_names{i} = cell(1, i);
%     U = all_U1{i};
%     AB = X2 / [X1; U];
%     A = AB(:, 1:n);
%     An = A^(num_steps_to_plot-2);
%     A_nondiag = A - diag(diag(A));
%     B = AB(:, (n+1):end);
%     for i2 = 1:i
%         x = abs(B(:, i2));
%         [this_maxk, this_ind] = maxk(x, 2);
%         if this_maxk(1)/this_maxk(2) > 2
%             this_ind = this_ind(1);
%         end
%         
%         % Names for registered lines graphing
%         all_names{i}{:,i2} = my_model_base.get_names(this_ind, true);
%         if ischar(all_names{i}{:,i2})
%             all_names{i}{:,i2} = {all_names{i}{:,i2}};
%         end
%         
%         % First feature: max acf
%         signal_name = sort(all_names{i}{:,i2});
%         found_registration = false;
%         [this_y, which_sparsity_in_rank] = max(all_acf{i}(i2, :));
%         
% %         this_dat = [i; this_y; i2];
%         which_rank = i;
%         which_line_in_rank = i2;
%         this_dat = table(which_rank, this_y, which_line_in_rank,...
%             which_sparsity_in_rank);
%         for i4 = 1:length(registered_names)
%             % Attach them to a previous line if it exists
%             if isequal(registered_names{i4}, signal_name)
%                 registered_lines{i4} = [registered_lines{i4}; ...
%                     this_dat]; %#ok<SAGROW>
%                 found_registration = true;
%                 break
%             end
%         end
%         if ~found_registration
%             if isempty(registered_names)
%                 registered_names = {signal_name};
%             else
%                 registered_names = [registered_names {signal_name}]; %#ok<AGROW>
%             end
%             registered_lines = [registered_lines ...
%                 {this_dat} ]; %#ok<AGROW>
%         end
%     end
% end



% Raw Error plots
% figure
% plot(all_err(:,1))
% hold on
% plot(all_err(:,2))
% title('Errors for alternating minimization')
% legend({'Initial (L2) step', 'Second (cvx) step'})
% xlabel('Iteration')
% ylabel('L2 error')

% TEST
% dat_struct.traces = smoothdata(dat_struct.traces, 'sgolay').^2;
dat_struct.traces = smoothdata(dat_struct.traces, 'gaussian', 3);
% dat_struct.traces = dat_struct.traces.^2;
% dat_struct.traces = dat_struct.traces ./ sum(dat_struct.traces.^2,1);
% dat_struct.traces = dat_struct.traces ./ std(dat_struct.traces,1);
if to_solve_PCA_basis
    % Truncate the data first for speed purposes
    [r, d, ~, U_X1] = optimal_truncation(X1);
    r = round(r/4);
    U_X1 = U_X1(:,1:r);
    X1 = U_X1'*X1;
    % X2 = U_X1'*(X2-mean(X2,2));
    X2 = U_X1'*X2;
    % Regularize the modes so the 1st isn't dominant
    X1 = X1./sqrt(d(1:r)+1e-1);
    X2 = X2./sqrt(d(1:r)+1e-1);
end


if to_use_L1
    % lambda = 1e15; % The U matrix has entries on the order of 1e-16...
%     lambda = 1e-2;
    which_fit = 9;
    all_U = cell(num_iter, 1);
else
end
    % Note: func_dmdc doesn't work well because it tries to do an SVD on U,
    % which is sparse and throws everything off
    %     [~, ~, B, ~, Uhat, ~, ~, ~, ~, ~, A] = ...
    %         func_DMDc(X1, X2, full(U), r, r_ctr);
    %     Uhat = Uhat(:,1:r);
    %     A = Uhat*A*Uhat';
    if to_use_L1
%         cvx_begin
%             variable A(n,n)
%             variable U(r_ctr, m)
%             minimize( norm(A*X1 + B*U - X2, 2) + lambda*norm(U,1) )
%         cvx_end
        res = X2 - A*X1;
        all_fits = zeros(m,5);
        all_intercepts = zeros(m,1);
        this_U = zeros(size(U));
%         for i2 = 1:(m-1)
        for i2 = 1:m
%             [all_fits, fit_info] = lasso(B, res(:,i2+1),...
%                 'NumLambda', 10);
            [all_fits, fit_info] = lasso(B, res(:,i2),...
                'NumLambda', 10);
            this_U(:, i2) = all_fits(:,which_fit); % Which fit = determined by eye
            all_intercepts(i2) = fit_info.Intercept(which_fit);
        end
        all_U{i} = this_U + all_intercepts';
        U = this_U;
        fprintf('Number of nonzero control signals: %d\n', nnz(U))
        
    else      
    end
        % Use sequential LS
%         cvx_begin
%             variable A(n,n)
%             variable U(r_ctr, m)
% %             minimize( norm(A*X1 + B*U - X2, 2) + lambda*norm(U,1) )
%             minimize( norm(A*X1 + B*U - X2, 2) )
%             U(sparsity_pattern) == 0 %#ok<NOPTS,EQEFF>
%         cvx_end


%         else
    %   Try 2-step penalization
%         AB = A*B;
%     %     ABB = [AB, B];
%         A2 = A^2;
%     %     U_raw = ABB\(X3 - A2*X1);
%     %     U = U_raw(1:r_ctr,:);
%         U_raw = AB\(X3 - A2*X1 - B*U*S_left);
%         U_raw = U_raw*S_right;
%         U = (U_raw + U) ./ 2;
%     end
%     cvx_begin
%         variable U1(r_ctr, m)
%         variable U2(r_ctr, m)
%         minimize( norm(A2*X1 + AB*U1 + B*U2 - X3, 2))
%         U1(:,1:end-1) == U2(:,2:end) %#ok<NOPTS,EQEFF>
%     cvx_end
%     U_raw = U1;


    
approx_X1 = zeros(size(X1));
approx_X1 = [X1(:,1) approx_X1];
if ~to_use_L1
    U = all_U{end-1};
end
AB = X2/[X1; full(U)];
A = AB(:,1:n);
B = AB(:,(n+1):end);
for i = 1:m
    approx_X1(:,i+1) = A*approx_X1(:,i) + B*U(:,i);
end
plot_2imagesc_colorbar(X1, approx_X1, '2 1', 'Data', 'Reconstruction');

figure;
this_neuron = 'AVAL';
i = my_model_base.name2ind(this_neuron);
plot(X1(i,:));
hold on; 
plot(approx_X1(i,:))
title(this_neuron)

%     X1U = [X1; Ui];
% %     all_errs_A(i, :, :) = (X2/X1U)*X1U-X2;
%     all_errs_A_mat(i, :, :) = (X3/X1U)*X1U-X3;
%     all_errs_A(i) = norm(reshape(all_errs_A_mat(i, :, :), n, m));
%     this_model = CElegansModel(dat_struct, settings2);
%     all_errs_model(i) = ...
%         this_model.AdaptiveDmdc_obj.calc_reconstruction_error();
%==========================================================================


%% Plot elimination path for all datasets

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
    
    settings_base = struct(...
        'to_subtract_mean',false,...
        'to_subtract_mean_sparse',false,...
        'to_subtract_mean_global',false,...
        'add_constant_signal',false,...
        'use_deriv',false,...
        'augment_data', 7,...
        'filter_window_dat', 1,...
        'dmd_mode','no_dynamics',...
        'to_add_stimulus_signal', false,...
        'autocorrelation_noise_threshold', 0.6,...
        ...'global_signal_subset', {{'DT'  'VT'  'REV'  'FWD'  'SLOW'}},...
        'lambda_sparse',0);
    settings_base.global_signal_mode = 'ID_binary_transitions';
    if i > num_type_1 % Differently named states
        settings_base.global_signal_subset = ...
            {'Reversal', 'Dorsal turn', 'Ventral turn'};
    end
    my_model_time_delay = CElegansModel(dat_struct, settings_base);
    
    s = struct('to_calculate_false_detection', false, ...
        'to_calculate_error', false, 'max_iter', 20);
    [B_prime_lasso_td_3d, all_intercepts_td, elimination_neurons, ...
        eliminated_names, which_ctr] = ...
        sparse_encoding_analysis(my_model_time_delay, s);
    
    sz = size(B_prime_lasso_td_3d);
    B_prime_lasso_td_3d = {reshape(...
        B_prime_lasso_td_3d(which_ctr,:,:), sz(2:3))};
    all_intercepts_td = all_intercepts_td(which_ctr,:);
    elimination_neurons = elimination_neurons(which_ctr,:);
    eliminated_names = eliminated_names(which_ctr,:);
    tmp_table = table(eliminated_names, elimination_neurons,...
            all_intercepts_td, B_prime_lasso_td_3d, which_ctr);
    if i == 1
        elimination_table = tmp_table;
    else
        elimination_table = [elimination_table; tmp_table]; %#ok<AGROW>
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
x = 10;
all_counts_early_enough = sum(registered_dat{:,1}(:,1:10), 2);
registered_dat = [registered_dat ...
    table(all_counts) table(all_counts_early_enough)];

%---------------------------------------------
%% Actually plot histogram data for each neuron
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
xlabel('Encoding rank')
ylabel('Occurences across individuals')
%% Save
if to_save    
    fname = sprintf('%sfigure_5e2_%d_test', foldername, 4);
    this_fig = all_figs{1};
    sz = {'0.9\columnwidth', '0.12\paperheight'};
    matlab2tikz('figurehandle',this_fig,'filename',...
        [fname '_raw.tex'], ...
        'width', sz{1}, 'height', sz{2});
    saveas(this_fig, fname, 'png');
end
%==========================================================================


%% Plot improvement of neurons for controller subsets across individuals
%% First build models
all_filenames = get_Zimmer_filenames();
settings_ideal = struct(...
    'add_constant_signal',false,...
    'augment_data', 0,...
    'filter_window_dat', 1,...
    ...'autocorrelation_noise_threshold', 0.3,...
    'dmd_mode','func_DMDc',...
    'global_signal_mode', 'ID_binary_transitions',...
    'to_add_stimulus_signal', false);

num_type1 = 5;
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
        settings_vec{i} = settings_ideal;
        if i <= num_type1
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
        all_models_table = [all_models_table cell2table(obj_vec)]; %#ok<AGROW>
        all_models_table.Properties.VariableNames(end) = ...
            all_settings_loop_names(iInput);
    end
end
%% Calculate the correlations for each model and the clusters
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
%% Now calculate the data differences by cluster (scratch)
% all_differences = {...
%     'Linear_correlation',...
%     'No_control',...
%     'Rev_and_turns',...
%     {'No_control', 'Linear_correlation'},...
%     {'Reversal', 'No_control'},...
%     {'Forward', 'No_control'},...
%     {'Rev_and_turns', 'No_control'},...
%     {'All', 'Rev_and_turns'} };
% all_differences_names = {...
%     'Linear_correlation', 'Only_NC', 'Only_rev_and_turns',...
%     'NC_vs_linear', 'Reversal_vs_NC', 'Forward_vs_NC', ...
%     'Rev_and_turns_vs_NC', 'Fwd_vs_Rev_and_turns'};
% all_differences = {...
%     'Linear_correlation',...
%     'No_control',...
%     'Rev_and_turns',...
%     'All',...
%     {'No_control', 'Linear_correlation'},...
%     {'Reversal', 'Linear_correlation'},...
%     {'Forward', 'Linear_correlation'},...
%     {'Rev_and_turns', 'Linear_correlation'},...
%     {'Fwd_and_turns', 'Linear_correlation'},...
%     {'All', 'Rev_and_turns'},...
%     {'All', 'Fwd_and_turns'} };
% all_differences_names = {...
%     'Linear_correlation', 'Only_NC', 'Only_rev_and_turns', 'All',...
%     'NC_vs_linear', 'Reversal_vs_linear', 'Forward_vs_linear', ...
%     'Rev_and_turns_vs_linear', 'Fwd_and_turns_vs_linear',...
%     'All_vs_Rev_and_turns', 'All_vs_Fwd_and_turns'};
%% Now calculate the data differences by cluster
all_differences = {...
    'Linear_correlation',...
    'No_control',...
    'Rev_and_turns',...
    {'Reversal', 'Linear_correlation'},...
    {'Rev_and_turns', 'Reversal'},...
    {'All', 'Rev_and_turns'} };
all_differences_names = {...
    'Line', 'No_Control', 'Full',...
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
% for i = 1:length(all_differences_names)
%     boxplot_on_table(final_dat, all_differences_names{i});
%     ylabel('Improvement in correlation')
%     title(all_differences_names{i}, 'Interpreter', 'none')
% end

% boxplot_on_table(final_dat, []);
% ylabel('Improvement in correlation')
% title('All differences', 'Interpreter', 'none')

% boxplot_on_table(final_dat(1:4,:), [], my_cmap_3d, 'rows');
% title('Absolute correlation')
% boxplot_on_table(final_dat(5:end-2,:), [], my_cmap_3d, 'rows');
% title('Marginal correlation improvement vs linear baseline')
% boxplot_on_table(final_dat(end-1:end,:), [], my_cmap_3d, 'rows');
% title('Marginal correlation improvement vs almost complete model')

% Do not include the "other" group of neurons
clear all_figs
color_order = [4 3 2 1];
cmap = my_cmap_3d(color_order, :);
opt = {'Color', 'k', 'LineStyle', '--', 'HandleVisibility','off'};

all_figs{1} = boxplot_on_table(final_dat(1:3,1:end-1), [], cmap, 'rows');
title('Absolute correlation')
ylim([-1 1])
line([-0.5 15], [0 0], opt{:});

all_figs{2} = boxplot_on_table(final_dat(4:end,1:end-1), [], cmap, 'rows');
title('Marginal correlation improvement')
ylim([-1 1])
line([-0.5 15], [0 0], opt{:});
%==========================================================================














