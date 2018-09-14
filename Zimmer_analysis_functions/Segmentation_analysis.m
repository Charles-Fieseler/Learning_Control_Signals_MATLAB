
%% Use segmentation object
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

settings = struct('verbose',false);
window_settings = struct('window_step', 1);

my_seg = AdaptiveSegmentation(filename, settings, window_settings);

figure;
plot(my_seg.all_error_reductions)
title('Percent error reductions with the same control signal')

plotSVD(my_seg.all_B0',struct('PCA3d',true,'PCA_opt','o'));


%==========================================================================


%% Use segmentation object on RPCA data
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';

if ~exist('my_model_preprocess','var')
    % Use CElegans model to preprocess
    settings = struct(...
        'to_subtract_mean',false,...
        'to_subtract_mean_sparse',false,...
        'to_subtract_mean_global',false,...
        'dmd_mode','func_DMDc',...
        'filter_window_dat',6,...
        'use_deriv',false,...
        'to_normalize_deriv',false);
    settings.global_signal_mode = 'ID_binary_and_grad';
    my_model_preprocess = CElegansModel(filename, settings);
end

% Segment, but force every window to have its own calculation
% sparse_ind = my_model_preprocess.control_signals_metadata{'sparse',:}{:};
% sparse_ctr = my_model_preprocess.control_signal(sparse_ind,:);
% settings = struct('verbose',false, 'force_redo_every_window',true,...
%     'external_u',sparse_ctr);
% window_settings = struct('window_step',1, 'window_size',300);
% Just regular segmenting
settings = struct('verbose',true, 'force_redo_every_window',true);
window_settings = struct('window_step',1, 'window_size',200);
my_seg = AdaptiveSegmentation(...
    my_model_preprocess.dat, settings, window_settings);

% Plot things!
figure;
plot(my_seg.all_error_reductions)
title('Percent error reductions with the same control signal')

plotSVD(my_seg.all_B0',struct('PCA3d',true,'PCA_opt','o'));
[~,~,~,proj3d] = plotSVD(my_seg.all_B0);
plot_colored(proj3d,...
    my_model_preprocess.state_labels_ind(1:size(proj3d,2)),...
    my_model_preprocess.state_labels_key);

%==========================================================================

