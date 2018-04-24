function [ ad_obj_augment2 ] = ...
    robustPCA_twice_then_dmdc( to_plot, to_save, filter_window, use_deriv,...
    lambda_2nd,...
    adaptive_dmdc_settings)
% Does Robust PCA twice, then adaptive_dmdc
%   1st time: low lambda value, with most of the data in the sparse
%       component. The extremely low-rank component is interpreted as
%       encoding global states
%   2nd time: high lambda value, with most of the data in the 'low-rank'
%       component (which is actually nearly full-rank). The extremely
%       sparse component is interpreted as control signals, and the
%       'low-rank' component should be model-able using dmdc
%   adaptive_dmdc: fit a DMDc model to the 2nd 'low-rank' component, using
%       the extremely sparse component (2nd robustPCA) and the extremely
%       low-rank component (1st robustPCA) as control signals
if ~exist('to_plot','var')
    to_plot = false;
end
if ~exist('to_save','var')
    to_save = false;
end
if ~exist('filter_window','var')
    filter_window = 3;
end
if ~exist('use_deriv','var')
    use_deriv = false;
end
if ~exist('lambda_2nd','var')
    lambda_2nd = 0.05;
end
if ~exist('adaptive_dmdc_settings','var')
    adaptive_dmdc_settings = struct();
end

%% Use Robust PCA twice to separate into motor-, inter-, and sensory neurons
filename = '../../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat5.traces,filter_window).';
if use_deriv
    % This is an analysis step that Zimmer did in the original paper...
    % basically it gets the derivatives to have similar magnitudes as the
    % original data
    dTraces = dat5.tracesDif';
    %normalize by the peak value of non derivative traces
    dTracesNorm = dTraces.*...
        repmat(1./max(abs(this_dat),[],2),1,size(dTraces,2)); 

    this_deriv = my_filter(dTracesNorm, filter_window);
    this_dat = [this_dat(:,2*filter_window:end); ...
        this_deriv(:,2*filter_window-1:end)];
end

% Expect that the motor neuron dynamics are ~4d, from eigenworm analysis
%   This value of lambda (heuristically) gives a dimension of 4 for the raw
%   data
lambda = 0.0065;
[L_raw, S_raw] = RobustPCA(this_dat, lambda);
L_filter = my_filter(L_raw,10);
if to_plot
    % Plot the VERY low-rank component
    % plotSVD(L_filter(:,15:end)',struct('PCA3d',true,'sigma',false));
    % title('Dynamics of the low-rank component')
    [u_low_rank, s_low_rank] = ...
        plotSVD(L_filter(:,15:end)',struct('sigma_modes',1:3));
    title('3 svd modes (very low rank component)')
else
    [u_low_rank, s_low_rank] = ...
        plotSVD(L_filter(:,15:end)',struct('sigma',false));
end
s_low_rank = diag(s_low_rank);
card_low_rank = rank(s_low_rank);
% u_low_rank = plotSVD(L_raw,struct('sigma_modes',1:4));
% figure;
% imagesc(L_filter);
% title('Reconstruction of the very low-rank component (data)')

% 2nd RobustPCA, with much more sparsity
[L_2nd, S_2nd] = RobustPCA(this_dat, lambda_2nd);
% [L_2nd, S_2nd] = RobustPCA(S_raw, lambda);
% Plot the 2nd low-rank component
filter_window = 1;
L_filter2 = my_filter(L_2nd,filter_window);
% plotSVD(L_filter2(:,filter_window:end),struct('sigma_modes',1:3));
if to_plot
    [~,~,~,proj3d] = plotSVD(L_filter2(:,filter_window:end),...
        struct('PCA3d',true,'sigma',false));
    plot_colored(proj3d,...
        dat5.SevenStates(end-size(proj3d,2)+1:end),dat5.SevenStatesKey,'o');
    title('Dynamics of the low-rank component (data)')
    % figure;
    % imagesc(S_2nd')
    % title('Sparsest component')
    drawnow
end

% Augment full data with 2nd Sparse signal AND lowest-rank signal
%   Note: just the sparse signal doesn't work
ad_dat = L_2nd - mean(L_2nd,2);
S_2nd = S_2nd - mean(S_2nd,2);
tol = 1e-2;
S_2nd_nonzero = S_2nd(max(abs(S_2nd),[],2)>tol,:);
L_low_rank = u_low_rank(:,1:card_low_rank) * ...
    s_low_rank(1:card_low_rank,1:card_low_rank);
L_low_rank = L_low_rank - mean(L_low_rank,1);
% Create the augmented dataset
L_low_rank = L_low_rank';
num_pts = min([size(L_low_rank,2) size(S_2nd_nonzero,2) size(ad_dat,2)]);
ad_dat = [ad_dat(:,1:num_pts);...
    S_2nd_nonzero(:,1:num_pts);...
    L_low_rank(:,1:num_pts)];
% Adaptive dmdc
if isempty(fieldnames(adaptive_dmdc_settings))
    x_ind = 1:size(this_dat,1);
    id_struct = struct('ID',{dat5.ID},'ID2',{dat5.ID2},'ID3',{dat5.ID3});
    adaptive_dmdc_settings = struct('to_plot_cutoff',true,...
        'to_plot_data_and_outliers',true,...
        'id_struct',id_struct,...
        'sort_mode','user_set',...
        'x_indices',x_ind,...
        'dmd_mode','naive');
end
ad_obj_augment2 = AdaptiveDmdc(ad_dat, adaptive_dmdc_settings);

% Plot reconstructions
dat_approx_control = ad_obj_augment2.plot_reconstruction(true,true);

interesting_neurons = [41, 114, 51, 42, 45];
for i=interesting_neurons
    ad_obj_augment2.plot_reconstruction(true,true,true,i);
end

if to_save
    save('robustPCA_twice_then_dmdc_struct');
end
%==========================================================================

end

