

%% 3d Visualizations
% I have also found several some other 3d visualizations to be very
% helpful. First I'll go through the data processing that I used to get
% them, then show some pretty pictures.
%
% Functions used here:
%   plotSVD: a function that plots several common PCA visualizations
%   RobustPCA: an algorithm that will be explained below; implementation on
%       MathWorks
%   plot_colored: a function for plotting 3d data in different colors
%
% Original paper: Candès, E.J., Li, X., Ma, Y. and Wright, J., 2011. Robust principal component analysis?. Journal of the ACM (JACM), 58(3), p.11.
% 


%% Robust PCA
% This algorithm is not very similar to normal PCA, and I think is better
% described as a pre-processing step to determine which parts of the data
% are amenible to PCA or other algorithms. Fundamentally it attempts to
% decompose a data matrix _X_ into:
%   $$ X = L + S $$
% where _L_ is Low-rank (i.e. the SVD sigma values drop off very quickly),
% and _S_ is Sparse (i.e. there are few non-zero entries). There is a single
% important parameter here, the penalty for adding an extra non-zero entry
% to the matrix _S_ ($lambda$).
%
% To illustrate the properties of this algorithm, I'll do two
% visualizations with two different values of $lambda$. NOTE: the MATLAB
% implementation I'm using is not particularly fast, so the code blocks
% with robustPCA will take ~1 minute.
%
%% 
% First, let's import the data and filter a bit so that it is cleaner.
% Concatenate the derivatives on the end of the data for a more accurate
% picture.
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat_struct = importdata(filename);
my_filter = @(dat,w) filter(ones(w,1)/w,1,dat);
this_dat = my_filter(dat_struct.traces,3).';

use_deriv = true;
if use_deriv
    this_deriv = my_filter(dat_struct.tracesDif,3).';
    this_dat = [this_dat(:,10:end); this_deriv(:,9:end)];
end

%% 
% Do the robustPCA algorithm with a small value of lambda. This means
% that the low-rank component is VERY low-rank.
lambda = 0.0065;
[L_very_low, ~] = RobustPCA(this_dat.', lambda);
% Plot the VERY low-rank component
L_filter_very_low = my_filter(L_very_low, 10);
figure;
imagesc(L_filter_very_low.');
title('Reconstruction of the very low-rank component (data)')

%%
% These very low-rank modes have some suggestive interpretations:
% * modes 1 and 2 describe discrete underlying states
% * mode 3 describes overall drift, which may be due to stress or simple
%       chemical bleaching
% 
plotSVD(L_filter_very_low(:,15:end),struct('sigma_modes',1:3));
title('3 svd modes (very low rank component)')

%% 
% 2nd RobustPCA, with a much more sparse matrix for S (higher value of
% lambda)
lambda = 0.05;
[L_high_rank, S_very_sparse] = RobustPCA(this_dat.', lambda);

%%
% Now plot the super sparse components that the algorithm has identified;
% some of the obvious sensory neurons are clearly picked out
S_filter_very_sparse = my_filter(S_very_sparse, 10);
figure;
imagesc(S_filter_very_sparse.');
title('Reconstruction of the very sparse component (data)')

%%
% For a good 3d visualization, I've found that plotting the 'low-rank'
% component from the above algorithm (which takes out the sparse spikes)
% gives a very good 3d visualization, with separation between the different
% types of turns
%
% This plots the regular 3d-pca projection as well as a more clear colored
% version
filter_window = 10;
L_filter_high_rank = my_filter(L_high_rank,filter_window)';
[~,~,~,proj3d] = plotSVD(L_filter_high_rank(:, filter_window:end),...
    struct('PCA3d',true, 'sigma',false));
plot_colored(proj3d,...
    dat_struct.SevenStates(2*filter_window-1:end),...
    dat_struct.SevenStatesKey,'o');
title('Dynamics of the low-rank component (data)')

%%
% For comparison, we can look at the 3d diagram produced by the
% reconstructed data. The first step is to get the AdaptiveDmdc object;
% see AdaptiveDmdc_documentation for a more thorough explanation:
id_struct = struct(...
    'ID', {dat_struct.ID},...
    'ID2', {dat_struct.ID2},...
    'ID3', {dat_struct.ID3});
settings = struct('to_normalize_envelope', true,...
    'to_subtract_mean',true,...
    'to_plot_nothing',true,...
    'id_struct',id_struct);

ad_obj2 = AdaptiveDmdc(this_dat, settings);

%%
% Use this object to reconstruct the data. First, plot it in comparison to
% the original data:
approx_data = ad_obj2.plot_reconstruction(true, true).';

%%
% Now use robust PCA and visualize this using the same algorithm as above
lambda = 0.05;
[L_reconstruct, S_reconstruct] = RobustPCA(approx_data, lambda);
% Plot the 2nd low-rank component
filter_window = 10;
L_filter2 = my_filter(L_reconstruct,filter_window)';
[u,s,v,proj3d] = plotSVD(L_filter2(:,filter_window:end),...
    struct('PCA3d',true,'sigma',false));
plot_colored(proj3d,...
    dat_struct.SevenStates(2*filter_window-1:end),dat_struct.SevenStatesKey,'o');
title('Dynamics of the low-rank component (reconstructed)')
%==========================================================================


