function [fig1, fig2, this_caxis, this_cmap] = ...
    plot_signal_reconstruction_w_threshold(...
    my_model_time_delay, B_prime_lasso_td_3d, all_intercepts_td,...
    all_thresholds_3d, ...
    which_ctr, which_iter,...
    this_caxis, this_cmap, num_neurons_to_plot, fig_opt, tspan)
% Produces two figures:
%   the unrolled matrix of neurons that predict a control signal, i.e. the
%       "variable selection" as performed by LASSO
%   the signal and a reconstruction with the threshold line

% Settings for both iteration plots
if ~exist('this_caxis', 'var')
    this_caxis = [];
end
if ~exist('num_neurons_to_plot', 'var')
    num_neurons_to_plot = 10;
end
if ~exist('fig_opt', 'var')
    fig_opt = {'DefaultAxesFontSize', 24};
end
if ~exist('tspan', 'var')
    tspan = [0 3000];
end

% Plot the unrolled matrix of one control signal
%---------------------------------------------
% First plot
names = my_model_time_delay.get_names([], true);
% ind = contains(my_model_time_delay.state_labels_key, {'REV', 'DT', 'VT'});
% Unroll one of the controllers
unroll_sz = [my_model_time_delay.original_sz(1), my_model_time_delay.augment_data];
dat_unroll1 = reshape(B_prime_lasso_td_3d(which_ctr,:,which_iter), unroll_sz);
fig4_normalization = max(max(abs(dat_unroll1))); % To use later as well

fig1 = figure(fig_opt{:});
[ordered_dat, ordered_ind] = top_ind_then_sort(dat_unroll1, num_neurons_to_plot);
imagesc(ordered_dat./fig4_normalization)
if isempty(this_caxis)
    this_cmap = cmap_white_zero(ordered_dat);
    this_caxis = caxis;
    colormap(this_cmap); % i.e. equal to the first plot
    colorbar
else
    colormap(this_cmap); % i.e. equal to the first plot
    caxis(this_caxis)
end
if which_iter > 1
    title(sprintf('%d Neurons Eliminated', which_iter-1))
else
    title('Predictors (All Neurons)')
end
yticks(1:unroll_sz(1))
yticklabels(names(ordered_ind))
xlabel('Delay frames')

%---------------------------------------------
% Plot a reconstruction
%---------------------------------------------
% Get the reconstructions
X1 = my_model_time_delay.dat(:,1:end-1);
ctr = my_model_time_delay.control_signal(which_ctr,:);
ctr_reconstruct = B_prime_lasso_td_3d(which_ctr,:,which_iter) * X1;
ctr_reconstruct_td = [ctr(1), ...
    ctr_reconstruct + all_intercepts_td(which_ctr, which_iter)];

% Plot
[~, ~, ~, ~, ~, ~, fig2] = ...
    calc_false_detection(ctr, ctr_reconstruct_td,...
            all_thresholds_3d(which_ctr, which_iter), [], [], true, true, false);
if which_iter > 1
    title(sprintf('%d Neurons Eliminated', which_iter-1))
else
    title('Reconstruction (All Neurons)')
end
set(gca, 'box', 'off')
set(gca, 'FontSize', fig_opt{2})
% legend off
yticklabels('')
xlabel('Time')
xlim(tspan)

end