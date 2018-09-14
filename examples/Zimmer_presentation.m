% A simple script to produce plots for a presentation
to_save = false;
foldername = '.\';


%% Interactive: Fixed points
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
my_model_fixed_points = CElegansModel(filename, settings);

% Plot the original data
all_figs{1} = my_model_fixed_points.plot_colored_data(false, 'o');
% Now plot the fixed points
my_model_fixed_points.run_with_only_global_control(@(x)...
    plot_colored_fixed_point(x,'REVSUS', true, all_figs{1}) );
my_model_fixed_points.run_with_only_global_control(@(x)...
    plot_colored_fixed_point(x,'FWD', true, all_figs{1}) );

% Save figures
if to_save
    fname = sprintf('%sfigure_1_%d', foldername, 1);
    this_fig = all_figs{1};
    saveas(this_fig, fname);
end
%==========================================================================


%% Interactive: Only state labels (movie)
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
all_figs = cell(1,1);
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'global_signal_mode', 'ID_binary_and_grad',...
    'lambda_sparse',0);
my_model_state_labels = CElegansModel(filename, settings);

% Plot the original data
all_figs{1} = my_model_state_labels.plot_reconstruction_interactive(false, 45);

movie_filename = '';
my_model_state_labels.plot_colored_arrow_movie(...
                [], [], movie_filename, [], [], [], true);

% Save figures
if to_save
    fname = sprintf('%sfigure_2_%d', foldername, 1);
    this_fig = all_figs{1};
    prep_figure_no_axis(this_fig);
    saveas(this_fig, fname, 'png');
end

%==========================================================================


%% Basic plots: RPCA of sensory neuron
all_figs = cell(3,1);
% Use objects from above
settings = struct(...
    'to_subtract_mean',true,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'use_deriv',true,...
    'to_normalize_deriv',true,...
    'global_signal_mode', 'ID_binary_and_grad',...
    'lambda_sparse',0.11);
my_model_very_sparse = CElegansModel(filename, settings);

% Plot same neuron with different levels of sparsity
all_figs{1} = my_model_state_labels.plot_reconstruction_interactive(true,15);
all_figs{2} = my_model_very_sparse.plot_reconstruction_interactive(true,15);
all_figs{3} = my_model_fixed_points.plot_reconstruction_interactive(true,15);

% Save figures
if to_save
    for i = 1:length(all_figs)
        fname = sprintf('%sfigure_3_%d', foldername, i);
        this_fig = all_figs{i};
        prep_figure_no_axis(this_fig);
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================


%% Basic plots: Individual neuron reconstructions
% Use objects from above
neurons_of_interest = [...
    ...119,... % Unlabeled
    124,... % VA01
    90,... % SMBDL
    114,... % VB02
    45];% AVAR
all_figs = cell(length(neurons_of_interest),1);
for i = 1:length(neurons_of_interest)
%     all_figs{2*i-1} = ...
%         my_model_state_labels.plot_reconstruction_interactive(...
%         false,neurons_of_interest(i));
%     all_figs{2*i} = ...
%         my_model_fixed_points.plot_reconstruction_interactive(...
%         false,neurons_of_interest(i));
    all_figs{i} = ...
        my_model_fixed_points.plot_reconstruction_interactive(...
        false,neurons_of_interest(i));
end

% Save figures
if to_save
    for i = 1:length(all_figs)
        fname = sprintf('%sfigure_4_%d', foldername, i);
        this_fig = all_figs{i};
        prep_figure_no_axis(this_fig);
        saveas(this_fig, fname, 'png');
    end
end
%==========================================================================

