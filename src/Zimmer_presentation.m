% A simple script to produce plots for a presentation
to_save = true;
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\figures\';


%% Fixed points
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
view(my_viewpoint)
% Now plot the fixed points
my_model_fixed_points.run_with_only_global_control(@(x)...
    plot_colored_fixed_point(x,'REVSUS', true, all_figs{1}) );
my_model_fixed_points.run_with_only_global_control(@(x)...
    plot_colored_fixed_point(x,'FWD', true, all_figs{1}) );
[~, b] = all_figs{1}.Children.Children;

% Add invisible axes on top and place the stars there so they will be
% visible
% ax = all_figs{1}.Children(2);
% axHidden = axes('Visible','off','hittest','off'); % Invisible axes
% linkprop([ax axHidden],{'CameraPosition' 'XLim' 'YLim' 'ZLim' 'Position'}); % The axes should stay aligned
% set(b(1), 'Parent', axHidden)
% set(b(2), 'Parent', axHidden)

% Save figures
if to_save
    fname = sprintf('%sfigure_5_%d', foldername, 1);
    this_fig = all_figs{1};
    prep_figure_no_axis(this_fig)
%     ax.Visible = 'Off';
%     axes(ax)
%     zoom(1.175) % Decided by hand
%     axes(axHidden)
%     zoom(1.175) % Decided by hand
    saveas(this_fig, fname, 'png');
end
%==========================================================================

