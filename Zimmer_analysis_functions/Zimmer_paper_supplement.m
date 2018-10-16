
%% Define folder to save in and data folders
to_save = false;
foldername = 'C:\Users\charl\Documents\Current_work\Zimmer_draft_paper\Supplement\Supplemental_figures\';
my_viewpoint = [0, 80];

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
%==========================================================================


%% Define colormaps
%---------------------------------------------
% Set up colormap for RPCA visualizations
%---------------------------------------------
num_colors = 16;
my_cmap_RPCA = brewermap(num_colors,'OrRd');
my_cmap_RPCA(1,:) = ones(1,3);
colormap(my_cmap_RPCA)
my_cmap_RPCA(end,:) = zeros(1,3);
set(0, 'DefaultFigureColormap', my_cmap_RPCA)
% caxis([-0.5,1.0])

%---------------------------------------------
% Set up colormap for 3d visualizations
%---------------------------------------------
num_possible_roles = 4;
my_cmap_3d = colormap(parula(num_possible_roles));
my_cmap_3d = [my_cmap_3d; zeros(1,3)];
my_cmap_3d(4,:) = [249, 222, 12]./256; % Make yellow more visible

% Dict for switching between simple label plots and bar-graph-matching
% colors
my_cmap_dict = containers.Map(...
    {1, 2, 3, 4, 5},... %original: NOSTATE, REV, VT, DT, FWD
    {5, 4, 1, 2, 3}); %Want: VT, DT, FWD, REV, NOSTATE

close all
%==========================================================================



%% Build all models
%---------------------------------------------
% Define all settings
%---------------------------------------------
settings = struct(...
    'to_subtract_mean', true,...
    'to_subtract_mean_sparse', false,...
    'to_subtract_mean_global', false,...
    'dmd_mode', 'func_DMDc',...
    'add_constant_signal', false,...
    'to_add_stimulus_signal', false,...
    'filter_window_dat', 0,...
    'global_signal_mode', 'ID_binary',...
    'use_deriv', false);

settings_full = settings;

settings_global_only = settings;
settings_global_only.lambda_sparse = 0;

settings_no_control = settings;
settings_no_control.global_signal_mode = 'None';
settings_no_control.lambda_sparse = 0;
settings_no_control.dmd_mode = 'tdmd';

%---------------------------------------------
% Build all models
%---------------------------------------------
all_models_full = cell(n,1);
all_models_global_only = cell(n,1);
all_models_no_control = cell(n,1);

for i = 1:n
    dat_struct = importdata(all_filenames{i});
    if i > num_type_1
        % A lot of the prelet data files have very bad initial frames
        dat_struct.traces = dat_struct.traces(100:end,:);
    end
    all_models_full{i} = CElegansModel(dat_struct, settings_full);
    all_models_global_only{i} = ...
        CElegansModel(dat_struct, settings_global_only);
    all_models_no_control{i} = ...
        CElegansModel(dat_struct, settings_no_control);
end
%==========================================================================


%% 3d Reconstructions (all models)
all_figs_data = cell(n,1);
all_figs_full = cell(n,1);
all_figs_global_only = cell(n,1);
all_figs_no_control = cell(n,1);

%---------------------------------------------
% Create figures
%---------------------------------------------
for i = 1:n
    m = all_models_full{i};
    m.set_simple_labels();
    all_figs_full{i} = m.plot_colored_reconstruction(false);
    title('Proportional and sparse control')
    
    m = all_models_global_only{i};
    m.set_simple_labels();
    all_figs_global_only{i} = m.plot_colored_reconstruction(false);
    title('Only proportional control')
    
    m = all_models_no_control{i};
    m.set_simple_labels();
    all_figs_no_control{i} = m.plot_colored_reconstruction(false);
    title('No control')
    
    all_figs_data{i} = m.plot_colored_data(false, 'o');
    drawnow;
end

%---------------------------------------------
% Now make the colormap match the bar graphs
%---------------------------------------------
all_figs = {all_figs_full, all_figs_global_only, ...
    all_figs_no_control, all_figs_data};
% new_labels_key = ...
%     {'Simple Forward',...
%     'Dorsal Turn',...
%     'Ventral Turn',...
%     'Simple Reverse',...
%     'NOSTATE'};
% for i = 1:n
%     for i2 = 1:length(all_figs)
%         this_fig = all_figs{i2}{i};
%         for i3 = 1:length(new_labels_key)
%             try
%                 this_fig.Children(2).Children(i3).CData = ...
%                     my_cmap_3d(my_cmap_dict(i3),:);
%             catch
%                 warning('Ignoring extra colors')
%             end
%         end
% %         view(my_viewpoint)
%     end
% end

%---------------------------------------------
% Save figures
%---------------------------------------------
if to_save
    for i = 1:length(all_figs)
        if isempty(all_figs{i})
            continue;
        end
        for i2 = 1:n
            fname = sprintf('%sfigure_s_%d_%d', foldername, i2, i);
            this_fig = all_figs{i}{i2};
            ax = this_fig.Children(2);
            ax.Clipping = 'Off';
            prep_figure_no_axis(this_fig);
            zoom(1.12);
            saveas(this_fig, fname, 'png');
        end
    end
end

%==========================================================================
