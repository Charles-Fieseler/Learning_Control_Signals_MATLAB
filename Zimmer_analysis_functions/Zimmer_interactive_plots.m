

%% Create CElegansModel Object
% Use default settings, except for analysis mode
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
settings = struct();
% This way to get the global signal corresponds to column 3 in the paper,
% and is much faster to do than column 2
%   Note that we're using all default settings here, so the plots produced
%   will be slightly different
which_column = 3;
switch which_column
    case 1
        error('Not implemented yet; messier to do')
    case 2
        settings.global_signal_mode = 'ID_and_offset';
    case 3
        settings.global_signal_mode = 'ID_and_offset';
    otherwise
        error('Only 3 columns in paper plot')
end

% Actual calculations
my_model = CElegansModel(filename, settings);
%==========================================================================


%% Plot the overall reconstruction
help my_model.plot_reconstruction_interactive

include_control_signal = false;
my_model.plot_reconstruction_interactive(include_control_signal)
%==========================================================================
