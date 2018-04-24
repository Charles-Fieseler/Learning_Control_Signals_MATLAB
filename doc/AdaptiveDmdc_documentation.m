

%% Create the analysis object
% This analysis uses the 'AdaptiveDmdc' class, i.e. adaptive Dynamic Mode
% Decomposition with Control, and the steps of this algorithm will be
% explained in this document
% First, we want to set the options for our analysis. This is done using a
% struct, but all the defaults should run as is. For now we will turn off
% all plotting, to be explained one-by-one
settings = struct('to_plot_nothing',true);
%%
% Next we need to set the filename to be analyzed. Clearly this will depend
% on your folder organization.
filename = '../../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
%%
% I'm not exactly sure of the format yet, so for now let's get the data
% that we actually want to analyze out manually
Zimmer_struct = importdata(filename);
dat = Zimmer_struct.traces.';
%%
% We can also add the neuron names to the settings struct, so that the
% algorithm can use them for some plots.
id_struct = struct(...
    'ID', {Zimmer_struct.ID},...
    'ID2', {Zimmer_struct.ID2},...
    'ID3', {Zimmer_struct.ID3});
settings.id_struct = id_struct;
%%
% Finally, we create the analysis object.
ad_obj = AdaptiveDmdc(dat, settings);

%%
% This object has various properties that are explained by typing either:
%
%   >> help AdaptiveDmdc
%   
%   >> help ad_obj
%
% Note that these display a lot of text!

%% The algorithm
% The algorithm will be explained by going through the documentation
% function by function, starting with the initializer
help AdaptiveDmdc.AdaptiveDmdc

%%
% Importing the settings is pretty clear. Preprocessing has several
% options, all set in the above 'settings' struct, e.g. subtracting the
% mean. The first algorithmic piece is the function "calc_data_outliers":
help AdaptiveDmdc.calc_data_outliers

%%
help AdaptiveDmdc.calc_outlier_indices

%%
help AdaptiveDmdc.calc_dmd_and_errors
%%
help AdaptiveDmdc.plot_using_settings

%%
% I've been using the last, optional, function as a lead for a way to
% understand the latent states and transition signals, but it's exploratory
% more than anything
help AdaptiveDmdc.augment_and_redo

%% Plotting
% There are several plotting functions that are useful for visualizing what
% is going on, and one that is interactive.
% Some sorting methods (e.g. sparsePCA) do not use this type of error
% detection, but all of the 'DMD_*' methods do. This command by default
% plots how the signals were used to sort the neurons. Each data
% point can be clicked to show the residual (original data - DMD fit), with
% data points marked if they contributed to the counted error.
%
% NOTE: has interactivity if plotted
ad_obj.plot_data_and_outliers();

%%
% Another useful visualization is a two-part visualization of the data +
% control signal the left and the data with the control signal set to 0 on
% the right
ad_obj.plot_data_and_control();

%%
% Visualizations related to the key result is are reconstructions, either
% with control or not. An uncontrolled linear model does not at all
% capture the dynamics in C elegans, and often leads to NaN or 0 value
% predictions.
%   Just to reiterate: this algorithm takes values out of the passed data
%   and uses them as predictors (the control signal). This means that the
%   entire dataset is not reconstructed, only part of it.
help AdaptiveDmdc.plot_reconstruction
%%
% This is using control, and displays only part of the data:
ad_obj.plot_reconstruction(true);

%% 
% This plot is without control, and shows all the data
ad_obj.plot_reconstruction();

%%
% We can also display individual neuron reconstructions
neuron_ind = 46;
use_control = true;
include_control_signal = true;
to_compare_raw = true;
ad_obj.plot_reconstruction(...
    use_control, include_control_signal, to_compare_raw, neuron_ind);

%% 
% Another neuron with interesting behavior
neuron_ind = 33;
ad_obj.plot_reconstruction(...
    use_control, include_control_signal, to_compare_raw, neuron_ind);

%==========================================================================


