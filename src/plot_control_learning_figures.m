warning('This is a script and will modify global variables')
warning('Make sure my_model_base is initialized, and control signals are learned')

%% PLOT3: Visual connection with expert signals
%---------------------------------------------
% Note: replot the "correct" signals
tspan = 100:1000; % decided by hand, SAME AS ABOVE FIGURE
% ctr = my_model_base.control_signal;
my_model_simple = my_model_base;
my_model_simple.set_simple_labels();
my_model_simple.remove_all_control();
my_model_simple.build_model();
ctr = my_model_simple.control_signal;

%---------------------------------------------
% Reversal figure
%---------------------------------------------
all_figs{1} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(4,:), 'LineWidth', 2};

subplot(2,1,1)
plot(ctr(4,tspan), opt{:})
title('Reversal')
ylabel('Expert')

subplot(2,1,2)
[learned_u_REV] = get_signal_actuating_neuron(...
    'AVA', text_names, registered_lines, all_U1);
plot(learned_u_REV(tspan), opt{:})
xlim([0 length(tspan)])
ylabel('Learned')

prep_figure_no_box_no_zoom(all_figs{3});

%---------------------------------------------
% Dorsal turn figure
%---------------------------------------------
all_figs{2} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(2,:), 'LineWidth', 2};

subplot(2,1,1)
plot(ctr(2,tspan), opt{:})
title('Dorsal Turn')

subplot(2,1,2)
[learned_u_DT] = get_signal_actuating_neuron(...
    'SMDD', text_names, registered_lines, all_U1);
plot(learned_u_DT(tspan), opt{:})
xlim([0 length(tspan)])

prep_figure_no_box_no_zoom(all_figs{4});

%---------------------------------------------
% Ventral turn figure
%---------------------------------------------
all_figs{3} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(1,:), 'LineWidth', 2};

subplot(2,1,1)
plot(ctr(3,tspan), opt{:})
title('Ventral Turn')

subplot(2,1,2)
neur = 'SMDV';
[learned_u_VT] = get_signal_actuating_neuron(...
    neur, text_names, registered_lines, all_U1);
plot(learned_u_VT(tspan), opt{:})
xlim([0 length(tspan)])

prep_figure_no_box_no_zoom(all_figs{5});

%---------------------------------------------
% Forward figure
%---------------------------------------------
all_figs{4} = figure('DefaultAxesFontSize', 14);
opt = {'color', my_cmap_3d(3,:), 'LineWidth', 2};

subplot(2,1,1)
plot(ctr(1,tspan), opt{:})
title('Forward')

subplot(2,1,2)
[learned_u_FWD] = get_signal_actuating_neuron(...
    {'RIB', 'AVB'}, text_names, registered_lines, all_U1);
plot(learned_u_FWD(tspan), opt{:})
xlim([0 length(tspan)])

prep_figure_no_box_no_zoom(all_figs{6});

%---------------------------------------------
