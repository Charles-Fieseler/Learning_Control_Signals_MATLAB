% Produce some pid test data

%% Settings

% 2 Neurons
% sz = [2, 3000];
% kp = [0.5; 0.2];
% % ki = [0.01; 0.01];
% ki = [0; 0];
% kd = [];
% set_points = [[0.5 1];
%               [1 -1]];
          
% 1 Neuron
% sz = [1, 3000];
% kp = [0.5];
% ki = [0];
% % ki = [0.8];
% kd = [];
% set_points = [-1 0.5];

% 1 Neuron, 3 states
sz = [1, 3000];
kp = [0.5];
ki = [0];
% ki = [0.8];
kd = [];
set_points = [-1 -0.1, 1.3];

% Some dependence on the initial condition
initial_condition_dependence = zeros(size(set_points));
% initial_condition_dependence(1,1) = 1;

noise = 0.01;

% transition_mat = [[0.99 0.01];
%                   [0.01 0.99]];
transition_mat = [[0.99 0.01 0.00];
                  [0.01 0.98 0.01];
                  [0.00 0.01 0.99]];
perturbation_mat = zeros(sz);
perturbation_mat(1,100) = 0.1;
perturbation_mat(1,300) = -0.1;
perturbation_mat(1,500) = 0.2;
perturbation_mat(1,700) = -0.2;
perturbation_mat(1,1000:1050) = 0.1;
perturbation_mat(1,2000:2050) = -0.5;

num_neurons = sz(1);
              
%% Get data
[dat, ctr_signal, state_vec] = ...
    test_dat_pid(sz, kp, ki, kd, set_points, transition_mat,...
    perturbation_mat, [], [], [], initial_condition_dependence, noise);
% Simple plots
% figure; plot(dat')
% title('Data')
% figure; plot(ctr_signal')
% title('Control signal (integral error')
% figure; plot(state_vec)
% title('State vector')

%% Put in the right format, save, and create a model
ID = {{'1','2'}};
grad = gradient([dat; ctr_signal]');
% grad = gradient(dat');
dat_struct = struct(...
    ...'traces', {[dat; ctr_signal]'},...
    'traces', [dat; ctr_signal]',...
    'tracesDif',grad(1:end-1,:),...
    'ID',ID,...
    'ID2',ID,...
    'ID3',ID,...
    'TwoStates', state_vec,...
    'TwoStatesKey',{{'State 1','State 2','State 3'}},...
    'fps',1);

use_deriv = false;
augment_data = 0;
ctr_signal = ctr_signal(:,1:end-augment_data);
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    ...'filter_window_dat',0,...
    'filter_window_dat',10,...
    'filter_window_global',0,...
    'add_constant_signal',false,...
    ...'dmd_mode','func_DMDc',...
    ...'dmd_mode','sparse',...
    'augment_data',augment_data,...
    'use_deriv',use_deriv,...
    ...'AdaptiveDmdc_settings',struct('x_indices',x_ind),...
    ...'custom_global_signal',ctr_signal,...
    'custom_control_signal',ctr_signal((num_neurons+1):end,:),... % Not using the integral error, only perturbations
    ...'lambda_sparse',0.022); % This gets some of the perturbations, and some of the transitions...
    'lambda_sparse',0); % Don't want a sparse signal here
settings.global_signal_mode = 'ID_binary';
my_model_PID1 = CElegansModel(dat_struct, settings);

%% Now redo the model but with the additional integral control signal 
% settings.global_signal_mode = 'ID_and_length_count';
% settings.global_signal_mode = 'ID_binary_and_length_count';
% settings.global_signal_mode = 'ID_binary_and_x_times_state';
% my_model_PID2 = CElegansModel(dat_struct, settings);

%% Now redo the model but with 2x additional integral control signals

use_cumulative_sum = true;
if use_cumulative_sum
    % Use a cumulative sum signal
    settings.global_signal_mode = 'ID_binary_and_cumsum_x_times_state_and_length_count';
    signal_functions = {SumXtimesStateDependentRow()};
    setup_arguments = {'normalize_cumsum_x_times_state'};
    signal_indices = {'cumsum_x_times_state'};
    dependent_signals = table(signal_functions, signal_indices, setup_arguments);
else
    % Use an integral signal
    settings.global_signal_mode = 'ID_binary_and_cumtrapz_x_times_state_and_length_count';
    signal_functions = {TrapzXtimesStateDependentRow()};
    setup_arguments = {'normalize_cumtrapz_x_times_state'};
    signal_indices = {'cumtrapz_x_times_state'};
    dependent_signals = table(signal_functions, signal_indices, setup_arguments);
end

% num_neurons = 2;
% num_states = 2;

% Define the table for the dependent row objects
% settings.dependent_signals = dependent_signals;
% 
% my_model_PID3 = CElegansModel(dat_struct, settings);

%% A different model for matrices with unknown states (but good segmentation)
tol = 1e-3;
% noise = 3e-5;
cascade_ind = cumsum(...
    abs(noise*(2*rand(size(state_vec))-1) + ...
    [zeros(size(state_vec,1),1) diff(state_vec)])>tol)...
    + 1;
num_segments = length(unique(cascade_ind));
state_key = cell(num_segments,1);
for j = 1:num_segments
    state_key{j} = sprintf('state_%d',j);
end

dat_struct2 = struct(...
    ...'traces', {[dat; ctr_signal]'},...
    'traces', dat',...
    'tracesDif',grad(1:end-1,:),...
    'ID',ID,...
    'ID2',ID,...
    'ID3',ID,...
    'TwoStates', cascade_ind,...
    'TwoStatesKey',{state_key},...
    'fps',1);

settings.global_signal_mode = 'ID_binary';
if isfield(settings,'dependent_signals')
    settings = rmfield(settings,'dependent_signals');
end

my_model_PID4 = CElegansModel(dat_struct2, settings);

figure;
subplot(2,1,1)
b_rows = my_model_PID4.AdaptiveDmdc_obj.A_original(1:num_neurons,...
    (num_neurons+1):(end-1));
learned_state_vec = zeros(size(state_vec));
for i=1:size(b_rows,2)
    learned_state_vec(logical(my_model_PID4.control_signal(i,:))) = b_rows(i);
end
imagesc(learned_state_vec)
title('Control dynamics (learned coefficients)')
subplot(2,1,2)
imagesc(state_vec)
title('Control dynamics (true labels)')

%my_model_PID4.plot_reconstruction_interactive(false)

%% Set up the true dynamics
% A_true = zeros(sz(1), size(my_model_PID3.control_signal,1));
% for i = 1:sz(1)
%     A_true(i,:) = [];
% end
if ~any(any(initial_condition_dependence))

    if any(ki>0)
        obj = my_model_PID3;
        A_true = ...
            [eye(num_neurons).*(1-kp)... % Intrinsic dynamics
            set_points.*kp... % ID_binary
            ...zeros(num_neurons,1)... % Constant
            -eye(num_neurons).*ki./obj.normalize_cumsum_x_times_state... % s1*x1
            -eye(num_neurons).*ki./obj.normalize_cumsum_x_times_state... % s2*x2
            set_points.*ki./obj.normalize_length_count... % Length counts
            eye(num_neurons)]; % perturbations
    else
        obj = my_model_PID1;
        A_true = ...
            [eye(num_neurons).*(1-kp)... % Intrinsic dynamics
            set_points.*kp... % ID_binary
            ...zeros(num_neurons,1)... % constant
            eye(num_neurons)];% Perturbations
    end
    A_true = [A_true;...
        zeros(size(obj.control_signal,1), size(A_true,2))];

    set_true = struct('external_A_orig',A_true, 'dmd_mode','external',...
        'to_plot_nothing',true,...
        'sort_mode','user_set', 'x_indices',1:num_neurons,...
        'to_subtract_mean',false);
%     my_model_true = AdaptiveDmdc(obj.dat_with_control, set_true);
% 
%     my_model_true.plot_reconstruction(true,true,true,1);
end

%% Plots
% my_model_PID3.plot_reconstruction_interactive();
% 
% [x, ctr] = my_model_PID3.generate_time_series(3000);

% figure;
% plot(x(1,:));
% hold on
% plot(my_model_PID3.dat(1,:))
% legend({'First channel reconstruction','data'})

% figure;
% plot(ctr(4,:));
% hold on
% plot(my_model_PID3.control_signal(4,:))

% figure;
% plot(o.control_signal(4,:))
% hold on
% plot(o.control_signal(6,:))
% legend({'Cumulative Sum','Length Count'})

% figure;
% plot(obj.control_signal(4,:)./obj.normalize_cumsum_x_times_state)
% hold on
% plot(set_points(2)*obj.control_signal(6,:)./obj.normalize_length_count)
% legend({'Cumulative Sum','Length Count'})
% 
% figure;
% plot( -(set_points(2)*obj.control_signal(6,:)./obj.normalize_length_count - ...
%     obj.control_signal(4,:)./obj.normalize_cumsum_x_times_state) )
% hold on
% plot(ctr_signal(1,:))
% legend({'Best learn-able control signal',...
%     'Real control signal'})

% my_model_PID1.plot_reconstruction_interactive(false);
% title('Only integral controller')
% my_model_PID2.plot_reconstruction_interactive(false);
% title('Also has length counts')

% figure;imagesc(my_model_PID1.AdaptiveDmdc_obj.A_separate)
% title('Only integral controller')
% figure;imagesc(my_model_PID2.AdaptiveDmdc_obj.A_separate)
% title('Also has length counts')

%% Scratch work on learning the control signal
real_transitions = [360, 367,  870, 907];

% Function for getting windows
%   Important: first window can't just have each channel as constant!
window_length = 30;
x0 = 871; % Has non-trivial dynamics here
get_X = @(i) dat( (x0+i):(x0+window_length+i) );
get_X_U = @(i) [get_X(i); ...
    ones(1,window_length+1)];

% Initialize
AB1 = get_X(1)/get_X_U(0);

tend = 10;
err = zeros(1,tend);
for i=1:tend
    err(i) = norm(get_X(i+1) - AB1*get_X_U(i));
end

disp(err)









