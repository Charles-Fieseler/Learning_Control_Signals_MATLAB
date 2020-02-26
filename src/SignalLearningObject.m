classdef SignalLearningObject < SettingsImportableFromStruct
    %% Separates control signals and intrinsic dynamics
    %
    % INPUTS:
    %       file_or_dat: struct (recommended) or matrix of data, or
    %           filename that contains the data. If a struct, should have
    %           the following fields:
    %
    %       settings: struct of options. Need not be passed. Run
    %           >> defaults = my_SignalLearningObject.set_defaults()
    %           to get the default options. These are:
    %         verbose
    %         % Getting the control signal
    %         designated_controller_channels - Designate some of the data
    %               input rows as controllers
    %         to_add_stimulus_signal - Bool; looks for a stimulus struct and
    %               attempts to add a control signal.
    %               See method: add_stimulus_signal()
    %         global_signal_mode - Mode for how to produce control signals.
    %               Default processes labeled behaviors into an onset signal
    %         global_signal_subset - If labeled behaviors are found, a subset
    %               can be designated as controllers, or 'all' (default)
    %         global_signal_pos_or_neg - 'only_pos', 'only_neg', or
    %               'pos_and_neg'. Postprocessing step on the control signals,
    %               determining whether to keep only the positive portion, the
    %               negative, or both.
    %         enforce_diagonal_sparse_B - whether to enforce a diagonal
    %               patterns on the control signal mapping
    %         enforce_zero_entries - whether to enforce a sparsity pattern on
    %               the dynamics matrix (A) or the control mapping (B), in a
    %               cell array of the format:
    %                    2x1 cell array: {from, to}
    %                    i.e. {neuron ID or number, neuron or ctr_signal}
    %               Note that 'neuron ID' is only possible for C elegans
    %               connectome data.
    %               Note also that this option requires 'dmd_mode' to support
    %               sparsity enforcement. Otherwise, this errors
    %
    %         % Data processing
    %         filter_window_dat - if >0, applies a moving average filter on the
    %               data
    %         filter_window_global - if >0, applies a moving average filter on
    %               the global signal (recommended)
    %         filter_aggressiveness - if >0, applies a low-pass filer to each
    %               neuron using MATLAB's cheby2 filter method. Attempts to
    %               learn the cutoff frequency
    %               See method: smart_filter()
    %         autocorrelation_noise_threshold - Removes neurons of they have a
    %               lower autocorrelation than this threshold
    %         to_subtract_mean - whether to mean subtract data
    %         to_subtract_baselines - whether to mean subtract a 'baseline'
    %               defined per neuron by finding long stretches of 'flat'
    %               activity.
    %               See method: calc_baseline()
    %         to_subtract_mean_global - whether to mean subtract global signal
    %         offset_control_signal - amount to offset control signal in time
    %         dmd_mode - which method to use for dmdc, i.e. calculating the A
    %               and B matrices. Different methods may have external
    %               dependencies and settings. See AdaptiveDmdc for supported
    %               methods
    %
    %         AdaptiveDmdc_settings - Settings for the external class that
    %               executes the core DMDc algorithm. See: AdaptiveDmdc
    %         % Data importing
    %         augment_data - number of time-delay embeds to use
    %         use_deriv - whether to use a pre-calculated derivative of the
    %               data. Requires a specific field and the struct input method
    %         use_only_deriv - same as above, but use only the derivative
    %         to_normalize_deriv - whether to divide the derivatives by the std
    %               of the ORIGINAL data
    %         to_save_raw_data - whether to keep the raw data; can help with
    %               memory usage to turn off
    %
    %         % Additional rows
    %         add_constant_signal - whether to add a new row that is just
    %               constant
    %
    %         % Plotting
    %         cmap - colormap to be used by all plotting functions
    %
    % OUTPUTS -
    %   OUTPUT1 -
    %   OUTPUT2 -
    %
    % EXAMPLES
    %
    %   EXAMPLE1
    %
    %
    %   EXAMPLE2
    %
    %
    %
    % Dependencies
    %   .m files, .mat files, and MATLAB products required:(updated on 21-Feb-2020)
    %             MATLAB (version 9.4)
    %             Signal Processing Toolbox (version 8.0)
    %             Statistics and Machine Learning Toolbox (version 11.3)
    %             convertKato2Zimmer.m
    %             checkModes.m
    %             func_DMD.m
    %             func_DMDc.m
    %             AdaptiveDmdc.m
    %             sparse_dmd_fast.m
    %             AbstractDmd.m
    %             SettingsImportableFromStruct.m
    %             top_ind_then_sort.m
    %             optimal_truncation.m
    %             calc_contiguous_blocks.m
    %             calc_snr.m
    %             svd_truncate.m
    %             cmap_white_zero.m
    %             plotSVD.m
    %             plot_2imagesc_colorbar.m
    %             plot_colored.m
    %             plot_gray_lines.m
    %             plot_std_fill.m
    %             brewermap.m
    %             RobustPCA.m
    %             acf.m
    %             optimal_SVHT_coef.m
    %             TVRegDiff.m
    %
    %   See also: AdaptiveDMDc
    %
    % Author: Charles Fieseler
    % University of Washington, Dept. of Physics
    % Email address: charles.fieseler@gmail.com
    % Website: coming soon
    % Created: 24-Apr-2018
    %========================================
    
    properties (SetAccess={?SettingsImportableFromStruct}, Hidden=true)
        verbose
        % Getting the control signal
        designated_controller_channels
        to_add_stimulus_signal
        global_signal_mode
        global_signal_subset
        global_signal_pos_or_neg
        enforce_diagonal_sparse_B
        enforce_zero_entries
        
        % Data processing
        filter_window_dat
        filter_window_global
        filter_aggressiveness
        autocorrelation_noise_threshold
        to_subtract_mean
        to_subtract_baselines
        to_subtract_mean_global
        offset_control_signal
        dmd_mode
        AdaptiveDmdc_settings
        % Data importing
        augment_data
        use_deriv
        use_only_deriv
        to_normalize_deriv
        to_save_raw_data
        
        % Additional rows
        add_constant_signal
        
        % Plotting
        cmap
    end
    
    properties (Access=private, Transient=true)
        raw
        raw_deriv
        
        L_global_raw
        S_global_raw
        L_sparse_raw
        S_sparse_raw
        
        state_labels_ind_raw
    end
    
    properties (Hidden=true)%, SetAccess=private)
        dat
        original_sz
        dat_sz
        total_sz
        num_data_pts
        time_vec
        fps
        
        A_old
        dat_old
        
        % For time-delay embedding
        A_unroll
        
        % Preprocessing
        baselines
        
        % Separation matrices
        L_global
        S_global
        L_global_modes
        S_global_nnz
        L_global_rank
        
        L_sparse
        S_sparse
        L_sparse_modes
        S_sparse_nnz
        L_sparse_rank
        
        state_labels_ind
        state_labels_ind_old
        state_labels_key
        state_labels_key_old
        
        stimulus_struct
        
        % Generative model
        normalize_length_count
        
        % Changing the control signals and/or matrix
        user_control_matrix
        user_control_input
        user_neuron_ablation
        user_control_reconstruction
        
        custom_control_signal
    end
    
    properties (SetAccess=private)
        control_signal
        control_signals_metadata
        
        AdaptiveDmdc_obj
    end
    
    properties (Dependent)
        dat_without_control
        dat_with_control
        x_indices
    end
    
    methods
        function self = SignalLearningObject(file_or_dat, settings)
            
            %% Set defaults and import settings
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
            %==========================================================================
            
            %% Import data
            if ischar(file_or_dat)
                tmp_dat = importdata(file_or_dat);
                if isstruct(tmp_dat)
                    % Assume a "classic" Zimmer struct
                    self.import_from_struct(tmp_dat);
                elseif isnumeric(tmp_dat)
                    self.raw = tmp_dat;
                else
                    error('Filename must contain a struct or matrix')
                end
            elseif isnumeric(file_or_dat)
                self.raw = file_or_dat;
            elseif isstruct(file_or_dat)
                % Assume a "classic" Zimmer struct
                self.import_from_struct(file_or_dat);
            elseif iscell(file_or_dat)
                % Assume a "raw" (i.e. Kato) Zimmer struct
                assert(length(file_or_dat)==2,...
                    'Must pass two strings for Kato-type import')
                self.import_from_struct(...
                    convertKato2Zimmer(file_or_dat));
            else
                error('Must pass data matrix or filename')
            end
            self.preprocess();
            %==========================================================================
            
            %% Get control signal and DMDc model with that signal
            self.build_model();
            %==========================================================================
            
            %% Initialize user control structure
            self.reset_user_control();
            %==========================================================================
            
        end
        
        function calc_AdaptiveDmdc(self)
            % Uses external class AdaptiveDmdc
            self.AdaptiveDmdc_settings.initial_sparsity_pattern = ...
                self.get_initial_sparsity_pattern();
            
            self.AdaptiveDmdc_obj = AdaptiveDmdc(...
                [self.dat; self.control_signal],...
                self.AdaptiveDmdc_settings);
        end
        
        function sparsity = get_initial_sparsity_pattern(self)
            if self.enforce_diagonal_sparse_B || ...
                    ~isempty(self.enforce_zero_entries)
                assert(strcmp(self.dmd_mode, 'sparse') || ...
                    strcmp(self.dmd_mode, 'no_dynamics_sparse') || ...
                    strcmp(self.dmd_mode, 'sparse_fast'),...
                    'Matrix structure can only be enforced if dmd_mode is sparse')
            else
                sparsity = logical([]); % i.e. the default
                return
            end
            
            % Set the initial sparsity structure for A and B
            % concatenated with each other
            sz = [self.original_sz(1), size(self.control_signal,1)];
            if strcmp(self.dmd_mode, 'no_dynamics_sparse')
                A0 = ones(sz(1), sz(1)); % Set the entire A matrix to 0
            else
                A0 = zeros(sz(1), sz(1));
            end
            B0 = zeros(sz);
            % Set the sparsity structure for the specific options
            if self.enforce_diagonal_sparse_B
                ind = self.control_signals_metadata{'sparse',:}{:};
                B0(:, ind) = ones(sz(1), length(ind)) - ...
                    diag( ones(length(ind), 1) );
            end
            if ~isempty(self.enforce_zero_entries)
                % Format for each cell:
                %   2x1 cell array: {from, to}
                %   i.e. {neuron ID or number, neuron or ctr_signal}
                % Note: if control signal, will be an entire row
                names = self.get_names();
                ctr_names = self.control_signals_metadata.Row;
                for i = 1:length(self.enforce_zero_entries)
                    xy = self.enforce_zero_entries{i};
                    if ischar(xy{1}) || iscell(xy{1})
                        xy{1} = self.name2ind(xy{1});
                    end
                    if isnumeric(xy{2})
                        warning('Assuming sparsity enforcement on neural connections')
                        A0(xy{1}, xy{2}) = 1;
                    elseif any(contains(names, xy{2}))
                        xy{2} = self.name2ind(xy{2});
                        A0(xy{1}, xy{2}) = 1;
                    elseif any(ismember(xy{2}, ctr_names))
                        xy{2} = self.control_signals_metadata{xy{2}, :}{:};
                        B0(xy{1}, xy{2}) = 1;
                    else
                        error('Could not parse enforce_zero_entries input')
                    end
                end
            end
            sparsity = logical([A0, B0]);
        end
        
        function set_simple_labels(self, label_dict, new_labels_key)
            if self.verbose
                disp('Setting simple labels')
            end
            if ~exist('label_dict','var')
                if length(self.state_labels_key) == 8
                    % Classic format
                    label_dict = containers.Map(...
                        {1,2,3,4,5,6,7,8},...
                        {1,1,2,3,4,4,4,5});
                    if self.verbose
                        disp('Original format: Classic Zimmer')
                    end
                elseif length(self.state_labels_key) == 6
                    % Prelet format
                    label_dict = containers.Map(...
                        {0,1,2,3,4,5},...
                        {1,5,2,3,4,5});
                    if self.verbose
                        disp('Original format: Quiescence')
                    end
                elseif length(self.state_labels_key) == 5
                    return
                else
                    error('Unknown state label format')
                end
            end
            if ~exist('new_labels_key','var')
                new_labels_key = ...
                    {'Simple Forward',...
                    'Dorsal Turn',...
                    'Ventral Turn',...
                    'Simple Reverse',...
                    'NOSTATE'};
            end
            if isempty(label_dict) && isempty(new_labels_key)
                % Just reset
                self.state_labels_key = self.state_labels_key_old;
                self.state_labels_ind = self.state_labels_ind_old;
                return
            else
                assert(...
                    length(unique(cell2mat(label_dict.values))) ==...
                    length(new_labels_key),...
                    'New behavioral indices must all have labels')
            end
            
            self.state_labels_ind_old = self.state_labels_ind;
            self.state_labels_key_old = self.state_labels_key;
            
            f = @(x) label_dict(x);
            self.state_labels_ind = arrayfun(f, self.state_labels_ind);
            self.state_labels_key = new_labels_key;
        end
        
        function build_model(self)
            % Builds the control signals and then solves the DMDc problem
            self.calc_all_control_signals();
            self.calc_AdaptiveDmdc();
            self.postprocess();
        end
    end
    
    methods % Adding control signals
        
        function reset_user_control(self)
            self.user_control_input = [];
            self.user_control_matrix = [];
            self.user_neuron_ablation = [];
        end
        
        function remove_all_control(self)
            % Removes ALL control signals and metadata, including what was
            % calculated in initialization
            self.custom_control_signal = [];
            self.control_signal = [];
            self.control_signals_metadata = [];
            self.L_global_modes = [];
            self.L_sparse_modes = [];
        end
        
        function add_manual_control_signal(self, ...
                neuron_ind, neuron_amps, ...
                signal_ind, signal_amps)
            % Adds a row to the control matrix (B) going from
            assert(max(neuron_ind)<self.dat_sz(1),...
                'Control target must be in the original data set')
            if isempty(find(neuron_ind, 1))
                warning('Attempt to add an unconnected controller; passing.')
                return
            end
            this_ctr_connectivity = ...
                zeros(size(self.dat_without_control, 1), 1);
            if isscalar(neuron_amps)
                neuron_amps = neuron_amps*ones(size(neuron_ind));
            end
            this_ctr_connectivity(neuron_ind) = neuron_amps;
            self.user_control_matrix = ...
                [self.user_control_matrix ...
                this_ctr_connectivity];
            if isempty(signal_ind)
                self.user_control_input = [self.user_control_input;...
                    signal_amps];
            else
                this_signal = zeros(1,self.total_sz(2));
                if isscalar(signal_amps)
                    signal_amps = signal_amps*ones(size(signal_ind));
                end
                this_signal(signal_ind) = signal_amps;
                self.user_control_input = [self.user_control_input;...
                    this_signal];
            end
        end
        
        function add_partial_original_control_signal(self,...
                signal_ind, custom_signal, signal_start, is_original_neuron)
            % Adds some of the current control signals to the user control
            % matrix
            self.user_control_reconstruction = [];
            num_neurons = size(self.dat_without_control,1);
            if ~exist('signal_ind','var')
                signal_ind = (num_neurons+1):self.total_sz(1);
                is_original_neuron = true;
            end
            if ~exist('custom_signal','var') || isempty(custom_signal)
                custom_signal = self.AdaptiveDmdc_obj.dat;
            elseif size(custom_signal,1) < max(signal_ind)
                % Then it doesn't include the original data, which is the
                % format we're looking for
                tmp = zeros(max(signal_ind),...
                    size(custom_signal,2));
                tmp(signal_ind,:) = custom_signal;
                custom_signal = tmp;
            end
            if ~exist('signal_start','var') || isempty(signal_start)
                assert(size(custom_signal,2) == self.total_sz(2),...
                    'Custom signal must be defined for the entire tspan')
            else
                tmp = zeros(size(custom_signal, 1), self.total_sz(2));
                signal_end = signal_start+size(custom_signal, 2)-1;
                tmp(:,signal_start:signal_end) = custom_signal;
                custom_signal = tmp;
            end
            if ~exist('is_original_neuron','var')
                is_original_neuron = (max(signal_ind)<num_neurons);
            end
            if ~is_original_neuron
                signal_ind = signal_ind + num_neurons;
            end
            assert(max(signal_ind) <= self.total_sz(1),...
                'Indices must be within the discovered control signal')
            
            A = self.AdaptiveDmdc_obj.A_original(1:num_neurons,:);
            for i = 1:length(signal_ind)
                this_ind = signal_ind(i);
                this_connectivity = abs(A(:,this_ind))>0;
                this_amp = A(:,this_ind);
                this_signal = custom_signal(this_ind,:);
                self.add_manual_control_signal(...
                    this_connectivity, this_amp, [], this_signal);
            end
            
        end
        
        function ablate_neuron(self, neuron_ind)
            % "Ablates" a neuron by setting all connections to 0
            self.user_neuron_ablation = neuron_ind;
        end
        
        function set_AdaptiveDmdc_controller(self)
            % These are set manually in other functions
            new_control_matrix = self.user_control_matrix;
            new_control_input = self.user_control_input;
            ablated_neurons = self.user_neuron_ablation;
            % Save original matrices data, if not already saved
            if isempty(self.A_old)
                self.A_old = self.AdaptiveDmdc_obj.A_separate;
            end
            if isempty(self.dat_old)
                self.dat_old = self.AdaptiveDmdc_obj.dat;
            end
            % Get new matrices and data
            num_real_neurons = size(self.dat_without_control,1);
            num_controllers = size(new_control_matrix,2);
            num_neurons_and_controllers = ...
                num_real_neurons + num_controllers;
            new_control_matrix = [new_control_matrix;...
                zeros(num_controllers)];
            A_new = ...
                [self.A_old(1:num_neurons_and_controllers,1:num_real_neurons), ...
                new_control_matrix];
            dat_new = ...
                [self.dat_without_control;...
                new_control_input];
            %             ctrb(self.A_old(1:num_real_neurons,1:num_real_neurons), ...
            %                 new_control_matrix(1:num_real_neurons,:));
            % Ablate neurons
            A_new(ablated_neurons, ablated_neurons) = 0;
            % Update the object properties
            self.AdaptiveDmdc_obj.A_separate = A_new;
            self.AdaptiveDmdc_obj.dat = dat_new;
        end
        
        function reset_AdaptiveDmdc_controller(self)
            % Update the object properties
            self.AdaptiveDmdc_obj.A_separate = self.A_old;
            self.AdaptiveDmdc_obj.dat = self.dat_old;
            
            self.A_old = [];
            self.dat_old = [];
        end
        
        function out = run_with_only_global_control(self, func)
            % Sets the model to only use global control, then performs the
            % function on the object.
            % Input:
            %   func - function that only accepts this object as an
            %       argument, e.g. a method of this function, or something
            %       that uses a property directly
            %
            % Output:
            %   out - the FIRST (if more than one) output of the function
            
            num_neurons = self.dat_sz(1);
            ctr_ind = (num_neurons+1):(self.total_sz(1)-num_neurons);
            is_original_neuron = false;
            self.add_partial_original_control_signal(ctr_ind,...
                [], [], is_original_neuron)
            self.set_AdaptiveDmdc_controller();
            
            out = func(self);
            
            self.reset_user_control();
            self.reset_AdaptiveDmdc_controller();
        end
        
        function out = run_with_only_sparse_control(self, func)
            % Sets the model to only use sparse control, then performs the
            % function on the object.
            % Input:
            %   func - function that only accepts this object as an
            %       argument, e.g. a method of this function, or something
            %       that uses a property directly
            %
            % Output:
            %   out - the FIRST (if more than one) output of the function
            
            ctr_ind = 1:self.dat_sz(1);
            is_original_neuron = false;
            self.add_partial_original_control_signal(ctr_ind,...
                [], [], is_original_neuron)
            self.set_AdaptiveDmdc_controller();
            
            out = func(self);
            
            self.reset_user_control();
            self.reset_AdaptiveDmdc_controller();
        end
        
        function mega_rev_binary = calc_mega_rev(self, binary_labels)
            mega_rev_ind = contains(self.state_labels_key, ...
                {'REV', 'DT', 'VT'});
            mega_rev_binary = logical(sum(...
                binary_labels(mega_rev_ind,:),1));
            % Can jump over 10 frames of FWD/SLOW behavior
            [rev_starts, rev_ends] = calc_contiguous_blocks(...
                mega_rev_binary, 0, 10);
            for i = 1:length(rev_starts)
                mega_rev_binary(rev_starts(i):rev_ends(i)) = ...
                    true;
            end
        end
        
    end
    
    methods % Interpreting control signals
        
        function [signals, signal_mat, mean_signal, all_signal_ind] = ...
                get_control_signal_during_label(self, ...
                which_label, num_preceding_frames)
            if ~exist('num_preceding_frames','var')
                num_preceding_frames = 0;
            end
            assert(ismember(which_label, self.state_labels_key),...
                'Invalid state label')
            
            % Get indices for this behavior
            which_label_num = ...
                find(strcmp(self.state_labels_key, which_label));
            transition_ind = diff(self.state_labels_ind==which_label_num);
            start_ind = max(...
                find(transition_ind==1) - num_preceding_frames + 1, 1);
            end_ind = find(transition_ind==-1);
            if self.state_labels_ind(end) == which_label_num
                end_ind = [end_ind length(transition_ind)];
            end
            if self.state_labels_ind(1) == which_label_num
                start_ind = [1 start_ind];
            end
            assert(length(start_ind)==length(end_ind))
            n = length(start_ind);
            
            % Get the control signals for these indices
            signals = cell(n, 1);
            ctr = self.dat_sz(1)+1;
            min_signal_length = 0;
            all_signal_ind = [];
            for i = 1:n
                these_ind = start_ind(i):end_ind(i);
                all_signal_ind = [all_signal_ind, these_ind]; %#ok<AGROW>
                signals{i} = self.dat_with_control(ctr:end,these_ind);
                if length(these_ind) < min_signal_length || i == 1
                    min_signal_length = length(these_ind);
                end
            end
            
            % Get the mean
            num_channels = self.total_sz - self.dat_sz;
            signal_mat = zeros(...
                num_channels(1), min_signal_length, length(signals));
            for i = 1:length(signals)
                signal_mat(:,:,i) = signals{i}(:, 1:min_signal_length);
            end
            mean_signal = mean(signal_mat, 3);
        end
        
        function control_direction = calc_control_direction(self,...
                which_ctr_modes, use_original)
            % Get control matrices and columns to project
            ad_obj = self.AdaptiveDmdc_obj;
            x_dat = 1:ad_obj.x_len;
            if use_original
                % Query intrinsic dynamics
                x_ctr = x_dat;
            else
                % Query control dynamics
                x_ctr = (ad_obj.x_len+1):self.total_sz(1);
            end
            dynamic_matrix = ad_obj.A_original(x_dat, x_ctr);
            % Reconstruct the attractor and project it into the same space
            control_direction = dynamic_matrix(:, which_ctr_modes);
        end
        
        function attractor_reconstruction = calc_attractor(self, ...
                which_label, use_only_global)
            % Get the dynamics and control matrices, and the control signal
            % Input:
            %   use_only_global (false) - will only the control signal from
            %           global modes to determine the attractor. Only
            %           implemented for ID global modes for now
            if ~exist('use_only_global','var')
                use_only_global = false;
            end
            
            ad_obj = self.AdaptiveDmdc_obj;
            x_dat = 1:ad_obj.x_len;
            x_ctr = (ad_obj.x_len+1):self.total_sz(1);
            %             x_ctr = ad_obj.x_len + ...
            %                 self.control_signals_metadata{which_label,:}{:};
            A = ad_obj.A_original(x_dat, x_dat);
            B = ad_obj.A_original(x_dat, x_ctr);
            if strcmp(which_label,'all')
                if use_only_global
                    error('use_only_global not defined for all modes')
                end
                u = self.dat_with_control(x_ctr, :);
            elseif ~use_only_global
                [u_cell, ~, ~, ~] = ...
                    self.get_control_signal_during_label(which_label);
                u = [];
                for j = 1:length(u_cell)
                    u = [u, u_cell{j}]; %#ok<AGROW>
                end
            elseif strcmp(self.global_signal_mode, 'ID')
                u = zeros(length(x_ctr), 1);
                % The input is just the index directly
                u(end-2) = find(strcmp(...
                    which_label, self.state_labels_key));
                u(end-1) = 1; % This channel is just constant
                u(end) = mean(self.dat_with_control(end,:)); % time signal
                error('Old code for determing controller... need to update')
            elseif strcmp(self.global_signal_mode, 'ID_binary')
                u = 1;
                % The input is just the index directly, but we need to
                % shrink the B matrix to just the relevant row
                ind = strcmp(which_label, self.state_labels_key);
                B = B(:, ind);
            else
                error('use_only_global only implemented for global_signal_mode=ID')
            end
            % Reconstruct the attractor and project it into the same space
            % Find x (fixed point) in:
            %   x = A x + B u
            attractor_reconstruction = ((eye(length(A))-A)\B)*u;
        end
        
        function [attractor_overlap, all_ctr_directions,...
                attractor_reconstruction] =...
                calc_attractor_overlap(self, use_dynamics)
            % Assign "roles" to the neurons by calculating the overlap
            % between their intrinsic dynamic kick and the direction of the
            % different attractors
            %   Note: particularly for similar attractors like FWD and
            %   SLOW, overlap scores will be very similar
            %
            % Input:
            %   use_dynamics (true) - whether to calculate the attractor
            %               dynamically or just use the centroid of that
            %               behavior
            %
            %   TO DO: how does the the location of the origin affect this?
            if ~exist('use_dynamics','var')
                use_dynamics = true;
            end
            which_neurons = 1:self.original_sz(1);
            use_original = true;
            
            % Get intrinsic control kicks (matrix)
            all_ctr_directions = self.calc_control_direction(...
                which_neurons, use_original);
            % Get attractor positions (mean vector for each label)
            %   Note: normalize the length of these vectors
            all_labels = self.state_labels_key;
            attractor_reconstruction = zeros(...
                length(which_neurons), length(all_labels));
            for i = 1:length(all_labels)
                if use_dynamics
                    this_attractor = mean(...
                        self.calc_attractor(all_labels{i}), 2);
                else
                    [~, ~, ~, this_ind] =...
                        self.get_control_signal_during_label(all_labels{i});
                    this_attractor = mean(self.dat_without_control(...
                        which_neurons, this_ind),2);
                end
                attractor_reconstruction(:,i) = ...
                    this_attractor/norm(this_attractor);
            end
            % Calculate overlap
            attractor_overlap = ...
                all_ctr_directions * attractor_reconstruction;
        end
        
        function [neuron_roles, neuron_names] = ...
                calc_neuron_roles_in_transition(self,...
                use_only_known_neurons, abs_tol, use_dynamics)
            % Assigns neuron roles based on direction of intrinsic dynamic
            % push overlap with various attractors
            %   By default, averages together FWD+SLOW and the reversals;
            %   ignores turning states because they don't seem to act as
            %   true attractors
            %   Also by default only returns known neuron names
            if ~exist('use_only_known_neurons','var') ...
                    || isempty(use_only_known_neurons)
                use_only_known_neurons = true;
            end
            if ~exist('abs_tol','var') || isempty(abs_tol)
                abs_tol = 0.15;
            end
            if ~exist('use_dynamics','var')
                use_dynamics = true;
            end
            
            % Get attractor overlap and normalize
            [attractor_overlap, all_ctr_directions] =...
                self.calc_attractor_overlap(use_dynamics);
            all_norms = zeros(size(all_ctr_directions,2),1);
            for i=1:size(all_ctr_directions,2)
                all_norms(i) = norm(all_ctr_directions(:,i));
            end
            attractor_overlap = attractor_overlap./all_norms;
            % Get names and sort if required
            neuron_names = self.get_names();
            if use_only_known_neurons
                neuron_ind = ~strcmp(neuron_names,'');
                neuron_names = neuron_names(neuron_ind);
            else
                neuron_ind = 1:self.original_sz(1);
            end
            neuron_ind = find(neuron_ind);
            % Simplify the behavioral categories
            simplified_labels_cell = {{'FWD','SLOW'},{'REVSUS','REV1','REV2'}};
            all_labels = self.state_labels_key;
            simplified_labels_ind = cell(size(simplified_labels_cell));
            simplified_overlap = zeros(...
                size(attractor_overlap,1), length(simplified_labels_cell));
            for i=1:length(simplified_labels_cell)
                simplified_labels_ind(i) = ...
                    { contains(all_labels,simplified_labels_cell{i}) };
                simplified_overlap(:,i) = mean(...
                    attractor_overlap(:,simplified_labels_ind{i}),2);
            end
            % Use a hard threshold and determine the neuron roles
            simplified_overlap(abs(simplified_overlap)<abs_tol) = 0;
            neuron_roles = cell(size(neuron_ind));
            for i = 1:length(neuron_ind)
                this_neuron = neuron_ind(i);
                this_kick = simplified_overlap(this_neuron,:);
                if this_kick(1)>0 && this_kick(2)>0
                    neuron_roles{i} = 'both';
                elseif this_kick(1)>0
                    neuron_roles{i} = 'simple FWD';
                elseif this_kick(2)>0
                    neuron_roles{i} = 'simple REVSUS';
                else
                    neuron_roles{i} = 'other';
                end
            end
        end
        
        function [neuron_roles, neuron_names] = ...
                calc_neuron_roles_in_global_modes(self,...
                use_only_known_neurons, class_tol, max_err_percent, ...
                to_include_turns,...
                use_derivs, combine_derivs)
            if ~exist('use_only_known_neurons','var') || ...
                    isempty(use_only_known_neurons)
                use_only_known_neurons = true;
            end
            if ~exist('class_tol','var') || isempty(class_tol)
                class_tol = 0.001;
            end
            if ~exist('max_err_percent','var') || isempty(max_err_percent)
                max_err_percent = 0;
            end
            if ~exist('to_include_turns', 'var')
                to_include_turns = false;
            end
            if ~exist('use_derivs', 'var')
                use_derivs = self.use_deriv;
            end
            if ~exist('combine_derivs', 'var')
                combine_derivs = self.use_deriv;
            end
            
            num_neurons = self.original_sz(1);
            if self.use_deriv && ~use_derivs
                num_neurons = num_neurons/2;
            end
            if any(ismember(self.control_signals_metadata.Row,'ID'))
                assert(~to_include_turns,...
                    'Turns not implemented for ID signal (only ID_binary)')
                ind = self.control_signals_metadata{'ID',:}{:};
                B_global = self.AdaptiveDmdc_obj.A_original(1:num_neurons,...
                    ...(2*num_neurons+1):end-2);
                    self.original_sz(1)+ind);
                % Narrow these down to which neurons are important for which behaviors
                %   Assume a single control signal (ID); ignore offset
                group1 = (B_global > class_tol);
                group2 = (B_global < -class_tol);
                all_groups = [group1, group2];
            elseif any(ismember(self.control_signals_metadata.Row,'ID_binary'))
                ind = self.control_signals_metadata{'ID_binary',:}{:};
                assert(~any(ismember(self.control_signals_metadata.Row,'constant')),...
                    'A constant control will give incorrect results')
                B_global = self.AdaptiveDmdc_obj.A_original(1:num_neurons,...
                    ...(2*num_neurons+1):end-2);
                    num_neurons+ind);
                if combine_derivs
                    num_neurons = num_neurons/2;
                    B_global = B_global(1:num_neurons,:) + ...
                        B_global((num_neurons+1):end,:);
                end
                fwd_ind = contains(self.state_labels_key,...
                    {'FWD','SLOW'});
                rev_ind = contains(self.state_labels_key,...
                    ...{'REVSUS','REV1','REV2'});
                    'REVSUS');
                if isempty(find(fwd_ind, 1))
                    fwd_ind = contains(self.state_labels_key,...
                        {'simple FWD'});
                    rev_ind = contains(self.state_labels_key,...
                        {'simple REVSUS'});
                end
                if isempty(find(fwd_ind, 1))
                    fwd_ind = contains(self.state_labels_key,...
                        {'Forward'});
                    rev_ind = contains(self.state_labels_key,...
                        {'Reversal'});
                end
                if isempty(find(fwd_ind, 1))
                    error('Could not find any foward indices...')
                end
                % Narrow these down to which neurons are important for which behaviors
                group1 = (sum(B_global(:,rev_ind),2) > class_tol);
                group2 = (sum(B_global(:,fwd_ind),2) > class_tol);
                all_groups = [group1, group2];
                % Also do turns, if checked
                if to_include_turns
                    dt_ind = contains(self.state_labels_key, 'DT');
                    vt_ind = contains(self.state_labels_key, 'VT');
                    if isempty(find(dt_ind, 1))
                        dt_ind = contains(self.state_labels_key,...
                            {'Dorsal turn'});
                        dt_ind = dt_ind(1:end-1);
                        vt_ind = contains(self.state_labels_key,...
                            {'Ventral turn'});
                        vt_ind = vt_ind(1:end-1);
                    end
                    if isempty(find(dt_ind, 1)) && isempty(find(vt_ind, 1))
                        assert('No turn indices found...')
                    end
                    group3 = (sum(B_global(:,dt_ind),2) > class_tol);
                    group4 = (sum(B_global(:,vt_ind),2) > class_tol);
                    all_groups = [all_groups, group3, group4];
                    % Make the turns dominate the group designation;
                    % otherwise there are too many groups
                    for i = 1:size(group3,1)
                        if group3(i) || group4(i)
                            all_groups(i, 1:2) = 0;
                        end
                    end
                end
            end
            % Get names and sort if required
            neuron_names = self.get_names();
            neuron_names = neuron_names(1:num_neurons);
            if use_only_known_neurons
                % Also get rid of ambiguous neurons, which will have long
                % (concatenated) names
                neuron_ind = logical( ...
                    (~strcmp(neuron_names,'')).*...
                    (cellfun(@(x)length(x)<6, neuron_names)));
                neuron_ind = neuron_ind(1:num_neurons);
                neuron_names = neuron_names(1:num_neurons);
                neuron_names = neuron_names(neuron_ind);
            else
                neuron_ind = 1:num_neurons;
            end
            neuron_ind = find(neuron_ind);
            % Treat high error neurons as a different category
            %   Error is the reconstruction error normalized by the neuron
            %   activity L2 norm
            if max_err_percent>0
                %                 approx_dat = self.AdaptiveDmdc_obj.calc_reconstruction_control();
                %                 all_err = self.calc_balanced_error(approx_dat);
                all_err = 1-self.calc_correlation_matrix(true);
                %                 all_err = ...
                %                     sum( (self.dat - approx_dat).^2,2 ) ./...
                %                     sum(self.dat.^2,2);
                group_error = (all_err>max_err_percent);
                if combine_derivs
                    group_error = (group_error(1:num_neurons) + ...
                        group_error((num_neurons+1):end))/2;
                end
            else
                group_error = zeros(size(group1));
            end
            % Cast into strings
            neuron_roles = cell(size(neuron_ind));
            for i = 1:length(neuron_ind)
                this_neuron = neuron_ind(i);
                if group_error(this_neuron)
                    neuron_roles{i} = 'high error';
                    continue
                end
                this_str = 'group';
                for iG = 1:size(all_groups,2)
                    if all_groups(this_neuron, iG)
                        this_str = [this_str ' ' num2str(iG)]; %#ok<AGROW>
                    end
                end
                if strcmp(this_str, 'group')
                    this_str = 'other';
                end
                neuron_roles{i} = this_str;
            end
        end
        
        function err = calc_balanced_error(self, approx_dat)
            % Calculates the error between the passed reconstruction and
            % the original data by emphasizing "events" i.e. outliers in
            % the error vector of each neuron
            err_mat = real(self.dat - approx_dat);
            
            %             x = 1:size(self.dat,1);
            TF_err = isoutlier(err_mat, 'movmedian', 500, 2);
            %             TF_dat = isoutlier(self.dat, 'movmedian', 500, 2);
            
            err = zeros(size(self.dat,1),1);
            for i = 1:length(err)
                err(i) = ...
                    ( sum(err_mat(i,TF_err(i,:)).^2,2) + ...
                    0.5*sum(err_mat(i,:).^2,2) ) / ...
                    (1 + 1.5*sum(self.dat(i,:).^2,2) );
            end
        end
        
        function [corr_diag, corr_mat] = calc_correlation_matrix(self,...
                return_derivatives, normalize_mode)
            % Calculates the correlation matrix between the data and the
            % reconstruction, returning the diagonal values first
            %
            % Input:
            %   return_derivatives (false) - if derivatives are included in
            %       the model, whether or not to return them
            %   normalize_mode ('None') - whether to normalize the
            %       correlations by e.g.:
            %       1) the maximum variance left in that
            %           neuron after optimal svd truncation ('SVD')
            %       2) the additive improvement over a correlation with a
            %           straight line (will be strictly less than 1)
            %           ('linear')
            
            if ~exist('return_derivatives','var')
                return_derivatives = false;
            end
            if ~exist('normalize_mode', 'var')
                normalize_mode = 'None';
            end
            dat_reconstruction = ...
                self.AdaptiveDmdc_obj.calc_reconstruction_control();
            % If using cross-validation, dat_reconstruction is smaller
            this_dat = self.dat(:, 1:size(dat_reconstruction,2));
            
            corr_mat = corrcoef([this_dat' dat_reconstruction']);
            
            n = self.dat_sz(1);
            if self.use_deriv && ~return_derivatives
                n = n/2;
                corr_diag = diag( corr_mat((2*n+1):(3*n),1:n) );
            else
                corr_diag = diag( corr_mat((n+1):end,1:n) );
            end
            
            if strcmp(normalize_mode, 'SVD')
                % Look at the ratio of signal variance to total variance in each neuron
                [~, dat_signal] = calc_snr(self.dat);
                max_variance = var(dat_signal-mean(dat_signal,2), 0, 2) ./...
                    var(self.dat-mean(self.dat,2), 0, 2);
                corr_diag = corr_diag./sqrt(max_variance);
            elseif strcmp(normalize_mode, 'linear')
                % Look at the difference of correlation explained via
                % reconstruction and via a straight line fit
                corr_diag = corr_diag - self.calc_linear_correlation();
            end
        end
        
        function linear_corr = calc_linear_correlation(self)
            % Calculates the correlation between the data (per neuron) and
            % a linear fit, to be used as a baseline
            assert(~self.use_deriv, ('Does not work with derivatives'))
            n = self.dat_sz(1);
            linear_X = (1:self.dat_sz(2))';
            linear_X = [ones(size(linear_X)), linear_X];
            linear_reconstruction = linear_X*(linear_X\(self.dat'));
            linear_corr = corrcoef([self.dat' linear_reconstruction]);
            linear_corr = diag( linear_corr((n+1):end,1:n) );
        end
        
        function names = get_names(self, neuron_ind, keep_nums)
            % Calls sub-object AdaptiveDmdc method of the same name in most
            % cases, but also works if that object has not been
            % initialized
            %   Note that this still uses a static function from the
            %   AdaptiveDmdc class
            %
            % Input:
            %   neuron_ind ([]) - an integer or string refering to any
            %       number of neurons; the default means all neurons
            %   keep_nums (false) - to return numbers when no name is found
            if ~exist('neuron_ind', 'var') || isempty(neuron_ind)
                neuron_ind = 1:self.original_sz(1);
            end
            if ~exist('keep_nums', 'var')
                keep_nums = false;
            end
            if ischar(neuron_ind)
                warning('Passed a character already; returning')
                names = neuron_ind;
                return
            end
            try
                names = self.AdaptiveDmdc_obj.get_names(neuron_ind,...
                    true, false, false, true);
            catch
                % Call recursively if a list is input
                if ~isscalar(neuron_ind)
                    names = cell(size(neuron_ind));
                    for n=1:length(neuron_ind)
                        this_neuron = neuron_ind(n);
                        names{n} = self.get_names(this_neuron);
                    end
                    return
                end
                % Actually get the name
                id_struct = self.AdaptiveDmdc_settings.id_struct;
                if neuron_ind > length(id_struct.ID)
                    neuron_ind = neuron_ind - length(id_struct.ID);
                    if neuron_ind > length(id_struct.ID)
                        names = '';
                        return
                    end
                end
                names = {id_struct.ID{neuron_ind},...
                    id_struct.ID2{neuron_ind},...
                    id_struct.ID3{neuron_ind}};
                names = AdaptiveDmdc.parse_names({names});
                if length(names)==1 % i.e. a single neuron
                    names = names{1};
                end
            end % try
            
            if keep_nums
                if iscell(names)
                    for i = 1:length(names)
                        if isempty(names{i})
                            names{i} = num2str(neuron_ind(i));
                        end
                    end
                elseif isempty(names)
                    names = num2str(neuron_ind);
                end
            end
            
            if self.augment_data > 1
                if ischar(names)
                    names = {names};
                end
                names = repmat(names, [1, self.augment_data]);
            end
            
        end %function
        
        function ind = name2ind(self, neuron_names)
            % Finds the indices of a neuron or cell array of neurons
            %
            % 2 important notes:
            %   If the name is found in more than one neuron, all will be
            %   returned
            %   The order of the indices doesn't correspond to the order of
            %   the input names (if a cell array)
            if isnumeric(neuron_names)
                warning('Already passed a number; returning')
                ind = neuron_names;
                return;
            end
            if strcmp(neuron_names, 'all')
                ind = 1:self.dat_sz(1);
            else
                ind = find(contains(self.get_names(), neuron_names));
            end
            if self.augment_data > 0
                ind = ind(ind<self.original_sz(1));
            end
        end
        
        function calc_unrolled_matrix(self)
            % Unrolls a matrix with time delay embedding
            sz = self.dat_sz;
            aug = self.augment_data;
            n = self.original_sz(1);
            all_unroll = zeros(n, aug, n);
            A = self.AdaptiveDmdc_obj.A_separate(1:n,1:(aug*n));
            if aug <= 1
                self.A_unroll = A;
                return
            end
            
            for i = 1:self.original_sz(1)
                all_unroll(:,:,i) = real(reshape(A(i,:), [sz(1)/aug, aug]));
            end
            self.A_unroll = all_unroll;
        end
    end
    
    methods % Plotting
        
        function fig = plot_reconstruction_user_control(self, ...
                include_control_signal, neuron_ind)
            if ~exist('include_control_signal','var')
                include_control_signal = true;
            end
            if ~exist('neuron_ind','var')
                neuron_ind = 0;
            end
            % Uses manually set control signals
            self.set_AdaptiveDmdc_controller();
            
            % [With manual control matrices]
            [self.user_control_reconstruction, fig] = ...
                self.AdaptiveDmdc_obj.plot_reconstruction(true, ...
                include_control_signal, true, neuron_ind);
            title('Data reconstructed with user-defined control signal')
            
            % Reset AdaptiveDmdc object
            self.reset_AdaptiveDmdc_controller();
        end
        
        function fig = plot_reconstruction_interactive(self,...
                include_control_signal, neuron_ind)
            % Plots an interactive heatmap of the original data and the
            % reconstruction
            % Input (default in parentheses):
            %   include_control_signal (true) - whether to also plot the
            %       control signal; note that some control signals may have
            %       different max/min than the data and may plot well
            %   neuron_ind (0) - which neuron to plot; a value of 0 means
            %       the entire heatmap is plotted
            %
            % GUI interactivity:
            %   - Left click on a row to plot the trace of that neuron with
            %   its name, if identified (a new figure will pop up)
            %   - Right click to just show the name of the neuron
            %
            %   Note: You can use zoom without triggering the GUI
            if ~exist('include_control_signal','var')
                include_control_signal = true;
            else
                assert(islogical(include_control_signal),...
                    'First argument should be boolean')
            end
            if ~exist('neuron_ind','var')
                neuron_ind = 0;
            elseif ischar(neuron_ind)
                neuron_ind = self.name2ind(neuron_ind);
                assert(~isempty(neuron_ind),...
                    'No neuron of that name found')
                if self.augment_data>1
                    neuron_ind = neuron_ind(1);
                end
            end
            if length(neuron_ind)>1
                for i = 1:length(neuron_ind)
                    fig = self.plot_reconstruction_interactive(...
                        include_control_signal, neuron_ind(i));
                end
                return
            end
            
            % Plot
            [self.user_control_reconstruction, fig] = ...
                self.AdaptiveDmdc_obj.plot_reconstruction(true, ...
                include_control_signal, true, neuron_ind);
            %             title('Data reconstructed with user-defined control signal')
            
            % If single neuron and control signal, then do a 2-panel
            % subplot
            if include_control_signal && neuron_ind>0
                ax1 = fig.Children(2);
                subplot(2,1,1,ax1);
                subplot(2,1,2);
                ax2 = fig.Children(1);
                if any(contains(self.control_signals_metadata.Row,'sparse'))
                    % For RPCA-style controllers
                    plot(self.control_signal(neuron_ind,:),...
                        'LineWidth',2)
                    title('Sparse control signal')
                else
                    % Show the control signal that has the highest input
                    % into this neuron
                    n = self.dat_sz(1);
                    this_B = self.AdaptiveDmdc_obj.A_separate(...
                        neuron_ind, n+1:end);
                    [~, ctr_ind] = max(abs(this_B));
                    plot(self.control_signal(ctr_ind,:),...
                        'LineWidth',2)
                    title(sprintf('Strongest control signal: %d', ctr_ind))
                end
                xlim([0,self.original_sz(2)])
                linkaxes([ax1, ax2], 'x');
            end
            
            % Setup interactivity
            if neuron_ind == 0
                [~, im1, ~, im2] = fig.Children.Children;
                im1.ButtonDownFcn = @(x,y) self.callback_plotter_reconstruction(x,y);
                im2.ButtonDownFcn = @(x,y) self.callback_plotter_reconstruction(x,y);
            end
            
        end
        
        function fig = plot_colored_data(self, plot_pca, plot_opt, cmap)
            if ~exist('cmap', 'var')
                cmap = self.cmap;
            end
            if ~exist('plot_pca','var')
                plot_pca = false;
            end
            if ~exist('plot_opt','var')
                plot_opt = 'plot';
            end
            [self.L_sparse_modes,~,~,proj3d] = plotSVD(self.L_sparse,...
                struct('PCA3d',plot_pca,'sigma',false));
            fig = plot_colored(proj3d,...
                self.state_labels_ind(end-size(proj3d,2)+1:end),...
                self.state_labels_key, plot_opt, cmap);
            title('Dynamics of the low-rank component (data)')
        end
        
        function fig = plot_colored_neuron(self, neuron, cmap, fig)
            if ~exist('cmap', 'var')
                cmap = [];
            end
            if ~exist('fig', 'var')
                fig = figure('DefaultAxesFontSize',12);
            end
            if ischar(neuron)
                neuron_name = neuron;
                neuron = self.name2ind(neuron);
                %                 neuron = neuron(1); % Do not get derivatives
            elseif isnumeric(neuron)
                neuron_name = self.get_names(neuron);
                if iscell(neuron_name)
                    neuron_name = neuron_name{1};
                end
            end
            
            if length(neuron) > 1
                for i = 1:length(neuron)
                    fig = self.plot_colored_neuron(neuron(i), cmap);
                end
                return
            end
            
            fig = plot_colored(self.dat_with_control(neuron, :),...
                self.state_labels_ind, self.state_labels_key, 'plot', ...
                cmap, fig);
            title(sprintf('Neuron ID: %s', neuron_name))
        end
        
        function fig = plot_colored_user_control(self, fig, use_same_fig)
            % Plots user control data on top of colored original dataset
            %   fig (see text) - Figure handle if plotting on top of one
            %       that exists; otherwise, makes new figure with data
            %   use_same_fig (false) - To plot the reconstruction on top of
            %       the original data; if so, the reconstruction is in
            %       black stars ('k*')
            assert(~isempty(self.user_control_reconstruction),...
                'No reconstructed data stored')
            if ~exist('use_same_fig','var')
                use_same_fig = true;
            end
            if use_same_fig
                plot_opt = '.';
            else
                plot_opt = 'o';
            end
            if ~exist('fig','var') || isempty(fig)
                fig = self.plot_colored_data(false, plot_opt);
            end
            
            modes_3d = self.L_sparse_modes(:,1:3);
            x = 1:size(modes_3d,1);
            proj_3d = (modes_3d.')*self.user_control_reconstruction(x,:);
            if use_same_fig
                plot3(proj_3d(1,:),proj_3d(2,:),proj_3d(3,:), 'k*')
            else
                fig = plot_colored(proj_3d,...
                    self.state_labels_ind(end-size(proj_3d,2)+1:end),...
                    self.state_labels_key);
                title('Dynamics of the low-rank component (reconstruction)')
            end
        end
        
        function fig = plot_colored_reconstruction(self, cmap, fig, ...
                use_same_fig)
            % Plots 3d colored reconstruction on top of colored original dataset
            % Input:
            %   cmap ([]) - Custom colormap; defaults to the current
            %       object's stored colormap
            %   fig (see text) - Figure handle if plotting on top of one
            %       that exists; otherwise, makes new figure with data
            %   use_same_fig (false) - To plot the reconstruction on top of
            %       the original data; if so, the reconstruction is in
            %       black stars ('k*')
            if ~exist('cmap', 'var')
                cmap = self.cmap;
            end
            if ~exist('use_same_fig','var')
                use_same_fig = false;
            end
            if use_same_fig
                plot_opt = '.';
            else
                plot_opt = 'o';
            end
            if ~exist('fig','var') || isempty(fig)
                fig = self.plot_colored_data(false, plot_opt);
            end
            if isempty(self.L_sparse_modes)
                self.L_sparse_modes = plotSVD(self.L_sparse,...
                    struct('PCA3d',false,'sigma',false));
            end
            
            this_dat = self.AdaptiveDmdc_obj.calc_reconstruction_control();
            modes_3d = self.L_sparse_modes(:,1:3);
            x = self.x_indices;
            proj_3d = (modes_3d(1:length(x),:).')*this_dat(x,:);
            if use_same_fig
                plot3(proj_3d(1,:),proj_3d(2,:),proj_3d(3,:), 'k*')
            else
                fig = plot_colored(real(proj_3d),...
                    self.state_labels_ind(end-size(proj_3d,2)+1:end),...
                    self.state_labels_key, 'plot', cmap);
                title('Dynamics of the low-rank component (reconstruction)')
            end
        end
        
        function fig = plot_colored_control_arrow(self, ...
                which_ctr_modes, arrow_base, arrow_length, fig,...
                use_original, arrow_mode, color)
            % Plots a control direction on top of colored original dataset
            % Input:
            %   which_ctr_modes - which neurons/channels to plot
            %   arrow_base (origin) - location of the arrow base
            %   arrow_length (1) - factor multiplying the arrow length
            %   fig (self.plot_colored_data) - plot on a user's figure
            %   use_original (false) - to plot the control signal from the
            %               original neuron (intrinsic dynamics, A matrix)
            %               or the controller (B matrix)
            %   arrow_mode ('separate') - plotting mode if multiple arrows.
            %               Other options:
            %               'mean' = average the arrows
            %               'mean_and_difference' = average the arrows and
            %               plot the difference between that and the
            %               'arrow_base'
            %   color ('k') - color of the arrow
            if ~exist('arrow_base','var') || isempty(arrow_base)
                arrow_base = [0, 0, 0];
            end
            if ~exist('arrow_length','var') || isempty(arrow_length)
                arrow_length = 1;
            end
            if ~exist('fig','var') || isempty(fig)
                fig = self.plot_colored_data(false, 'o');
                [~, b] = fig.Children.Children;
                alpha(b, 0.2)
            end
            if ~exist('use_original', 'var')
                use_original = false;
            end
            if ~exist('arrow_mode','var')
                arrow_mode = 'separate';
            end
            if ~exist('color', 'var')
                color = 'k';
            end
            
            if isscalar(arrow_length)
                arrow_length = arrow_length*ones(size(which_ctr_modes));
            end
            if length(color)==1
                tmp = cell(size(which_ctr_modes));
                for i=1:length(tmp)
                    tmp{i} = color;
                end
                color = tmp;
            end
            
            % One vector per ctr_mode
            control_direction = ...
                self.calc_control_direction(which_ctr_modes, use_original);
            
            modes_3d = self.L_sparse_modes(:,1:3);
            proj_3d = (modes_3d.')*control_direction;
            switch arrow_mode
                case 'separate'
                    % show separate arrows
                    
                case 'mean'
                    proj_3d = sum(arrow_length.*proj_3d,2);
                    arrow_length = ones(size(arrow_length));
                    
                case 'mean_and_difference'
                    proj_3d = sum(arrow_length.*proj_3d,2);
                    proj_3d = proj_3d - arrow_base;
                    %                     arrow_length = 20*ones(size(arrow_length));
                    arrow_length = ones(size(arrow_length));
                    
                otherwise
                    error('Unrecognized arrow_mode')
            end
            
            for j=1:size(proj_3d,2)
                quiver3(arrow_base(1),arrow_base(2),arrow_base(3),...
                    proj_3d(1,j),proj_3d(2,j),proj_3d(3,j), ...
                    arrow_length(j), color{j}, 'LineWidth', 2)
            end
        end
        
        function fig = plot_colored_arrow_movie(self,...
                attractor_view, use_generated_data, movie_filename,...
                intrinsic_arrow_mode, show_reconstruction,...
                arrow_factor,...
                end_frame, pause_time, beginning_pause)
            % Plots a movie of the 3 different control signal directions:
            %    Intrinsic dynamics
            %    Sparse controller
            %    Global mode
            % Input:
            %   attractor_view (true) - show the 'global mode' control
            %       signal as an attractor
            %   use_generated_data (false) - to use generated data (default
            %       is to use the original data)
            %   movie_filename ('') - if not empty, then saves a movie
            %   intrinsic_arrow_mode ('mean_and_difference') - mode for
            %       displaying the intrinsic (blue) kick; an alternative is
            %       just 'mean' which has the base of the arrow at the
            %       origin instead of the current data point
            %   show_reconstruction (true) - to show the reconstructed data
            %       point as it travels or not
            %   arrow_factor (1) - lengthen all arrows
            %   end_frame (self.original_sz(2)) - how many frames to show
            %   pause_time (0) - pause time between time steps
            %   beginning_pause (false) - pause at the beginning (i.e. to
            %       change plot settings/zoom/etc)?
            %
            % Summary of algorithm:
            %   Get the current neuron activities (***_length)
            %   Get the relevant matrix entries (A or B)
            %   Multiply to get the next time step, e.g. x_{k+1}=A*x_k
            %   Plot this arrow, using the current point as a base
            if ~exist('attractor_view','var') || isempty(attractor_view)
                attractor_view = true;
            end
            if ~exist('use_generated_data','var') || isempty(use_generated_data)
                use_generated_data = false;
            end
            if ~exist('movie_filename','var') || isempty(movie_filename)
                movie_filename = '';
            end
            if ~exist('intrinsic_arrow_mode', 'var') || isempty(intrinsic_arrow_mode)
                intrinsic_arrow_mode = 'mean_and_difference';
            end
            if ~exist('show_reconstruction','var') || isempty(show_reconstruction)
                show_reconstruction = true;
            end
            if ~exist('arrow_factor', 'var')
                arrow_factor = 1;
            end
            if ~exist('end_frame','var')
                end_frame = self.original_sz(2);
            end
            if ~exist('pause_time','var')
                pause_time = 0.00;
            end
            if ~exist('beginning_pause','var')
                beginning_pause = false;
            end
            % Set up the background data figure
            fig = self.plot_colored_data(false, 'o');
            [~, b] = fig.Children.Children;
            alpha(b, 0.2)
            legend off;
            
            if ~isempty(movie_filename)
                video_obj = VideoWriter(movie_filename);
                open(video_obj);
            end
            
            num_neurons = self.original_sz(1);
            neuron_ind = 1:num_neurons;
            % Get the indices for the global and sparse modes, if any
            ctr_name = self.global_signal_mode;
            if contains(ctr_name, '_and_')
                ind = strfind(ctr_name,'_and_');
                ctr_name = ctr_name(1:(ind-1));
            end
            global_ind = num_neurons + ...
                self.control_signals_metadata{ctr_name,:}{:};
            %             global_ind = self.control_signals_metadata{ctr_name,:}{:};
            %             sparse_ctr_id = ...
            %                 contains(self.control_signals_metadata.Row,'sparse');
            %             if any(sparse_ctr_id)
            %                 use_sparse = true;
            %                 sparse_ind = ...
            %                     self.control_signals_metadata{sparse_ctr_id,:}{:};
            %             else
            %                 use_sparse = false;
            %             end
            % Different type, for onset signals
            if contains(ctr_name, 'transitions')
                sparse_ind = global_ind - num_neurons;
                use_sparse = true;
                global_ind = [];
            end
            this_dat = self.dat_without_control;
            %             if self.to_subtract_mean
            this_dat = this_dat - mean(self.dat_without_control,2);
            %             end
            % Get the control signals associated with the types of
            % controllers
            if use_sparse
                this_ctr_sparse = self.control_signal(sparse_ind,:);
            else
                this_ctr_sparse = [];
            end
            this_ctr_global = self.dat_with_control(global_ind,:);
            
            modes_3d = self.L_sparse_modes(:,1:3).';
            if show_reconstruction
                this_reconstruction = ...
                    self.AdaptiveDmdc_obj.calc_reconstruction_control(...
                    [], [], [], true);
                
                this_reconstruction = this_reconstruction(neuron_ind,:);
                all_arrow_bases = modes_3d * this_reconstruction;
                all_dat_projected = modes_3d * this_dat;
                % Fix axis limits, assuming the intrinsic arrows are the
                % largest
                control_direction = ...
                    self.calc_control_direction(neuron_ind, true);
                %                 modes_3d = self.L_sparse_modes(:,1:3);
                proj_3d = modes_3d * control_direction;
                arrow_length_intrinsic = (this_reconstruction*arrow_factor).';
                
                proj_3d = arrow_length_intrinsic*(proj_3d)';
                self.expand_axes(fig, proj_3d);
            else
                all_arrow_bases = modes_3d * this_dat;
            end
            
            % Loop through and possibly save a movie, but first pause so
            % that the user can maximize/etc
            if beginning_pause
                disp('Change the figure settings as desired')
                pause
            end
            for i = 1:end_frame
                arrow_base = all_arrow_bases(:,i);
                % Intrinsic
                if show_reconstruction
                    arrow_length_intrinsic = ...
                        (this_reconstruction(:,i)*arrow_factor).';
                else
                    arrow_length_intrinsic = (this_dat(:,i)*arrow_factor).';
                end
                self.plot_colored_control_arrow(...
                    neuron_ind, arrow_base, arrow_length_intrinsic, fig,...
                    true, intrinsic_arrow_mode, 'b');
                % Sparse controller
                if use_sparse
                    arrow_length_sparse = (this_ctr_sparse(:,i)*arrow_factor).';
                    %                     self.plot_colored_control_arrow(...
                    %                         neuron_ind, arrow_base, 20*arrow_length_sparse, fig,...
                    %                         false, 'mean', 'r');
                    self.plot_colored_control_arrow(...
                        sparse_ind, arrow_base, 20*arrow_length_sparse, fig,...
                        false, 'mean', 'r');
                end
                % Global controller
                if attractor_view
                    if strcmp(self.global_signal_mode, 'ID')
                        label_ind = this_ctr_global(1,i);
                    elseif contains(self.global_signal_mode, 'ID_binary')
                        %                         label_ind = find(this_ctr_global(:,i));
                        label_ind = self.state_labels_ind(i);
                    end
                    label_str = self.state_labels_key{label_ind};
                    self.plot_colored_fixed_point(label_str, true, fig);
                elseif ~isempty(this_ctr_global)
                    arrow_length_global = (this_ctr_global(:,i)*arrow_factor).';
                    self.plot_colored_control_arrow(...
                        global_ind-num_neurons, arrow_base, ...
                        arrow_length_global, fig,...
                        false, intrinsic_arrow_mode, 'k');
                else
                    % Plot a dummy point to make the deletion later work
                    plot3(0,0,0, 'wo')
                end
                % Current point
                plot3(arrow_base(1), arrow_base(2), arrow_base(3),...
                    'ok', 'LineWidth', 5)
                if show_reconstruction
                    % Then the above command plots the reconstruction, and
                    % we need to add the original point
                    plot3(all_dat_projected(1,i),...
                        all_dat_projected(2,i),...
                        all_dat_projected(3,i),...
                        'k*', 'LineWidth',5)
                end
                
                if ~isempty(movie_filename)
                    frame = getframe(fig);
                    writeVideo(video_obj, frame);
                end
                pause(pause_time)
                
                % Prep the figure for the next loop
                c = fig.Children.Children;
                num_to_delete = 3;
                if show_reconstruction
                    num_to_delete = num_to_delete + 1;
                end
                if use_sparse
                    num_to_delete = num_to_delete + 1;
                end
                delete(c(1:num_to_delete));
            end
            
            if ~isempty(movie_filename)
                close(video_obj);
            end
        end
        
        function plot_mean_transition_signals(self, ...
                which_label, num_preceding_frames)
            % Uses hand-labeled behavioral states, and plots the activity
            % that is happening right before onset
            % Input:
            %   which_label - which behavioral label to focus on
            %   num_preceding_frames - how many frames preceding the onset
            %       of that behavior to plot
            [~, signal_mat, mean_signal] = ...
                self.get_control_signal_during_label(...
                which_label, num_preceding_frames);
            
            title_str1 = sprintf(...
                'Control signals for label %s; %d frames preceding',...
                which_label, num_preceding_frames);
            title_str2 = sprintf(...
                'Standard deviation for label %s; %d frames preceding',...
                which_label, num_preceding_frames);
            plot_2imagesc_colorbar(...
                mean_signal, std(signal_mat, [], 3), '1 2',...
                title_str1, title_str2);
        end
        
        function fig = plot_eigenvalues_and_frequencies(self, ...
                neuron, tol_eigen)
            % Plots two scatter plots:
            %   - The real and imaginary parts of the eigenvalues, colored
            %   by the weighting of the eigenvector on the chosen neuron
            %   - A power spectrum of frequencies, weighted by the real
            %   part to the power of the period (i.e. how much the next
            %   loop around has decayed/grown); colored in the same way
            %
            %  The eigenvalues have a callback function which plots the
            %  eigenvector (heatmap) with the neuron names as ticks
            %
            % Input:
            %   neuron - string or index of the neuron to emphasize. All
            %       eigenvalues will be plotted, with the colormap
            %       determined by the eigenvector loading on this neuron.
            %   tol_eigen - minimum magnitude of the imaginary part of an
            %       eigenvalue. If smaller than this, then the eigenvalue
            %       corresponds to a nearly constant (non-dynamic) mode.
            if ~exist('tol_eigen','var')
                tol_eigen = 1e-3;
            end
            if exist('neuron', 'var')
                if ischar(neuron)
                    neuron = self.name2ind(neuron);
                end
            else
                neuron = [];
            end
            
            % Get eigenvectors and eigenvalues of intrinsic dynamics
            n = self.original_sz(1);
            %             if self.use_deriv
            %                 n = n/2;
            %             end
            A = self.AdaptiveDmdc_obj.A_separate(1:n,1:n);
            [V, D] = eig(A, 'vector');
            
            %             neuron_colormap = sum(real(V(neuron,:)).*abs(V(neuron,:)),1);
            if ~isempty(neuron)
                neuron_colormap = abs(V(neuron,:));
            else
                neuron_colormap = ones(1, size(V,2));
            end
            %             neuron_ind = (abs(neuron_colormap)>tol_eigen);
            
            fig = figure;
            subplot(2,1,1)
            polarplot(D, 'o')
            %             scatter(real(D(neuron_ind)),imag(D(neuron_ind)),...
            %                 [],neuron_colormap(neuron_ind),'filled')
            %             title(sprintf('Colored by neuron %d loading',neuron))
            %             colorbar
            %             xlabel('Real part (growth/decay)')
            %             ylabel('Imaginary part (oscillation)')
            
            subplot(2,1,2)
            %             all_periods = abs(2*pi./angle(D(neuron_ind)));
            all_periods = abs(2*pi./angle(D));
            p_max = 2*self.original_sz(2);
            all_periods(all_periods>p_max) = 0;
            %             all_factors = abs(D(neuron_ind));
            all_factors = abs(D);
            for i = 1:length(all_periods)
                if all_periods(i)~=0
                    all_factors(i) = all_factors(i)^all_periods(i);
                end
            end
            %             nonzero_ind = logical( (all_factors>tol_eigen).*...
            %                 abs(all_periods)>tol_eigen );
            nonzero_ind = logical(all_factors>tol_eigen);
            scatter(all_periods(nonzero_ind),all_factors(nonzero_ind),...
                [],neuron_colormap(nonzero_ind),'filled')
            colorbar
            title('Period of 0 means non-oscillatory decay')
            xlabel('Period (frames)')
            ylabel('Amplitude after one period')
            
            % Setup interactivity
            %             [~, ~, ~, im2] = fig.Children.Children;
            %             im1.ButtonDownFcn = @(x,y) self.callback_plotter_eigenvector(x,y,V,D);
            %             im2.ButtonDownFcn = @(x,y) self.callback_plotter_eigenvector(x,y,V,D);
            
        end
        
        function fig = plot_initial_condition(self,...
                x0, u, tspan)
            % Plots a heatmap of the system evolving from an initial
            % condition
            % Input:
            %   x0 - initial point
            %   u - control signal
            %   tspan - vector of time points (not just end)
            
            n = self.original_sz(1);
            dat_approx = zeros(n, length(tspan));
            dat_approx(:,1) = x0;
            for i = 2:length(tspan)
                dat_approx = ...
                    self.AdaptiveDmdc_obj.calc_reconstruction_manual(...
                    dat_approx(:,i-1), u(:,i));
            end
            
            % Plot a heatmap
            error('Not yet implemented')
            
        end
        
        function fig = plot_matrix_A(self, ...
                named_only, zero_diag, only_neurons, only_deriv,...
                num_clusters)
            % Plot a heatmap of the dynamic matrix
            % Input:
            %   named_only (false) - only plot named neurons
            %   zero_diag (false) - set the diagonal entries to 0; often
            %       useful because they may be an order of magnitude larger
            %       than other entries
            %   only_neurons (true) - plots only neurons, not derivatives,
            %       if included. If not included, does nothing
            %   only_derivs (false) - plots only derivatives, if present
            %   num_clusters (0) - if >0, clusters the data and sorts the
            %       matrix according to the k-means cluster identities
            if ~exist('named_only', 'var')
                named_only = false;
            end
            if ~exist('zero_diag', 'var')
                zero_diag = false;
            end
            if ~exist('only_neurons', 'var')
                only_neurons = true;
            end
            if ~exist('only_deriv', 'var')
                only_deriv = false;
            end
            if ~exist('num_clusters', 'var')
                num_clusters = 0;
            end
            
            n = (self.original_sz(1)*max([self.augment_data,1]));
            ind = 1:n;
            if self.use_deriv
                n = n/2;
                if only_neurons
                    ind = 1:n;
                elseif only_deriv
                    ind = (n+1):(2*n);
                end
            end
            
            % Get the names and maybe a subset
            names = self.get_names();
            if named_only
                ind(cellfun(@isempty,names(ind))) = [];
            end
            
            A = self.AdaptiveDmdc_obj.A_separate(ind,ind);
            if zero_diag
                A(logical(eye(length(A)))) = 0;
                title_str = 'Matrix with diagonals set to 0';
            else
                title_str = 'Matrix of dynamics';
            end
            
            % Rearrange the matrix and names by a dynamic clustering
            if num_clusters > 0
                c = clusterdata(A, 'linkage', 'ward','maxclust',num_clusters);
                tmp = zeros(size(A));
                tmp_names = cell(size(names));
                i_start = 1;
                names = names(ind);
                all_ind = 1:length(A);
                for i = 1:num_clusters
                    these_ind = (c==i);
                    tmp_ind = i_start:(i_start + length(find(these_ind)) - 1);
                    i_start = i_start + length(find(these_ind));
                    %                     tmp(tmp_ind, all_ind) = A(these_ind, all_ind);
                    %                     tmp(all_ind, tmp_ind) = A(all_ind, tmp_ind);
                    tmp(tmp_ind, tmp_ind) = A(these_ind, these_ind);
                    tmp_names(tmp_ind) = names(these_ind);
                end
                A = tmp;
                names = tmp_names;
            end
            
            % Actually plot
            fig = figure;
            imagesc(real(A));
            colormap(cmap_white_zero(real(A)));
            xticks(1:length(ind))
            xtickangle(90)
            yticks(1:length(ind))
            title(title_str)
            colorbar
            
            if num_clusters > 0
                xticklabels(names)
                yticklabels(names)
            else
                xticklabels(names(ind))
                yticklabels(names(ind))
            end
        end
        
        function fig = plot_matrix_B(self, ...
                named_only, which_controller, neuron_mode)
            % Plot a heatmap of the dynamic matrix
            % Input:
            %   named_only (false) - to plot only named (ID'ed) neurons
            %   which_controller (first one) - which control signal to
            %       plot; listed in self.control_signals_metadata.Row; e.g.
            %       'ID_binary' 'sparse'
            %   neuron_mode ('neurons_only') - to plot derivatives or not
            if ~exist('named_only', 'var')
                named_only = false;
            end
            if ~exist('which_controller', 'var') || isempty(which_controller)
                which_controller = self.control_signals_metadata.Row{1};
            else
                assert(ismember(...
                    which_controller,self.control_signals_metadata.Row),...
                    'Control signal not found')
            end
            if ~exist('neuron_mode', 'var')
                neuron_mode = 'neurons_only';
            end
            
            n = self.original_sz(1);
            if self.use_deriv
                if strcmp(neuron_mode,'neurons_only')
                    n = n/2;
                    row_ind = 1:n;
                elseif strcmp(neuron_mode,'derivs_only')
                    n = n/2;
                    row_ind = (n+1):(2*n);
                end
            else
                row_ind = 1:n;
            end
            % Get rows and columns
            col_ind = self.original_sz(1) + ...
                (self.control_signals_metadata{which_controller,:}{:});
            
            % Get the names and maybe a subset
            names = self.get_names();
            if named_only
                row_ind(cellfun(@isempty,names(row_ind))) = [];
            end
            
            B = self.AdaptiveDmdc_obj.A_separate(row_ind, col_ind);
            title_str = sprintf('Matrix of controllers (%s; %s)',...
                which_controller, neuron_mode);
            
            % Actually plot
            fig = figure;
            imagesc(real(B));
            xticks(1:length(row_ind))
            xtickangle(90)
            yticks(1:length(row_ind))
            title(title_str, 'Interpreter','None')
            colorbar
            
            if contains(which_controller, 'ID_binary')
                if contains(which_controller, '3_transitions')
                    ind = contains(self.state_labels_key, ...
                        {'REV', 'DT', 'VT'});
                    xticklabels(self.state_labels_key(ind))
                else
                    xticklabels(self.state_labels_key)
                end
            end
            yticklabels(names(row_ind))
        end
        
        function fig = plot_data_RPCA(self, which_mode)
            % Either plots the sparse signal or the robustified data
            
            fig = figure;
            if strcmp(which_mode,'L')
                imagesc(self.L_sparse)
            else
                imagesc(self.S_sparse, [min(min(self.S_sparse))*0.2,max(max(self.S_sparse))*0.2])
            end
            yticks(1:self.original_sz(1))
            yticklabels(self.get_names())
            colorbar
            
            % Set up interactivity
            [~, im1] = fig.Children.Children;
            im1.ButtonDownFcn = @(x,y) self.callback_plotter_reconstruction(x,y);
        end
        
        function fig = plot_correlation_histogram(self, ...
                only_named, neurons_to_mark, only_original,...
                normalize_mode, fig)
            % Plots a histogram of the correlation coefficients between the
            % original data (smoothed, if applicable) and the
            % reconstruction
            %
            % Input:
            %   only_named (false) - only plot named neurons
            %   neurons_to_mark ([]) - neurons to mark on the plot
            %   only_original (true) - if data are time-delay embedded,
            %       only plot the original ones
            %   normalize_mode ('SVD') - whether to normalize the
            %       correlations by e.g. the maximum variance left in that
            %       neuron after optimal svd truncation (option: 'SVD')
            %   fig ([]) - if not passed, will open a new figure
            if ~exist('only_named','var')
                only_named = false;
            end
            if ~exist('neurons_to_mark', 'var')
                neurons_to_mark = [];
            elseif iscell(neurons_to_mark) || ischar(neurons_to_mark)
                neurons_to_mark = self.name2ind(neurons_to_mark);
            end
            if ~exist('only_original', 'var') || isempty(only_original)
                only_original = true;
            end
            if ~exist('normalize_mode', 'var') || isempty(normalize_mode)
                normalize_mode = 'None';
            end
            if ~exist('fig', 'var')
                fig = figure('DefaultAxesFontSize', 14);
            end
            all_corr = real(...
                self.calc_correlation_matrix(false, normalize_mode));
            if only_original
                plot_ind = 1:self.original_sz(1);
                all_corr = all_corr(plot_ind);
                neurons_to_mark = ...
                    neurons_to_mark(neurons_to_mark<self.original_sz(1));
            else
                plot_ind = 1:length(all_corr);
            end
            
            if only_named
                names = self.get_names(plot_ind);
                ind = ~cellfun(@isempty, names);
                all_corr = all_corr(ind);
                title_str = 'Histogram of named neuron correlations';
                if ~isempty(neurons_to_mark)
                    neurons_to_mark = find(neurons_to_mark==find(ind));
                end
            else
                title_str = 'Histogram of all neuron correlations';
            end
            
            histogram(all_corr, 'BinWidth', 0.05);
            title(title_str)
            if strcmp(normalize_mode, 'SVD')
                xlabel('SVD Normalized correlation')
            else
                xlabel('Correlation')
            end
            hold on
            for i = 1:length(neurons_to_mark)
                x = all_corr(neurons_to_mark(i));
                y = ylim();
                y = round(0.8*y);
                line([x x], y, 'color', 'k');%, ...
                %'LineWidth',2, 'LineStyle','--')
                text(x, y(2), self.get_names(neurons_to_mark(i)),...
                    'FontSize',14, 'Rotation',90);
            end
        end
        
        function fig = plot_unrolled_matrix(self, which_neuron,...
                plot_mode, sort_mode, threshold, keep_nums, pos_mode)
            % Plots unrolled matrix with various visualizations.
            %
            % Input:
            %   which_neuron - the numerical index or name of the neuron to
            %       plot. If 'all' or more than one neuron, an average is
            %       taken over the set.
            %   plot_mode ('imagesc') - which plotting function to use.
            %       Other options include: 'waterfall', 'mean' (also
            %       displays +-1 std), 'gray' (with partially transparent
            %       lines)
            %   sort_mode ('none') - whether to sort the resulting matrix
            %   threshold (0) - below this value, rows are not displayed
            %   keep_nums (false) - keep numbers in the neuron name lists
            %   pos_mode ('both') - if 'pos' ('neg') only the positive
            %       (negative) weights in the matrix are shown
            if ischar(which_neuron)
                which_neuron = self.name2ind(which_neuron);
            end
            if ~exist('plot_mode', 'var') || isempty(plot_mode)
                plot_mode = 'imagesc';
            end
            if ~exist('sort_mode', 'var') || isempty(sort_mode)
                sort_mode = 'none';
            end
            if ~exist('threshold', 'var') || isempty(threshold)
                threshold = 0;
            end
            if ~exist('keep_nums', 'var') || isempty(keep_nums)
                keep_nums = false;
            end
            if ~exist('pos_mode', 'var')
                pos_mode = 'both';
            end
            if ~isfield(self, 'A_unroll')
                self.calc_unrolled_matrix();
            end
            
            if length(which_neuron) == 1
                this_dat = reshape(mean(self.A_unroll(which_neuron,:,:),1),...
                    [self.augment_data,self.original_sz(1)]);
                y_label_str = 'Actuating neuron';
            else
                % If we're displaying more than one neuron, take an average
                this_dat = reshape(mean(self.A_unroll,1),...
                    [self.augment_data,self.original_sz(1)]);
                y_label_str = 'Neuron being actuated';
            end
            if strcmp(pos_mode, 'pos')
                this_dat = this_dat.*(this_dat>0);
                title_str = 'Actuation plot for positive weights';
            elseif strcmp(pos_mode, 'neg')
                this_dat = this_dat.*(this_dat<0);
                title_str = 'Actuation plot for negative weights';
            else
                title_str = 'Actuation plot for all weights';
            end
            thresh_ind = find(sum(abs(this_dat),1) > threshold);
            this_dat = this_dat(:,thresh_ind);
            
            if isnumeric(sort_mode)
                if round(sort_mode)==sort_mode
                    [this_dat, sort_ind] = top_ind_then_sort(...
                        this_dat', sort_mode);
                else
                    [this_dat, sort_ind] = top_ind_then_sort(...
                        this_dat', [], sort_mode);
                end
                this_dat = this_dat';
                ind = thresh_ind(sort_ind);
            else
                ind = thresh_ind;
            end
            
            if strcmp(plot_mode, 'imagesc')
                fig = figure('DefaultAxesFontSize', 14);
                imagesc(this_dat')
                colormap(cmap_white_zero(this_dat));
                colorbar
            elseif strcmp(plot_mode, 'waterfall')
                fig = figure('DefaultAxesFontSize', 14);
                h = waterfall(this_dat');
                h.LineWidth = 3;
                colormap(cmap_white_zero(this_dat));
                colorbar
            elseif strcmp(plot_mode, 'mean')
                fig = plot_std_fill(this_dat, 2);
                y_label_str = 'Mean amplitude';
            elseif strcmp(plot_mode, 'gray')
                fig = plot_gray_lines(this_dat');
            end
            
            xlabel('Time delay')
            ylabel(y_label_str)
            yticks(1:length(ind))
            names = self.get_names([], keep_nums);
            yticklabels(names(ind))
            title(title_str)
            
        end
        
    end
    
    methods %(Access=private)
        
        function defaults = set_defaults(self)
            defaults = struct(...
                'verbose',true,...
                ...% Getting the control signal
                'designated_controller_channels', {{}},...
                'to_add_stimulus_signal', true,...
                'enforce_diagonal_sparse_B', false,...
                'global_signal_mode', 'ID_binary_transitions',...
                'global_signal_subset', {{'all'}},...{{'REV1', 'REV2', 'DT', 'VT'}},...
                'global_signal_pos_or_neg', 'only_pos',...
                'enforce_zero_entries', {{}},...
                ...% Data processing
                'filter_window_dat', 0,...
                'filter_window_global', 10,...
                'filter_aggressiveness', 0,...
                'autocorrelation_noise_threshold', 0,...
                'AdaptiveDmdc_settings', struct(),...
                'augment_data', 0,...
                'to_subtract_mean', false,...
                'to_subtract_baselines', false,...
                'to_subtract_mean_global', false,...
                'offset_control_signal', false,...
                'dmd_mode', 'func_DMDc',...
                ...% Data importing
                'use_deriv', false,...
                'use_only_deriv', false,...
                'to_normalize_deriv', false,...
                'to_save_raw_data', true,...
                ...% Additional rows
                'add_constant_signal', true,...
                ...% Plotting
                'cmap', []);
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
            
            self.A_old = [];
            self.dat_old = [];
            self.control_signals_metadata = table();
        end
        
        function import_from_struct(self, Zimmer_struct)
            warning('Assumes struct of Zimmer type')
            % First, get data and maybe derivatives
            self.raw = Zimmer_struct.traces.';
            if self.use_deriv || self.use_only_deriv
                self.raw_deriv = Zimmer_struct.tracesDif.';
            end
            % Save stimulus information
            if isfield(Zimmer_struct, 'stimulus')
                self.stimulus_struct = Zimmer_struct.stimulus;
            else
                self.stimulus_struct = struct();
            end
            % Get time information
            if isfield(Zimmer_struct, 'timeVectorSeconds')
                self.time_vec = Zimmer_struct.timeVectorSeconds;
            elseif isfield(Zimmer_struct, 'tv')
                self.time_vec = Zimmer_struct.tv;
            end
            if isfield(Zimmer_struct, 'fps')
                self.fps = Zimmer_struct.fps;
            else
                self.fps = 1/(self.time_vec(2) - self.time_vec(1));
            end
            % Get label vectors and names
            %   Struct has field names like "SevenStates" or "FiveStates"
            fnames = fieldnames(Zimmer_struct);
            state_fields = fnames(contains(fnames, 'States'));
            % These fields are either structs or the labels directly
            % 	If they are structs, add their fields to the original data
            % 	struct and parse as usual
            %   Note that there may be more than one labeling; take the
            %   last one if so
            if isstruct(Zimmer_struct.(state_fields{end}))
                labels_struct = Zimmer_struct.(state_fields{end});
                fnames_labels = fieldnames(labels_struct);
                assert(length(fnames_labels)==2,...
                    'Unknown fields in data labels struct')
                for this_fname = fnames_labels'
                    n = this_fname{1};
                    Zimmer_struct.(n) = labels_struct.(n);
                end
                state_fields = fnames_labels;
            end
            % Actually extract the labels (should be row vector)
            tmp_states1 = Zimmer_struct.(state_fields{1});
            tmp_states2 = Zimmer_struct.(state_fields{2});
            if ismatrix(tmp_states1) && iscell(tmp_states2)
                if size(tmp_states1,1)==1
                    self.state_labels_ind_raw = tmp_states1;
                else
                    self.state_labels_ind_raw = tmp_states1';
                end
                self.state_labels_key = tmp_states2;
            elseif ismatrix(tmp_states2) && iscell(tmp_states1)
                if size(tmp_states2,1)==1
                    self.state_labels_ind_raw = tmp_states2;
                else
                    self.state_labels_ind_raw = tmp_states2';
                end
                self.state_labels_key = tmp_states1;
            else
                error('Expected a cell array (legend) and vector of labels in %s and',...
                    ['Zimmer_struct.' (state_fields{1})],...
                    ['Zimmer_struct.' (state_fields{2})]);
            end
            % Add this information to the subobject
            fnames = fieldnames(self.AdaptiveDmdc_settings);
            if isempty(fnames)
                fnames = {''}; % Just to make the following logic work
            end
            if ~ismember(fnames,'id_struct')
                id_struct = struct('ID',{Zimmer_struct.ID},...
                    'ID2',{Zimmer_struct.ID2},'ID3',{Zimmer_struct.ID3});
                self.AdaptiveDmdc_settings.id_struct = id_struct;
            end
        end
        
        %Data processing
        function preprocess(self)
            if self.verbose
                disp('Preprocessing...')
            end
            if self.use_deriv || self.use_only_deriv
                self.raw = self.preprocess_deriv();
            end
            
            % Remove neurons if they are under the autocorrelation
            % threshold
            if self.autocorrelation_noise_threshold > 0
                assert(self.autocorrelation_noise_threshold < 1,...
                    'Autocorrelation cannot be greater than 1')
                n = size(self.raw,1);
                all_acf = zeros(n, 1);
                for i = 1:n
                    all_acf(i) = acf(self.raw(i,:)', 1, false);
                end
                to_remove = all_acf < self.autocorrelation_noise_threshold;
                self.remove_neurons_and_metadata(to_remove, 'raw');
            end
            
            %If augmenting, stack data offset by 1 column on top of itself;
            %note that this decreases the overall number of columns (time
            %slices)
            self.dat_sz = size(self.raw);
            aug = self.augment_data;
            self.original_sz = self.dat_sz;
            if self.to_subtract_mean
                for jM=1:self.dat_sz(1)
                    self.dat(jM,:) = self.raw(jM,:) - mean(self.raw(jM,:));
                end
            elseif self.to_subtract_baselines
                [self.dat, self.baselines] = self.calc_baseline(self.raw);
            else
                self.dat = self.raw;
            end
            
            if aug>1
                new_sz = [self.dat_sz(1)*aug, self.dat_sz(2)-aug];
                new_dat = zeros(new_sz);
                for j = 1:aug
                    old_cols = j:(new_sz(2)+j-1);
                    new_rows = (1:self.dat_sz(1)) + ...
                        self.dat_sz(1)*(aug-j);
                    new_dat(new_rows,:) = self.dat(:,old_cols);
                end
                self.dat_sz = new_sz;
                self.dat = new_dat;
                self.AdaptiveDmdc_settings.x_indices = 1:size(new_dat,1);
            end
            
            % Moving average filter
            if self.filter_window_dat > 1
                self.dat = ...
                    self.flat_filter(self.dat.',self.filter_window_dat).';
            end
            if self.filter_aggressiveness > 0
                self.dat = self.smart_filter(self.dat.',...
                    self.filter_aggressiveness).';
            end
            
            if ~isempty(self.state_labels_ind_raw)
                self.state_labels_ind = ...
                    self.state_labels_ind_raw(end-size(self.dat,2)+1:end);
            end
            
            % Set up the settings for AdaptiveDMDc object
            fnames = fieldnames(self.AdaptiveDmdc_settings);
            if isempty(fnames)
                fnames = {''};
            end
            
            if ~ismember(fnames,'sort_mode')
                self.AdaptiveDmdc_settings.sort_mode = 'user_set';
            end
            if ~ismember(fnames,'x_indices')
                if ~self.use_deriv
                    x_ind = 1:self.dat_sz(1);
                else
                    error('Check this')
                    x_ind = 1:(2*self.dat_sz(1));
                end
                self.AdaptiveDmdc_settings.x_indices = x_ind;
            end
            if ~ismember(fnames,'dmd_mode')
                self.AdaptiveDmdc_settings.dmd_mode = self.dmd_mode;
            end
            if ~ismember(fnames,'to_plot_nothing')
                self.AdaptiveDmdc_settings.to_plot_nothing = true;
            end
            if ~ismember(fnames,'to_save_raw_data')
                self.AdaptiveDmdc_settings.to_save_raw_data = false;
            end
            if ~ismember(fnames,'data_already_augmented')
                self.AdaptiveDmdc_settings.data_already_augmented = self.augment_data;
            end
        end
        
        function remove_neurons_and_metadata(self, to_remove, field)
            if strcmp(field, 'raw') || strcmp(field, 'dat')
                tmp = self.(field);
                n = size(tmp,1);
                to_keep = 1:n;
                to_keep(to_remove) = [];
                assert(~isempty(to_keep),...
                    'Something has gone wrong; attempting to remove all data...')
                self.(field) = tmp(to_keep,:);
            else
                error('Unrecognized field')
            end
            % Also remove the corresponding metadata
            z = self.AdaptiveDmdc_settings.id_struct;
            to_keep_full_sz = to_keep;
            if strcmp(field, 'dat') && max(to_keep) > self.original_sz(1)
                if self.augment_data==0
                    error('Unknown reason for indices past edge of data')
                end
                to_keep = to_keep(to_keep<=self.original_sz(1));
            end
            id_struct = struct('ID',{z.ID(to_keep)},...
                'ID2',{z.ID2(to_keep)}, 'ID3',{z.ID3(to_keep)});
            self.AdaptiveDmdc_settings.id_struct = id_struct;
            if isfield(self.AdaptiveDmdc_settings, 'x_indices')
                assert( isequal(self.AdaptiveDmdc_settings.x_indices,...
                    1:n), 'Custom x_indices not supported with neuron removal')
                self.AdaptiveDmdc_settings.x_indices = ...
                    1:length(find(to_keep_full_sz));
            end
            
            warning('Metadata removal only works before AdaptiveDmdc object is defined')
        end
        
        function new_raw = preprocess_deriv(self)
            % Aligns and optionally normalizes the derivative signal
            deriv = self.raw_deriv;
            if self.to_normalize_deriv
                deriv = deriv .* (std(self.raw,[],2) ./ std(deriv,[],2));
            end
            
            % Derivative might be one frame short, so throw out the last frame
            if ~self.use_only_deriv
                try
                    new_raw = [self.raw; deriv];
                catch
                    new_raw = [self.raw(:,1:end-1); deriv];
                end
            else
                new_raw = deriv;
            end
        end
        
        function calc_all_control_signals(self)
            % Calls subfunctions to calculate control signals:
            %   partition_data_into_controllers for direct neuron traces
            %   calc_sparse_signal for an RPCA-defined sparse signal
            %   calc_global_signal for an RPCA or otherwise defined global
            %       signal
            %   add_custom_control_signal for an externally passed signal
            %   add_stimulus_signal for importing a signal from a struct
            % and finally:
            %   calc_dat_and_control_signal for making sure everything is
            %       the same length and cleaning them up
            self.add_custom_control_signal();
            
            self.partition_data_into_controllers();
            self.calc_sparse_signal();
            self.calc_global_signal();
            if self.to_add_stimulus_signal % From an external struct
                self.add_stimulus_signal();
            end
            
            self.calc_dat_and_control_signal();
        end
        
        function partition_data_into_controllers(self)
            % Partitions data into control signals based on external
            % designations
            %   Uses "designated_controller_channels" field which is by
            %   default empty
            %
            % Format of designated_controller_channels field:
            %   {'sleepy',{'RIS'}}
            %   {'my_rev',{'AVA','RIM'}}
            %   {'my_fwd',[44 46]}
            if isempty(self.designated_controller_channels)
                return
            end
            assert(iscell(self.designated_controller_channels),...
                'Designated controllers must be input as a cell array')
            n = length(self.designated_controller_channels);
            assert(mod(n,2)==0,...
                'Designated controllers must be input as a name-index pair')
            
            all_ind = [];
            for i = 1:(n/2)
                this_name = self.designated_controller_channels{2*i-1};
                this_ind = self.designated_controller_channels{2*i};
                if ~isnumeric(this_ind)
                    this_ind = self.name2ind(this_ind);
                end
                this_ctr = self.dat(this_ind,:);
                
                self.add_custom_control_signal(this_ctr, this_name);
                
                all_ind = [all_ind, this_ind]; %#ok<AGROW>
                if self.augment_data > 0
                    % Also remove the time-delayed copies
                    all_ind = [all_ind, ...
                        this_ind+self.original_sz(1)*(1:self.augment_data-1)]; %#ok<AGROW>
                end
            end
            if ~isequal(sort(all_ind), unique(all_ind))
                warning(['Designated control signals have duplicated neurons' ...
                    ' (This will not affect performance, only interpretation)'])
            end
            
            self.remove_neurons_and_metadata(all_ind, 'dat');
            
            self.dat_sz = size(self.dat);
        end
        
        function calc_global_signal(self, global_signal_mode)
            % Calculates the global control signals using one of several
            % methods
            if ~exist('global_signal_mode','var')
                global_signal_mode = self.global_signal_mode;
            end
            
            % Note that multiple '_and_xxxx' are supported
            additional_signals = {'length_count', 'x_times_state',...
                'cumsum_x_times_state', 'cumtrapz_x_times_state',...
                'ID_binary'};
            for i = 1:length(additional_signals)
                this_str = ['_and_' additional_signals{i}];
                if contains(global_signal_mode, this_str)
                    self.calc_global_signal(...
                        erase(global_signal_mode, this_str));
                    self.calc_global_signal(additional_signals{i});
                    % Really do want to quit the entire function!
                    return
                end
            end
            
            this_metadata = table();
            
            switch global_signal_mode
                case 'ID'
                    self.L_global_modes = self.state_labels_ind.';
                    this_metadata.signal_indices = ...
                        {1:size(self.state_labels_ind,1)};
                    
                case 'ID_binary'
                    binary_labels = self.calc_binary_labels(...
                        self.state_labels_ind, length(self.state_labels_key));
                    if ~isequal(sum(binary_labels,1),...
                            ones(1,size(binary_labels,2)))
                        warning('Not all time frames have state labels')
                    end
                    % Also filter this so the gradients aren't so sharp
                    if self.filter_window_global>0
                        binary_labels = self.flat_filter(...
                            binary_labels.', self.filter_window_global).';
                    end
                    if ~isempty(self.L_global_modes)
                        self.L_global_modes = [self.L_global_modes...
                            binary_labels(:,1:size(self.L_global_modes,1)).'];
                    else
                        self.L_global_modes = binary_labels.';
                    end
                    this_metadata.signal_indices = ...
                        {1:size(binary_labels,1)};
                    
                case 'ID_binary_and_grad'
                    self.calc_global_signal('ID_binary');
                    binary_ind = self.control_signals_metadata{'ID_binary',:}{:};
                    binary_ind = binary_ind - binary_ind(1) + 1;
                    ID_binary_dat = self.L_global_modes(:, binary_ind);
                    tmp = gradient(ID_binary_dat')';
                    this_metadata.signal_indices = ...
                        {size(self.L_global_modes,2) + ...
                        (1:size(tmp,2))};
                    
                    self.L_global_modes = [self.L_global_modes, tmp];
                    
                case 'ID_binary_grad_only'
                    binary_labels = self.calc_binary_labels(...
                        self.state_labels_ind, length(self.state_labels_key));
                    % Also filter this so the gradients aren't so sharp
                    if self.filter_window_global>0
                        binary_labels = self.flat_filter(...
                            binary_labels.', self.filter_window_global).';
                    end
                    % Now take the gradient
                    binary_labels_grad = gradient(binary_labels);
                    binary_labels_grad = binary_labels_grad ./ ...
                        max(max(binary_labels_grad));
                    if ~isempty(self.L_global_modes)
                        self.L_global_modes = [self.L_global_modes...
                            binary_labels_grad(:,1:size(self.L_global_modes,1)).'];
                    else
                        self.L_global_modes = binary_labels_grad.';
                    end
                    
                    this_metadata.signal_indices = ...
                        {1:size(binary_labels_grad,1)};
                    
                case 'ID_binary_transitions'
                    % Only adds 'on' signals for the 3 main transitions by
                    %   default:
                    %   REV1/REV2, DT, and VT
                    % Alternate transitions can be specified in:
                    %   self.global_signal_subset
                    % Note: 'mega_rev' is a special composite signal that
                    %   only triggers on the first/last reversal of a
                    %   burst (positive/negative)
                    binary_labels = self.calc_binary_labels(...
                        self.state_labels_ind, length(self.state_labels_key));
                    % Also filter this so the gradients aren't so sharp
                    if self.filter_window_global>0
                        binary_labels = self.flat_filter(...
                            binary_labels.', self.filter_window_global).';
                    end
                    binary_labels_grad = gradient(binary_labels);
                    binary_labels_grad = binary_labels_grad ./ ...
                        max(max(binary_labels_grad));
                    if contains(lower(self.global_signal_subset),...
                            'mega_rev')
                        mega_rev_binary = self.calc_mega_rev(binary_labels);
                        if self.filter_window_global>0
                            mega_rev_binary = self.flat_filter(...
                                double(mega_rev_binary).',...
                                self.filter_window_global).';
                        end
                        binary_labels_grad = gradient(mega_rev_binary);
                        binary_labels_grad = binary_labels_grad ./ ...
                            max(max(binary_labels_grad));
                    elseif ~strcmp(self.global_signal_subset, 'all')
                        ind = contains(self.state_labels_key, ...
                            self.global_signal_subset);
                        %                         assert(~isempty(ind),...
                        %                             ['Attempted to add control signals '...
                        %                             'but none were found; check global_signal_subset'])
                        binary_labels_grad = binary_labels_grad(ind,:);
                    end
                    % Now only keep the positive parts, and only of the
                    % mentioned 3 transitions
                    %   Note: global_signal_pos_or_neg sets whether to keep
                    %   the negative part also or alone
                    only_pos = binary_labels_grad.*(binary_labels_grad>0);
                    only_neg = binary_labels_grad.*(binary_labels_grad<0);
                    switch self.global_signal_pos_or_neg
                        case 'only_pos'
                            only_neg = [];
                        case 'only_neg'
                            only_pos = [];
                        case 'pos_and_neg'
                            %
                        otherwise
                            error('Unrecognized global_signal_pos_or_neg')
                    end
                    binary_labels_grad = [only_pos; only_neg];
                    
                    if ~isempty(self.L_global_modes)
                        self.L_global_modes = [self.L_global_modes...
                            binary_labels_grad(:,1:size(self.L_global_modes,1)).'];
                    else
                        self.L_global_modes = binary_labels_grad.';
                    end
                    
                    this_metadata.signal_indices = ...
                        {1:size(binary_labels_grad,1)};
                    
                case 'ID_simple'
                    tmp = self.state_labels_ind;
                    states_dict = containers.Map(...
                        {1,2,3,4,5,6,7,8},...
                        {-1,-1,0,0,0,0,1,0});
                    for i=1:length(tmp)
                        tmp(i) = states_dict(tmp(i));
                    end
                    self.L_global_modes = tmp.';
                    this_metadata.signal_indices = {1:size(tmp,2)};
                    
                case 'ID_and_grad'
                    self.L_global_modes = [self.state_labels_ind;...
                        gradient(self.state_labels_ind)].';
                    
                    this_metadata.signal_indices = {1:2}; % Indices should be a single vector
                    
                    %                 case 'length_count' % Not called alone
                    %                     tmp = self.state_length_count(self.state_labels_ind);
                    %                     self.normalize_length_count = ...
                    %                         ( (max(max(self.dat))-min(min(self.dat)))...
                    %                         ./(max(max(tmp))-min(min(tmp))) );
                    %                     tmp = tmp * self.normalize_length_count;
                    % %                     this_metadata.signal_indices = ...
                    % %                         {size(self.L_global_modes,2) + ...
                    % %                         (1:size(tmp,1))};
                    %                     this_metadata.signal_indices = {1:size(tmp,1)};
                    %
                    %                     self.L_global_modes = [self.L_global_modes,...
                    %                         tmp'];
                    
                case 'x_times_state' % Not called alone
                    binary_labels = self.calc_binary_labels(...
                        self.state_labels_ind, length(self.state_labels_key));
                    tmp = [];
                    ind = 1:size(binary_labels,2);
                    for i=1:size(binary_labels,1)
                        tmp = [tmp; self.dat.*binary_labels(i,ind)]; %#ok<AGROW>
                    end
                    this_metadata.signal_indices = {1:size(tmp,1)};
                    self.L_global_modes = [self.L_global_modes, tmp.'];
                    
                case 'None'
                    % Only way to assign an empty value
                    %                     error('Not working')
                    this_metadata.signal_indices = {[]};
                    
                otherwise
                    error('Unrecognized method')
            end
            
            % Save metadata for this global signal
            %   Note that this is after the sparse signal, if any
            if ~isempty(this_metadata.signal_indices{:})
                this_metadata.Properties.RowNames = {global_signal_mode};
                self.append_control_metadata(this_metadata, true);
            end
            % Also add a row of ones
            if ~ismember('constant',self.control_signals_metadata.Row) &&...
                    self.add_constant_signal
                signal_indices = {1};
                ones_metadata = table(signal_indices);
                ones_metadata.Properties.RowNames = {'constant'};
                self.append_control_metadata(ones_metadata, true);
                
                sz = size(self.L_global_modes,1);
                if sz == 0
                    sz = size(self.dat,2);
                end
                self.L_global_modes = [self.L_global_modes, ones(sz,1)];
            end
        end
        
        function [this_lambda, this_rank] = check_rank_with_lambda(self,...
                this_dat, this_lambda, max_rank)
            %   Loop to check for rank convergence if
            iter_max = 10;
            for i=1:iter_max
                if max_rank==0
                    break;
                end
                [~, ~,...
                    this_rank, ~] = ...
                    RobustPCA(this_dat, this_lambda, 10*this_lambda,...
                    1e-6, 70);
                if this_rank <= max_rank
                    fprintf("Reached target rank (%d); restarting with more accuracy\n",...
                        max_rank);
                    break
                else
                    % A smaller penalty means more data in the sparse
                    % component, less in the low-rank
                    if self.verbose
                        fprintf("Didn't reach target rank (%d); decreasing lambda\n",...
                            max_rank);
                    end
                    if i==iter_max
                        warning("Didn't converge on maximum rank")
                    end
                    this_lambda = this_lambda*0.9;
                end
            end
        end
        
        function smooth_and_save_global_modes(self)
            % Uses fields already present in the class to smooth and save
            % the global modes
            self.L_global = self.flat_filter(...
                self.L_global_raw.', self.filter_window_global).';
            tmp_dat = self.L_global - mean(self.L_global,2);
            [u, ~] = svd(tmp_dat(:,self.filter_window_global+5:end).');
            x = 1:self.L_global_rank;
            self.L_global_modes = real(u(:,x));
            
            self.S_global = real(self.S_global_raw);
        end
        
        function calc_sparse_signal(self)
            %             if self.lambda_sparse > 0
            %                 % Calculates very sparse signal
            %                 [self.L_sparse_raw, self.S_sparse_raw,...
            %                     self.L_sparse_rank, self.S_sparse_nnz] = ...
            %                     RobustPCA(self.dat, self.lambda_sparse);
            %                 % Save the metadata for this signal, which is the first
            %                 this_metadata = table();
            %                 this_metadata.signal_indices = {1:size(self.L_sparse_raw,1)};
            %                 this_metadata.Properties.RowNames = {'sparse'};
            %                 self.append_control_metadata(this_metadata);
            %             else
            % i.e. just skip this step
            self.L_sparse_raw = self.dat;
            self.S_sparse_raw = [];
            self.L_sparse_rank = 0;
            self.S_sparse_nnz = 0;
            %             end
            % For now, no processing
            self.L_sparse = self.L_sparse_raw;
            self.S_sparse = self.S_sparse_raw;
        end
        
        function add_custom_control_signal(self, ...
                custom_control_signal, custom_control_signal_name)
            if ~exist('custom_control_signal','var')
                custom_control_signal = [];
                %                 custom_control_signal = self.custom_control_signal;
                %                 if ~isempty(self.custom_control_signal) && ...
                %                         self.filter_window_global ~= 10
                %                     warning('Control signal filtering is not applied to custom signals')
                %                 end
                %                 self.custom_control_signal = [];
                %                 custom_control_signal_name = 'user_custom_control_signal';
            end
            if isempty(custom_control_signal)
                if self.verbose
                    disp('No custom control signal added')
                end
                return
            end
            % Calculate the metadata
            this_metadata = table();
            this_metadata.signal_indices = ...
                {1:size(custom_control_signal,1)};
            this_metadata.Properties.RowNames = {custom_control_signal_name};
            
            % Save in self
            self.append_control_metadata(this_metadata, true);
            self.custom_control_signal = [self.custom_control_signal;
                custom_control_signal];
        end
        
        function add_stimulus_signal(self)
            % Adds environmental stimulus to the control signal, if any
            try
                s_times = self.stimulus_struct.switchtimes;
            catch
                return
            end
            if isempty(s_times)
                return
            else
                s_times = round(s_times*self.fps);
                assert(length(s_times)==2,...
                    'Can only import an external stimulus with 2 switchtimes')
                assert(strcmp(self.stimulus_struct.type,'binarysteps'),...
                    'Can only import an external stimulus with binary switches')
            end
            
            id_signal = self.stimulus_struct.identity;
            if self.verbose
                fprintf('Found external control signal (%s; %s)\n',...
                    id_signal, self.stimulus_struct.type)
            end
            
            ctr_signal = zeros(1, self.original_sz(2));
            ctr0 = self.stimulus_struct.initialstate;
            ctr_signal(1:s_times(1)) = ctr0;
            ctr_signal(s_times(1)+1:s_times(2)) = ctr0 + 1;
            ctr_signal(s_times(2)+1:end) = ctr0;
            
            ctr_signal_binary = self.calc_binary_labels(ctr_signal);
            self.add_custom_control_signal(ctr_signal_binary,...
                sprintf('Stimulus_%s',id_signal));
        end
        
        function calc_dat_and_control_signal(self)
            % Uses results from 2 different Robust PCA runs
            
            % Data to be reconstructed is everything EXCEPT the sparse
            % control signals
            this_dat = self.L_sparse;
            sparse_signal = self.S_sparse;
            %             if ~self.to_separate_sparse_from_data && self.lambda_sparse > 0
            %                 this_dat = this_dat + sparse_signal;
            %             end
            if self.to_subtract_mean_global
                L_low_rank = self.L_global_modes - mean(self.L_global_modes,1);
            else
                L_low_rank = self.L_global_modes;
            end
            % Create the augmented dataset (these might have different
            % amounts of filtering, therefore slightly different sizes)
            L_low_rank = L_low_rank';
            if ~isempty(sparse_signal) && ~isempty(L_low_rank)
                num_pts = min([size(L_low_rank,2) ...
                    size(sparse_signal,2) size(this_dat,2)]);
                self.control_signal = ...
                    [sparse_signal(:,1:num_pts);...
                    L_low_rank(:,1:num_pts)];
            elseif ~isempty(sparse_signal)
                num_pts = min([size(sparse_signal,2) size(this_dat,2)]);
                self.control_signal = sparse_signal(:,1:num_pts);
            elseif ~isempty(L_low_rank)
                num_pts = min([size(L_low_rank,2) size(this_dat,2)]);
                self.control_signal = L_low_rank(:,1:num_pts);
            else
                num_pts = size(this_dat,2);
                self.control_signal = [];
            end
            if ~isempty(self.custom_control_signal)
                self.control_signal = [self.control_signal;
                    self.custom_control_signal(:,1:num_pts)];
            end
            
            % TEST: offset the control signals by one step backwards...
            % this makes sense particularly for the sparse signal
            if self.offset_control_signal
                self.control_signal = ...
                    [self.control_signal(:,2:end),...
                    zeros(size(self.control_signal,1),1)];
            end
            
            self.num_data_pts = num_pts;
            assert(num_pts>0,...
                'Something went wrong with a data signal...')
            self.dat = this_dat;
            dat_to_shorten = {'L_global','L_sparse','S_global','S_sparse',...
                'L_global_modes', 'custom_control_signal', 'dat'};
            for fname = dat_to_shorten
                tmp = self.(fname{1});
                if isempty(tmp)
                    continue
                end
                try
                    self.(fname{1}) = tmp(:,1:num_pts);
                catch
                    self.(fname{1}) = tmp(1:num_pts,:);
                end
            end
        end
        
        function postprocess(self)
            self.total_sz = size(self.dat_with_control);
            %             if ~isempty(self.state_labels_ind_raw)
            %                 self.state_labels_ind = ...
            %                     self.state_labels_ind_raw(end-self.total_sz(2)+1:end);
            %             end
            
            if ~self.to_save_raw_data
                for f=fieldnames(struct(self))'
                    fname = f{1};
                    if ~contains(fname,'_raw') && ~strcmp(fname,'raw')
                        continue
                    elseif strcmp(fname, 'to_save_raw_data')
                        self.(fname) = false;
                    else
                        self.(fname) = [];
                    end
                end
            end
            
            % Setup all dependent row objects, if any
            %             used_rows = [];
            %             for i = 1:size(self.dependent_signals,1)
            %                 this_ind = self.dependent_signals.signal_indices{i};
            %                 if ischar(this_ind)
            %                     assert(ismember(this_ind,...
            %                         self.control_signals_metadata.Row),...
            %                         'Dependent signal must refer to a valid training signal')
            %                     this_ind = self.control_signals_metadata{this_ind,:}{:};
            %                     self.dependent_signals.signal_indices{i} = this_ind;
            %                 end
            %                 % Check if the row indices make sense
            %                 used_rows = [used_rows, this_ind]; %#ok<AGROW>
            %                 if length(unique(used_rows)) < length(used_rows)
            %                     % Note: can't check if ind are out of bounds here...
            %                     error('Dependent signals cannot overwrite each other')
            %                 end
            %                 % These should be classes which define a setup() method
            %                 setup_strings = self.dependent_signals.setup_arguments(i);
            %                 setup_args = cell(size(setup_strings));
            %                 for i2 = 1:length(setup_strings)
            %                     if isempty(setup_strings{i2})
            %                         continue;
            %                     end
            %                     setup_args{i2} = self.(setup_strings{i2});
            %                 end
            %                 setup(self.dependent_signals.signal_functions{i}, setup_args);
            %             end
        end
        
        function callback_plotter_reconstruction(self, ~, evt)
            % On left click:
            %   Plots the original data and the reconstruction
            % On right click:
            %   Prints neuron name
            this_neuron = round(evt.IntersectionPoint(2));
            if evt.Button==1
                %                 self.plot_reconstruction_interactive(false, this_neuron);
                self.AdaptiveDmdc_obj.plot_reconstruction(true, ...
                    true, true, this_neuron, true);
                %                 warning('With a custom control signal the names might be off...')
            else
                self.plot_reconstruction_interactive(true, this_neuron);
            end
        end
        
        function callback_plotter_eigenvector(self, ~, evt, V, D)
            % On left click:
            %   Plots the eigenvector associated with the eigenvalue
            %   clicked
            this_eig = evt.IntersectionPoint;
            tol = 1e-6;
            this_ind = find(abs(this_eig(1)+1i*this_eig(2) - D)<tol);
            if evt.Button==1
                figure;
                imagesc(real(V(:,this_ind)))
                title(sprintf('Eigenvector for eigenvalue %d',this_ind))
                if self.use_deriv
                    yticks(1:(self.original_sz(1)/2))
                else
                    yticks(1:self.original_sz(1))
                end
                yticklabels(self.get_names())
                xticks([])
                colorbar
            end
        end
        
        function append_control_metadata(self, new_metadata, to_push_indices)
            if ~exist('to_push_indices','var')
                to_push_indices = false;
            end
            % Simple sanity check and then append
            if ~isempty(self.control_signals_metadata)
                if to_push_indices
                    last_ctr_ind = max(cellfun(@max,...
                        self.control_signals_metadata.signal_indices));
                    if ~isnan(new_metadata.signal_indices{:})
                        new_metadata.signal_indices = ...
                            {new_metadata.signal_indices{:} + last_ctr_ind};
                    end
                else
                    assert(~any(cellfun( @(x) any(...
                        ismember(new_metadata.signal_indices{:}, x) ),...
                        self.control_signals_metadata.signal_indices) ),...
                        'Control signals must not overlap in indices')
                end
            end
            self.control_signals_metadata = ...
                [self.control_signals_metadata;
                new_metadata];
        end
        
        function dat = smart_filter(self, dat, aggresiveness)
            % Learns a cutoff frequency for each neuron (column) in 'dat'
            % 'aggressiveness' (1.5) should be > 1.0
            if ~exist('aggresiveness','var')
                aggresiveness = 1.5;
            else
                assert(aggresiveness>1.0,...
                    'Filter aggresiveness must be greater than 1.0')
            end
            sz = size(dat);
            assert(sz(1)>sz(2),...
                'Filter column-wise')
            
            for i=1:sz(2)
                if self.use_deriv && i>sz(2)/2
                    % This basically won't work for them very well
                    break
                end
                this_dat = dat(:,i);
                
                % Get the fft
                Fs = 1;            % Sampling frequency
                L = sz(1);             % Length of signal
                Y = fft(this_dat);
                P2 = abs(Y/L);
                P1 = P2(1:round(L/2+1));
                P1(2:end-1) = 2*P1(2:end-1);
                f = Fs*(0:(L/2))/L;
                
                % Learn a cutoff frequency
                %   The lower the aggressiveness, the higher the
                %   frequencies that are let through
                noise_level = max(P1(round(length(P1)/2):end))*aggresiveness;
                noise_beginning = f(find(P1-noise_level>0,1,'last'));
                Fpass = noise_beginning;
                Fstop = Fpass*1.5;
                
                % Design a low-pass filter
                Apass = 0.1;
                Astop = 20;
                Fs = 1;
                design_method = 'cheby2';
                %                 disp(i)
                lpFilt = designfilt('lowpassiir', ...
                    'PassbandFrequency',Fpass, 'StopbandFrequency',Fstop,...
                    'PassbandRipple',Apass,'StopbandAttenuation',Astop,...
                    'SampleRate',Fs,...
                    'DesignMethod',design_method);
                
                % Apply the filter in a zero-phase algorithm
                dat(:,i) = filtfilt(lpFilt, this_dat);
            end
        end
        
    end
    
    methods % For dependent variables
        function out = get.dat_without_control(self)
            out = self.dat(self.x_indices,1:self.num_data_pts);
        end
        
        function out = get.dat_with_control(self)
            out = ...
                [self.dat(self.x_indices,1:self.num_data_pts);...
                self.control_signal];
        end
        
        function out = get.x_indices(self)
            if ~isempty(fieldnames(self.AdaptiveDmdc_settings))
                out = self.AdaptiveDmdc_settings.x_indices;
            else
                out = [];
            end
        end
        
        function set.x_indices(self, val)
            self.AdaptiveDmdc_settings.x_indices = val;
        end
    end
    
    methods (Static)
        function dat = flat_filter(dat, window_size)
            w = window_size;
            dat = filtfilt(ones(w,1)/w,1,dat);
        end
        
        function out = state_length_count(states)
            % Counts the length of each state up to that point
            % e.g.
            % >> state_length_count([1 1 2 2 2 2])
            %   [ [1 2 0 0 0 0]
            %     [0 0 1 2 3 4] ]
            % ... Testing out a start at 0!
            %   [ [0 1 0 0 0 0]
            %     [0 0 0 1 2 3] ]
            %
            % Assumes input is sequentially labeled states
            out = zeros(length(unique(states)), size(states,2));
            this_val = states(1);
            out(this_val, 1) = 1;
            for i=2:length(states)
                if this_val == states(i)
                    out(this_val, i) = out(this_val, i-1) + 1;
                    continue
                else
                    this_val = states(i);
                    out(this_val, i) = 1;
                end
            end
        end
        
        function [binary_labels, all_states] = ...
                calc_binary_labels(these_states, to_add_zeros)
            % Calculates a one-hot encoding of the input
            % Input:
            %   these_states - integer labels
            %   to_add_zeros (0) - to add rows of zeros if integer
            %       states are missing, up to the given value
            % Output:
            %   binary_labels - one-hot encoding of the labels
            %   all_states - all states that were found, indexing the rows
            if ~exist('to_add_zeros', 'var')
                to_add_zeros = 0;
            end
            
            if to_add_zeros > 0
                % Assume that the minimum value is present; others may be
                % missing
                state0 = min(these_states);
                if state0==0 % Kato format
                    these_states = these_states + 1;
                    state0 = 1;
                end
                all_states = state0:(state0 + to_add_zeros);
                sz = [to_add_zeros, 1];
            else
                all_states = sort(unique(these_states));
                sz = [length(all_states), 1];
            end
            binary_labels = zeros(sz(1),size(these_states,2));
            
            for i = 1:sz(1)
                binary_labels(i,:) = (these_states==all_states(i));
            end
        end
        
        function expand_axes(fig, dat)
            % Expands axes for data that will be plotted, e.g. for a movie
            figure(fig);
            x = xlim;
            x1 = min([dat(:,1); x(1)]);
            x2 = max([dat(:,1); x(2)]);
            xlim([x1 x2]);
            y = ylim;
            y1 = min([dat(:,2); y(1)]);
            y2 = max([dat(:,2); y(2)]);
            ylim([y1 y2]);
            z = zlim;
            z1 = min([dat(:,3); z(1)]);
            z2 = max([dat(:,3); z(2)]);
            zlim([z1 z2]);
        end
        
        function [dat, baselines] = calc_baseline(dat)
            % Calculates the baselines using only subsets of the data that
            % are relatively flat, then taking the portions of those
            % subsets after a changepoint to deal with refractory periods.
            % Sensitive to drift in the data; will be similar to the mean
            % if there is a lot of noise.
            f = @(x)abs(TVRegDiff(x,5,1e-4,[],[],[],[],false,false));
            sz = size(dat);
            baselines = nan(sz(1),1);
            for i = 1:sz(1)
                dxdt = f(dat(i,:));
                flat_ind = find(dxdt(1:end-1) < 2*median(dxdt));
                gmdist = fitgmdist(dat(i,flat_ind)',2, 'Replicates',10);
                clust_idx = cluster(gmdist, dat(i,flat_ind)');
                centroids = gmdist.mu;
                
                % Two cases: high/low plateaus (i.e. 2 real clusters) OR spikes with a
                % lot of low, i.e. basically one cluster
                ind1 = (clust_idx==1);
                ind2 = (clust_idx==2);
                if length(find(ind1)) > length(find(ind2))*20
                    ind = ind1;
                elseif length(find(ind2)) > length(find(ind1))*20
                    ind = ind2;
                else
                    [baselines(i), which_clust] = min(centroids);
                    ind = (clust_idx==which_clust);
                end
                [all_starts, all_ends] = calc_contiguous_blocks(ind, 30);
                baseline_ind = [];
                for i2 = 1:length(all_starts)
                    these_dat_ind = flat_ind(all_starts(i2):all_ends(i2));
                    chg_pt_dat = findchangepts(dat(i,these_dat_ind));
                    chg_pt_flat = find(flat_ind==these_dat_ind(chg_pt_dat));
                    baseline_ind = [baseline_ind; ...
                        flat_ind(chg_pt_flat:all_ends(i2))]; %#ok<AGROW>
                end
                if ~isempty(baseline_ind)
                    baselines(i) = mean(dat(i,baseline_ind));
                else
                    baselines(i) = mean(dat(i,ind));
                end
            end % for i
            dat = dat - baselines;
        end % function
    end
    
end

