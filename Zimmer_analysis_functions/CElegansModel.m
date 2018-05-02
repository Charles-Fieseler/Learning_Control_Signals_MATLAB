classdef CElegansModel < SettingsImportableFromStruct
    %% C elegans (linear) model
    % Using AdaptiveDmdc and RobustPCA on Calcium imaging data, this class
    % builds a linear model with control for the brain dynamics
    %
    % To get the control signal, does Robust PCA twice
    %   1st time: low lambda value, with most of the data in the sparse
    %       component. The extremely low-rank component is interpreted as
    %       encoding global states
    %   2nd time: high lambda value, with most of the data in the 'low-rank'
    %       component (which is actually nearly full-rank). The extremely
    %       sparse component is interpreted as control signals, and the
    %       'low-rank' component should be model-able using dmdc
    %
    % Then AdaptiveDmdc:
    %   AdaptiveDmdc: fit a DMDc model to the 2nd 'low-rank' component, using
    %       the extremely sparse component (2nd robustPCA) and the extremely
    %       low-rank component (1st robustPCA) as control signals
    %
    % INPUTS
    %   INPUT1 -
    %   INPUT2 -
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
    %   .m files, .mat files, and MATLAB products required:
    %
    %   See also: OTHER_FUNCTION_NAME
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
        lambda_global
        lambda_sparse
        
        % Data processing
        filter_window_dat
        filter_window_global
        augment_data
        to_subtract_mean
        dmd_mode
        AdaptiveDmdc_settings
        % Data importing
        use_deriv
        use_only_deriv
        to_normalize_deriv
    end
        
    properties (Hidden=true, SetAccess=private)
        raw
        raw_deriv
        dat
        original_sz
        dat_sz
        total_sz
        
        A_old
        dat_old
        
        % Robust PCA matrices
        L_global_raw
        S_global_raw
        L_global
        S_global
        L_global_modes
        
        L_sparse_raw
        S_sparse_raw
        L_sparse
        S_sparse
        L_sparse_modes
        
        state_labels_ind
        state_labels_ind_raw
        state_labels_key
    end
    
    properties (SetAccess=private)
        dat_with_control
        dat_without_control
        
        AdaptiveDmdc_obj
        % Changing the control signals and/or matrix
        user_control_matrix
        user_control_input
        user_neuron_ablation
        user_control_reconstruction
    end
    
    methods
        function self = CElegansModel(file_or_dat, settings)
            
            %% Set defaults and import settings
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
            %==========================================================================
            
            %% Import data
            if ischar(file_or_dat)
                %     self.filename = file_or_dat;
                tmp_dat = importdata(file_or_dat);
                if isstruct(tmp_dat)
                    self.import_from_struct(tmp_dat);
                elseif isnumeric(tmp_dat)
                    self.raw = tmp_dat;
                else
                    error('Filename must contain a struct or matrix')
                end
            elseif isnumeric(file_or_dat)
                self.raw = file_or_dat;
            else
                error('Must pass data matrix or filename')
            end
            self.preprocess();
            %==========================================================================
            

            %% Robust PCA (get control signal) and DMD with that signal
            self.calc_all_control_signals();
            self.calc_AdaptiveDmdc();
            self.postprocess();
            %==========================================================================
            

            %% Initialize user control structure
            self.reset_user_control();
            %==========================================================================


        end
        
    end
    
    methods % Adding control signals
        
        function reset_user_control(self)
            self.user_control_input = [];
            self.user_control_matrix = [];
            self.user_neuron_ablation = [];
        end
        
        function add_manual_control_signal(self, ...
                neuron_ind, neuron_amps, ...
                signal_ind, signal_amps)
            % Adds a row to the control matrix (B) going from 
            assert(max(neuron_ind)<self.dat_sz(1),...
                'Control target must be in the original data set')
            this_ctr_connectivity = ...
                zeros(size(self.dat_without_control, 1), 1);
            if isscalar(neuron_amps)
                neuron_amps = neuron_amps*ones(size(neuron_ind));
            end
            this_ctr_connectivity(neuron_ind) = neuron_amps;
            self.user_control_matrix = ...
                [self.user_control_matrix ...
                this_ctr_connectivity];
            if length(signal_ind) == self.total_sz(2)
                self.user_control_input = [self.user_control_input;...
                    signal_ind];
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
                signal_ind, custom_signal, signal_start)
            % Adds some of the current control signals to the user control
            % matrix
            self.user_control_reconstruction = [];
            num_neurons = self.original_sz(1);
            if ~exist('custom_signal','var')
                custom_signal = self.AdaptiveDmdc_obj.dat;
            elseif size(custom_signal,1) < max(signal_ind)
                % Then it doesn't include the original data, which is the
                % format we're looking for
                tmp = zeros(max(signal_ind),...
                    size(custom_signal,2));
                tmp(signal_ind,:) = custom_signal;
                custom_signal = tmp;
            end
            if ~exist('signal_start','var')
                assert(size(custom_signal,2) == self.total_sz(2),...
                    'Custom signal must be defined for the entire tspan')
            else
                tmp = zeros(size(custom_signal, 1), self.total_sz(2));
                signal_end = signal_start+size(custom_signal, 2)-1;
                tmp(:,signal_start:signal_end) = custom_signal;
                custom_signal = tmp;
            end
            assert(max(signal_ind) <= self.total_sz(1) && ...
                min(signal_ind) > num_neurons,...
                'Indices must be within the discovered control signal')
            
            A = self.AdaptiveDmdc_obj.A_separate(1:num_neurons,:);
            for i = 1:length(signal_ind)
                this_ind = signal_ind(i);
                this_connectivity = abs(A(:,this_ind))>0;
                this_amp = A(:,this_ind);
                this_signal = custom_signal(this_ind,:);
                self.add_manual_control_signal(...
                    this_connectivity, this_amp, this_signal);
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
            % Save original matrices data
            self.A_old = self.AdaptiveDmdc_obj.A_separate;
            self.dat_old = self.AdaptiveDmdc_obj.dat;
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
        end
        
        function [signals, signal_mat, mean_signal] =...
                get_control_signal_during_label(self, ...
                    which_label, num_preceding_frames)
            if ~exist('num_preceding_frames','var')
                num_preceding_frames = 1;
            end
            assert(ismember(which_label, self.state_labels_key),...
                'Invalid state label')
            
            % Get indices for this behavior
            which_label_num = ...
                find(strcmp(self.state_labels_key, which_label));
            transition_ind = diff(self.state_labels_ind==which_label_num);
            start_ind = max(...
                find(transition_ind==1) - num_preceding_frames, 1);
            end_ind = find(transition_ind==-1);
            if self.state_labels_ind(end) == which_label_num
                end_ind = [end_ind length(transition_ind)]; 
            end
            if self.state_labels_ind(1) == which_label_num
                start_ind = [1 start_ind]; 
            end
            assert(length(start_ind)==length(end_ind))
            n = length(start_ind);
            
            % Get the actual control signals for these indices
            signals = cell(n, 1);
            ctr = self.dat_sz(1)+1;
            min_signal_length = 0;
            for i = 1:n
                these_ind = start_ind(i):end_ind(i);
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
        
    end
    
    methods % Plotting
        
        function plot_reconstruction_user_control(self)
            % Uses manually set control signals
            self.set_AdaptiveDmdc_controller();
            
            % [With manual control matrices]
            self.user_control_reconstruction = ...
                self.AdaptiveDmdc_obj.plot_reconstruction(true, true);
            title('Data reconstructed with user-defined control signal')
            
            % Reset AdaptiveDmdc object
            self.reset_AdaptiveDmdc_controller();
        end
        
        function plot_colored_data(self, plot_pca, plot_opt)
            if ~exist('plot_pca','var')
                plot_pca = false;
            end
            if ~exist('plot_opt','var')
                plot_opt = 'o';
            end
            [self.L_sparse_modes,~,~,proj3d] = plotSVD(self.L_sparse,...
                struct('PCA3d',plot_pca,'sigma',false));
            plot_colored(proj3d,...
                self.state_labels_ind_raw(end-size(proj3d,2)+1:end),...
                self.state_labels_key, plot_opt);
            title('Dynamics of the low-rank component (data)')
        end
        
        function plot_colored_user_control(self)
            % Plots user control data on top of colored original dataset
            assert(~isempty(self.user_control_reconstruction),...
                'No reconstructed data stored')
            
            self.plot_colored_data(false, '.');
            
            modes_3d = self.L_sparse_modes(:,1:3);
            x = 1:size(modes_3d,1);
            proj_3d = (modes_3d.')*self.user_control_reconstruction(x,:);
            plot3(proj_3d(1,:),proj_3d(2,:),proj_3d(3,:), 'k*')
        end
        
        function plot_colored_fixed_point(self)
            % Plots the fixed point on top of colored original dataset
            self.plot_colored_data(false, '.');
            
            % Get the dynamics and control matrices, and the control signal
            ad_obj = self.AdaptiveDmdc_obj;
            x_dat = 1:ad_obj.x_len;
            x_ctr = (ad_obj.x_len+1):self.total_sz(1);
            A = ad_obj.A_original(x_dat, x_dat);
            B = ad_obj.A_original(x_dat, x_ctr);
            u = self.dat_with_control(x_ctr, :);
            % Reconstruct the attractor and project it into the same space
            attractor_reconstruction = (A\B)*u;
            
            modes_3d = self.L_sparse_modes(:,1:3);
            proj_3d = (modes_3d.')*attractor_reconstruction;
            plot3(proj_3d(1,:),proj_3d(2,:),proj_3d(3,:), 'k*')
        end
        
        function plot_colored_control_arrow(self, ...
                which_ctr_modes, arrow_base)
            if ~exist('arrow_base','var')
                arrow_base = [0, 0, 0];
            end
            % Plots a control direction on top of colored original dataset
            self.plot_colored_data(false, '.');
            
            % Get control matrices and columns to project
            ad_obj = self.AdaptiveDmdc_obj;
            x_dat = 1:ad_obj.x_len;
            x_ctr = (ad_obj.x_len+1):self.total_sz(1);
            B = ad_obj.A_original(x_dat, x_ctr);
            % Reconstruct the attractor and project it into the same space
            arrow_direction = B(:,which_ctr_modes);
            
            modes_3d = self.L_sparse_modes(:,1:3);
            proj_3d = (modes_3d.')*arrow_direction;
            arrow_length = 1;
            for j=1:size(proj_3d,2)
                quiver3(arrow_base(1),arrow_base(2),arrow_base(3),...
                    proj_3d(1,j),proj_3d(2,j),proj_3d(3,j), ...
                    arrow_length, 'LineWidth', 2)
            end
        end
        
        function plot_mean_transition_signals(self, ...
                which_label, num_preceding_frames)
            % Uses hand-labeled behavior
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
    end
    
    methods (Access=private)
        
        function set_defaults(self)
            defaults = struct(...
                'verbose',true,...
                ...% Getting the control signal
                'lambda_global', 0.0065,...
                'lambda_sparse', 0.05,...
                ...% Data processing
                'filter_window_dat', 3,...
                'filter_window_global', 10,...
                'AdaptiveDmdc_settings', struct(),...
                'augment_data', 0,...
                'to_subtract_mean',false,...
                'dmd_mode', 'naive',...
                ...% Data importing
                'use_deriv', false,...
                'use_only_deriv', false,...
                'to_normalize_deriv', false);
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
        end
        
        function import_from_struct(self, Zimmer_struct)
            warning('Assumes struct of Zimmer type')
            self.raw = Zimmer_struct.traces.';
            if self.use_deriv || self.use_only_deriv
                self.raw_deriv = Zimmer_struct.tracesDif.';
            end
            self.state_labels_ind_raw = Zimmer_struct.SevenStates;
            self.state_labels_key = Zimmer_struct.SevenStatesKey;
            if isempty(fieldnames(self.AdaptiveDmdc_settings))
                if ~self.use_deriv
                    x_ind = 1:size(self.raw,1);
                else
                    x_ind = 1:(2*size(self.raw,1));
                end
                id_struct = struct('ID',{Zimmer_struct.ID},...
                    'ID2',{Zimmer_struct.ID2},'ID3',{Zimmer_struct.ID3});
                self.AdaptiveDmdc_settings =...
                    struct('to_plot_cutoff',true,...
                    'to_plot_data_and_outliers',true,...
                    'id_struct',id_struct,...
                    'sort_mode','user_set',...
                    'x_indices',x_ind,...
                    'dmd_mode',self.dmd_mode);
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
            self.dat_sz = size(self.raw);
            
            %If augmenting, stack data offset by 1 column on top of itself;
            %note that this decreases the overall number of columns (time
            %slices)
            aug = self.augment_data;
            self.original_sz = self.dat_sz;
            if aug>0
                newSz = [self.dat_sz(1)*aug, self.dat_sz(2)-aug];
                newDat = zeros(newSz);
                for j=1:aug
                    thisOldCols = j:(newSz(2)+j-1);
                    thisNewRows = (1:self.dat_sz(1))+self.dat_sz(1)*(j-1);
                    newDat(thisNewRows,:) = self.raw(:,thisOldCols);
                end
                self.dat_sz = newSz;
                self.raw = newDat;
            end
            
            self.dat = self.raw;
            if self.to_subtract_mean
                for jM=1:self.dat_sz(1)
                    self.dat(jM,:) = self.raw(jM,:) - mean(self.raw(jM,:));
                end
            end

            % Moving average filter
            if self.filter_window_dat>1
                self.dat = ...
                    self.flat_filter(self.dat.',self.filter_window_dat).';
            end
        end
        
        function new_raw = preprocess_deriv(self)
            % Aligns and optionally normalizes the derivative signal
            deriv = self.raw_deriv;
            if self.to_normalize_deriv
                deriv = deriv .* (std(self.raw,[],2) ./ std(deriv,[],2));
            end
            
            % Derivative is one frame short, so throw out the last frame
            if ~self.use_only_deriv
                new_raw = [self.raw(:,1:end-1); deriv];
            else
                new_raw = deriv;
            end
        end
        
        function calc_all_control_signals(self)
            % Calls subfunctions to calculate control signals
            self.calc_global_signal();
            self.calc_sparse_signal();
            self.calc_dat_and_control_signal();
        end
        
        function calc_global_signal(self)
            % Gets VERY low-rank signal
            [self.L_global_raw, self.S_global_raw] = ...
                RobustPCA(self.dat, self.lambda_global);
            
            self.L_global = self.flat_filter(...
                self.L_global_raw, self.filter_window_global);
            tmp_dat = self.L_global - mean(self.L_global,2);
            [u, ~] = svd(tmp_dat(:,self.filter_window_global+5:end).');
            x = 1:rank(self.L_global);
            self.L_global_modes = u(:,x);
            
            self.S_global = self.S_global_raw;
        end
        
        function calc_sparse_signal(self)
            % Calculates very sparse signal
            [self.L_sparse_raw, self.S_sparse_raw] = ...
                RobustPCA(self.dat, self.lambda_sparse);
            self.L_sparse = self.L_sparse_raw;
            self.S_sparse = self.S_sparse_raw;
        end
        
        function calc_dat_and_control_signal(self)
            % Uses results from 2 different Robust PCA runs
            
            % Data to be reconstructed is everything EXCEPT the sparse
            % control signals
            this_dat = self.L_sparse - mean(self.L_sparse,2);
            % Sparse signal with thresholding
            sparse_signal = self.S_sparse - mean(self.S_sparse,2);
            tol = 1e-2;
            sparse_signal = sparse_signal(max(abs(sparse_signal),[],2)>tol,:);
            % Use top svd modes for the low-rank component
            L_low_rank = self.L_global_modes - mean(self.L_global_modes,1);
            % Create the augmented dataset (these might have different
            % amounts of filtering, therefore slightly different sizes)
            L_low_rank = L_low_rank';
            num_pts = min([size(L_low_rank,2) size(sparse_signal,2) size(this_dat,2)]);
            self.dat_with_control = ...
                [this_dat(:,1:num_pts);...
                sparse_signal(:,1:num_pts);...
                L_low_rank(:,1:num_pts)];
            self.dat_without_control = this_dat(:,1:num_pts);
        end
        
        function calc_AdaptiveDmdc(self)
            % Uses external class AdaptiveDmdc
            self.AdaptiveDmdc_obj = AdaptiveDmdc(self.dat_with_control,...
                self.AdaptiveDmdc_settings);
        end
        
        function postprocess(self)
            self.total_sz = size(self.dat_with_control);
            self.state_labels_ind = ...
                self.state_labels_ind_raw(end-self.total_sz(2)+1:end);
        end
    end
    
    methods (Static)
        function dat = flat_filter(dat, window_size)
            w = window_size;
            dat = filter(ones(w,1)/w,1,dat);
        end
    end
    
end

