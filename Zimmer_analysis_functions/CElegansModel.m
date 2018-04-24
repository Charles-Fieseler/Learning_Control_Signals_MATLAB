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
        AdaptiveDmdc_settings
        % Data importing
        use_deriv
    end
        
    properties (Hidden=true, SetAccess=private)
        raw
        original_sz
        dat_sz
        total_sz
        
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
    end
    
    properties (SetAccess=private)
        dat
        dat_with_control
        dat_without_control
        
        AdaptiveDmdc_obj
        
        user_control_matrix
        user_control_input
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
        
        function add_partial_original_control_signal(self, signal_ind)
            % Adds some of the current control signals to the user control
            % matrix
            assert(max(signal_ind)<=self.total_sz(1) && ...
                min(signal_ind)>self.original_sz(1),...
                'Indices must be within the discovered control signal')
            
            num_neurons = self.original_sz(1);
            A = self.AdaptiveDmdc_obj.A_separate(1:num_neurons,:);
            u = self.AdaptiveDmdc_obj.dat;
            for i = 1:length(signal_ind)
                this_ind = signal_ind(i);
                this_connectivity = abs(A(:,this_ind))>0;
                this_amp = A(:,this_ind);
                this_signal = u(this_ind,:);
                self.add_manual_control_signal(...
                    this_connectivity, this_amp, this_signal);
            end
            
        end
        
        function [A_old, dat_old] = set_AdaptiveDmdc_controller(self,...
                new_control_matrix, new_control_input)
            % Save original matrices data
            A_old = self.AdaptiveDmdc_obj.A_separate;
            dat_old = self.AdaptiveDmdc_obj.dat;
            % Get new matrices and data
            num_real_neurons = size(self.dat_without_control,1);
            num_controllers = size(new_control_matrix,2);
            num_neurons_and_controllers = ...
                num_real_neurons + num_controllers;
            new_control_matrix = [new_control_matrix;...
                zeros(num_controllers)];
            A_new = ...
                [A_old(1:num_neurons_and_controllers,1:num_real_neurons), ...
                new_control_matrix];
            dat_new = ...
                [self.dat_without_control;...
                new_control_input];
            % Update the object properties
            self.AdaptiveDmdc_obj.A_separate = A_new;
            self.AdaptiveDmdc_obj.dat = dat_new;
        end
        
        function reset_AdaptiveDmdc_controller(self, A_old, dat_old)
            % Update the object properties
            self.AdaptiveDmdc_obj.A_separate = A_old;
            self.AdaptiveDmdc_obj.dat = dat_old;
        end
        
    end
    
    methods % Plotting
        
        function plot_reconstruction_user_control(self)
            % Uses manually set control signals
            [A_old, dat_old] = ...
                self.set_AdaptiveDmdc_controller(...
                self.user_control_matrix, self.user_control_input);
            
            % [With manual control matrices]
            self.AdaptiveDmdc_obj.plot_reconstruction(true, true);
            title('Data reconstructed with user-defined control signal')
            
            % Reset AdaptiveDmdc object
            self.reset_AdaptiveDmdc_controller(A_old, dat_old);
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
                ...% Data importing
                'use_deriv', false);
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
        end
        
        function import_from_struct(self, Zimmer_struct)
            warning('Assumes struct of Zimmer type')
            self.raw = Zimmer_struct.traces.';
            if isempty(fieldnames(self.AdaptiveDmdc_settings))
                x_ind = 1:size(self.raw,1);
                id_struct = struct('ID',{Zimmer_struct.ID},...
                    'ID2',{Zimmer_struct.ID2},'ID3',{Zimmer_struct.ID3});
                self.AdaptiveDmdc_settings =...
                    struct('to_plot_cutoff',true,...
                    'to_plot_data_and_outliers',true,...
                    'id_struct',id_struct,...
                    'sort_mode','user_set',...
                    'x_indices',x_ind,...
                    'dmd_mode','naive');
            end
        end
        
        %Data processing
        function preprocess(self)
            if self.verbose
                disp('Preprocessing...')
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
        end
    end
    
    methods (Static)
        function dat = flat_filter(dat, window_size)
            w = window_size;
            dat = filter(ones(w,1)/w,1,dat);
        end
    end
    
end

