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
    %   .m files, .mat files, and MATLAB products required:(updated on 15-Jun-2018)
    %             MATLAB (version 9.2)
    %             Signal Processing Toolbox (version 7.4)
    %             Symbolic Math Toolbox (version 7.2)
    %             Curve Fitting Toolbox (version 3.5.5)
    %             System Identification Toolbox (version 9.6)
    %             Optimization Toolbox (version 7.6)
    %             Simulink Control Design (version 4.5)
    %             Statistics and Machine Learning Toolbox (version 11.1)
    %             Computer Vision System Toolbox (version 7.3)
    %             checkModes.m
    %             func_DMD.m
    %             func_DMDc.m
    %             AdaptiveDmdc.m
    %             sparse_dmd.m
    %             AbstractDmd.m
    %             SettingsImportableFromStruct.m
    %             plotSVD.m
    %             plot_2imagesc_colorbar.m
    %             plot_colored.m
    %             RobustPCA.m
    %             minimize.m
    %             xgeqp3_m.mexw64
    %             xormqr_m.mexw64
    %             optdmd.m
    %             varpro2.m
    %             varpro2dexpfun.m
    %             varpro2expfun.m
    %             varpro_lsqlinopts.m
    %             varpro_opts.m
    %             adjustedVariance.m
    %             computeTradeOffCurve.m
    %             invPow.m
    %             optimizeVariance.m
    %             sparsePCA.m
    %             abs.m
    %             blkdiag.m
    %             builtins.m
    %             cat.m
    %             conj.m
    %             conv.m
    %             ctranspose.m
    %             cumprod.m
    %             cumsum.m
    %             diag.m
    %             disp.m
    %             end.m
    %             eq.m
    %             exp.m
    %             find.m
    %             full.m
    %             ge.m
    %             gt.m
    %             hankel.m
    %             horzcat.m
    %             imag.m
    %             ipermute.m
    %             isreal.m
    %             kron.m
    %             ldivide.m
    %             le.m
    %             log.m
    %             lt.m
    %             max.m
    %             min.m
    %             minus.m
    %             mldivide.m
    %             mpower.m
    %             mrdivide.m
    %             mtimes.m
    %             ne.m
    %             nnz.m
    %             norm.m
    %             permute.m
    %             plus.m
    %             polyval.m
    %             power.m
    %             prod.m
    %             rdivide.m
    %             real.m
    %             reshape.m
    %             size.m
    %             sparse.m
    %             spy.m
    %             sqrt.m
    %             std.m
    %             subsasgn.m
    %             subsref.m
    %             sum.m
    %             times.m
    %             toeplitz.m
    %             transpose.m
    %             tril.m
    %             triu.m
    %             uminus.m
    %             uplus.m
    %             var.m
    %             vertcat.m
    %             eq.m
    %             ge.m
    %             gt.m
    %             le.m
    %             lt.m
    %             ne.m
    %             commands.m
    %             cvx_begin.m
    %             cvx_end.m
    %             cvx_error.m
    %             cvx_license.p
    %             cvx_version.m
    %             avg_abs_dev.m
    %             avg_abs_dev_med.m
    %             berhu.m
    %             cvx_recip.m
    %             det_inv.m
    %             det_rootn.m
    %             functions.m
    %             geo_mean.m
    %             huber_pos.m
    %             inv_pos.m
    %             lambda_max.m
    %             lambda_sum_largest.m
    %             log_normcdf.m
    %             log_sum_exp.m
    %             matrix_frac.m
    %             norm_nuc.m
    %             norms.m
    %             pow_abs.m
    %             pow_cvx.m
    %             pow_p.m
    %             pow_pos.m
    %             prod_inv.m
    %             quad_form.m
    %             quad_over_lin.m
    %             quad_pos_over_lin.m
    %             rel_entr.m
    %             sigma_max.m
    %             square.m
    %             square_abs.m
    %             square_pos.m
    %             sum_largest.m
    %             sum_log.m
    %             sum_square.m
    %             sum_square_abs.m
    %             sum_square_pos.m
    %             trace_inv.m
    %             trace_sqrtm.m
    %             vec.m
    %             det_inv.m
    %             det_rootn.m
    %             geo_mean.m
    %             inv_pos.m
    %             lambda_max.m
    %             log_sum_exp.m
    %             matrix_frac.m
    %             norms.m
    %             poly_env.m
    %             pow_abs.m
    %             pow_pos.m
    %             prod_inv.m
    %             quad_form.m
    %             quad_over_lin.m
    %             rel_entr.m
    %             sigma_max.m
    %             square_abs.m
    %             square_pos.m
    %             sum_square.m
    %             sum_square_abs.m
    %             sum_square_pos.m
    %             vec.m
    %             cvx_id.m
    %             cvx_setdual.m
    %             cvx_value.m
    %             bcompress.m
    %             buncompress.m
    %             cvx.m
    %             cvx_basis.m
    %             cvx_classify.m
    %             cvx_constant.m
    %             cvx_getdual.m
    %             cvx_isaffine.m
    %             cvx_isconcave.m
    %             cvx_isconstant.m
    %             cvx_isconvex.m
    %             cvx_isnonzero.m
    %             cvx_readlevel.m
    %             cvx_setdual.m
    %             cvx_value.m
    %             cvx_vexity.m
    %             in.m
    %             keywords.m
    %             matlab6.m
    %             sets.m
    %             sparsify.m
    %             svec.m
    %             type.m
    %             value.m
    %             cvxcnst.m
    %             disp.m
    %             display.m
    %             double.m
    %             logical.m
    %             rhs.m
    %             colon.m
    %             cvx_basis.m
    %             cvx_value.m
    %             cvxaff.m
    %             cvxdual.m
    %             disp.m
    %             display.m
    %             dof.m
    %             inuse.m
    %             isreal.m
    %             name.m
    %             problem.m
    %             size.m
    %             subsref.m
    %             type.m
    %             value.m
    %             cvx_id.m
    %             cvxobj.m
    %             disp.m
    %             display.m
    %             isempty.m
    %             isequal.m
    %             length.m
    %             ndims.m
    %             numel.m
    %             subsasgn.m
    %             subsref.m
    %             cvx_value.m
    %             cvxprob.m
    %             disp.m
    %             eliminate.m
    %             eq.m
    %             extract.m
    %             index.m
    %             ne.m
    %             newcnstr.m
    %             newdual.m
    %             newnonl.m
    %             newobj.m
    %             newtemp.m
    %             newvar.m
    %             pop.m
    %             solve.m
    %             spy.m
    %             subsasgn.m
    %             subsref.m
    %             touch.m
    %             apply.m
    %             cvx_collapse.m
    %             cvx_constant.m
    %             cvx_getdual.m
    %             cvx_id.m
    %             cvx_isaffine.m
    %             cvx_isconcave.m
    %             cvx_isconstant.m
    %             cvx_isconvex.m
    %             cvx_setdual.m
    %             cvx_value.m
    %             cvxtuple.m
    %             disp.m
    %             eq.m
    %             ge.m
    %             gt.m
    %             in.m
    %             le.m
    %             lt.m
    %             ne.m
    %             numel.m
    %             subsasgn.m
    %             subsref.m
    %             testall.m
    %             cvx_accept_concave.m
    %             cvx_accept_convex.m
    %             cvx_basis.m
    %             cvx_bcompress.m
    %             cvx_bcompress_mex.mexw64
    %             cvx_c2r.m
    %             cvx_check_dimension.m
    %             cvx_check_dimlist.m
    %             cvx_class.m
    %             cvx_classify.m
    %             cvx_clearpath.m
    %             cvx_clearspath.m
    %             cvx_collapse.m
    %             cvx_constant.m
    %             cvx_default_dimension.m
    %             cvx_eliminate_mex.mexw64
    %             cvx_expand_dim.m
    %             cvx_expert_check.m
    %             cvx_getdual.m
    %             cvx_global.m
    %             cvx_id.m
    %             cvx_ids.m
    %             cvx_isaffine.m
    %             cvx_isconcave.m
    %             cvx_isconstant.m
    %             cvx_isconvex.m
    %             cvx_isnonzero.m
    %             cvx_r2c.m
    %             cvx_readlevel.m
    %             cvx_reduce_size.m
    %             cvx_remap.m
    %             cvx_reshape.m
    %             cvx_setdual.m
    %             cvx_setpath.m
    %             cvx_setspath.m
    %             cvx_size_check.m
    %             cvx_subs2str.m
    %             cvx_subsasgn.m
    %             cvx_subsref.m
    %             cvx_use_sparse.m
    %             cvx_value.m
    %             cvx_vexity.m
    %             cvx_zeros.m
    %             structures.m
    %             cvx_create_structure.m
    %             cvx_invert_structure.m
    %             cvx_orthog_structure.m
    %             cvx_replicate_structure.m
    %             cvx_s_banded.m
    %             cvx_s_symmetric.m
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
        max_rank_global
        global_signal_mode
        lambda_sparse
        
        % Data processing
        filter_window_dat
        filter_window_global
        augment_data
        to_subtract_mean
        to_subtract_mean_sparse
        to_subtract_mean_global
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
        S_global_nnz
        L_global_rank
        
        L_sparse_raw
        S_sparse_raw
        L_sparse
        S_sparse
        L_sparse_modes
        S_sparse_nnz
        L_sparse_rank
        
        state_labels_ind
        state_labels_ind_raw
        state_labels_key
    end
    
    properties (SetAccess=public)
        dat_with_control
        dat_without_control
        
        AdaptiveDmdc_obj
        % Changing the control signals and/or matrix
        user_control_matrix
        user_control_input
        user_neuron_ablation
        user_control_reconstruction
        
        pareto_struct
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
            elseif isstruct(file_or_dat)
                self.import_from_struct(file_or_dat);
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
        
        function calc_pareto_front(self, lambda_vec, global_signal_mode,...
                to_append, max_val)
            % Calculates the errors as a function of the different lambda
            % values
            %   Currently only works with the sparse control signal
            if ~exist('global_signal_mode','var')
                global_signal_mode = 'ID';
            end
            if ~exist('to_append','var')
                to_append = false;
            end
            if ~exist('max_val','var')
                max_val = 0; % Do not truncate values
            end
            self.global_signal_mode = global_signal_mode;
            all_errors = zeros(size(lambda_vec));
            
            for i=1:length(lambda_vec)
                self.lambda_sparse = lambda_vec(i);
                self.calc_all_control_signals();
                self.calc_AdaptiveDmdc();
                self.postprocess();
                all_errors(i) = ...
                    self.AdaptiveDmdc_obj.calc_reconstruction_error();
            end
            
            if max_val>0
                all_errors(all_errors>max_val) = NaN;
            end
            
            if to_append && ~isempty(self.pareto_struct.lambda_vec)
                % Assume lambdas are the same
                self.pareto_struct.all_errors = ...
                    [self.pareto_struct.all_errors;...
                    all_errors];
                self.pareto_struct.global_signal_modes = ...
                    [self.pareto_struct.global_signal_modes;
                    global_signal_mode];
            else
                self.pareto_struct.lambda_vec = lambda_vec;
                persistence_model_error = ...
                    self.AdaptiveDmdc_obj.calc_reconstruction_error([],true);
                self.pareto_struct.all_errors = ...
                    [ones(size(all_errors))*persistence_model_error;...
                    all_errors];
                self.pareto_struct.global_signal_modes = ...
                    {'Persistence'; global_signal_mode};
            end
            
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
            
            A = self.AdaptiveDmdc_obj.A_separate(1:num_neurons,:);
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
            
            out = func(self);
            
            self.reset_user_control();
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
            
            out = func(self);
            
            self.reset_user_control();
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
            neuron_names = self.AdaptiveDmdc_obj.get_names();
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
                use_only_known_neurons, abs_tol)
            if ~exist('use_only_known_neurons','var')
                use_only_known_neurons = true;
            end
            if ~exist('abs_tol','var')
                abs_tol = 0.001;
            end
            
            num_neurons = self.original_sz(1);
            B_global = self.AdaptiveDmdc_obj.A_original(1:num_neurons,...
                            (2*num_neurons+1):end-2);
            % Narrow these down to which neurons are important for which behaviors
            %   Assume a single control signal (ID); ignore offset
            group1 = (B_global > abs_tol);
            group2 = (B_global < -abs_tol);
            % Get names and sort if required
            neuron_names = self.AdaptiveDmdc_obj.get_names();
            if use_only_known_neurons
                neuron_ind = ~strcmp(neuron_names,'');
                neuron_names = neuron_names(neuron_ind);
            else
                neuron_ind = 1:self.original_sz(1);
            end
            neuron_ind = find(neuron_ind);
            % Cast into strings
            neuron_roles = cell(size(neuron_ind));
            for i = 1:length(neuron_ind)
                this_neuron = neuron_ind(i);
                if group1(this_neuron)
                    neuron_roles{i} = 'group 1';
                elseif group2(this_neuron)
                    neuron_roles{i} = 'group 2';
                else
                    neuron_roles{i} = 'other';
                end
            end
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
            end
            if ~exist('neuron_ind','var')
                neuron_ind = 0;
            end
            
            % Plot
            [self.user_control_reconstruction, fig] = ...
                self.AdaptiveDmdc_obj.plot_reconstruction(true, ...
                include_control_signal, true, neuron_ind);
            title('Data reconstructed with user-defined control signal')
            
            % Setup interactivity
            if neuron_ind == 0
                [~, im1, ~, im2] = fig.Children.Children;
                im1.ButtonDownFcn = @(x,y) self.callback_plotter(x,y);
                im2.ButtonDownFcn = @(x,y) self.callback_plotter(x,y);
            end
            
        end
        
        function fig = plot_colored_data(self, plot_pca, plot_opt)
            if ~exist('plot_pca','var')
                plot_pca = false;
            end
            if ~exist('plot_opt','var')
                plot_opt = 'o';
            end
            [self.L_sparse_modes,~,~,proj3d] = plotSVD(self.L_sparse,...
                struct('PCA3d',plot_pca,'sigma',false));
            fig = plot_colored(proj3d,...
                self.state_labels_ind_raw(end-size(proj3d,2)+1:end),...
                self.state_labels_key, plot_opt);
            title('Dynamics of the low-rank component (data)')
        end
        
        function fig = plot_colored_user_control(self, fig, use_same_fig)
            % Plots user control data on top of colored original dataset
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
                    self.state_labels_ind_raw(end-size(proj_3d,2)+1:end),...
                    self.state_labels_key);
                title('Dynamics of the low-rank component (reconstruction)')
            end
        end
        
        function fig = plot_colored_fixed_point(self,...
                which_label, use_centroid, fig, use_only_global)
            % Plots the fixed point on top of colored original dataset
            if ~exist('fig','var') || isempty(fig)
                fig = self.plot_colored_data(false, '.');
            end
            if ~exist('use_centroid','var') || isempty(use_centroid)
                use_centroid = false;
            end
            if ~exist('which_label','var') || isempty('which_label')
                which_label = 'all';
            end
            if ~exist('use_only_global','var')
                use_only_global = false;
            end
            
            attractor_reconstruction = self.calc_attractor(...
                which_label, use_only_global);
            
            modes_3d = self.L_sparse_modes(:,1:3);
            proj_3d = (modes_3d.')*attractor_reconstruction;
            if use_centroid
                proj_3d = mean(proj_3d,2);
                plot3(proj_3d(1,:),proj_3d(2,:),proj_3d(3,:), ...
                    'k*', 'LineWidth', 50)
            else
                plot3(proj_3d(1,:),proj_3d(2,:),proj_3d(3,:), ...
                    'ko', 'LineWidth', 6)
            end
                
            title(sprintf(...
                'Fixed points for control structure in %s behavior(s)',...
                which_label))
        end
        
        function fig = plot_user_control_fixed_points(self,...
                which_label, use_centroid, fig)
            % Plots fixed points for manually set control on top of colored
            % original dataset
            assert(~isempty(self.user_control_reconstruction),...
                'No reconstructed data stored')
            if ~exist('fig','var')
                fig = self.plot_colored_data(false, '.');
            end
            if ~exist('use_centroid','var')
                use_centroid = false;
            end
            if ~exist('which_label','var')
                which_label = 'all';
            end
            
            % Get the dynamics and control matrices, and the control signal
            ad_obj = self.AdaptiveDmdc_obj;
            x_dat = 1:ad_obj.x_len;
%             x_ctr = (ad_obj.x_len+1):self.total_sz(1);
            A = ad_obj.A_original(x_dat, x_dat);
            B = self.user_control_matrix;
            u = self.user_control_input;
            if ~strcmp(which_label,'all')
                [~, ~, ~, u_ind] = ...
                    self.get_control_signal_during_label(which_label);
                u = u(:, u_ind);
            end
            % Reconstruct the attractor and project it into the same space
            % Find x (fixed point) in:
            %   x = A x + B u
            attractor_reconstruction = ((eye(length(A))-A)\B)*u;
            
            modes_3d = self.L_sparse_modes(:,1:3);
            proj_3d = (modes_3d.')*attractor_reconstruction;
            if use_centroid
                proj_3d = mean(proj_3d,2);
            end
            plot3(proj_3d(1,:),proj_3d(2,:),proj_3d(3,:), ...
                'k*', 'LineWidth', 1.5)
            title(sprintf(...
                'Fixed points for control structure in %s behavior(s)',...
                which_label))
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
                    arrow_length = 20*ones(size(arrow_length));
                    
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
                attractor_view, use_legend, movie_filename,...
                intrinsic_arrow_mode, show_reconstruction, pause_time)
            % Plots a movie of the 3 different control signal directions:
            %    Intrinsic dynamics
            %    Sparse controller
            %    Global mode
            if ~exist('attractor_view','var') || isempty(attractor_view)
                attractor_view = true;
            end
            if ~exist('use_legend','var') || isempty(use_legend)
                use_legend = false;
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
            if ~exist('pause_time','var')
                pause_time = 0.00;
            end
            fig = self.plot_colored_data(false, 'o');
            [~, b] = fig.Children.Children;
            alpha(b, 0.2)
            if ~use_legend
                legend off;
            end
            
            if ~isempty(movie_filename)
                video_obj = VideoWriter(movie_filename);
                open(video_obj);
            end
            
            num_neurons = self.original_sz(1);
            neuron_ind = 1:num_neurons;
            arrow_factor = 1;
            global_ind = ((2*num_neurons+1):self.total_sz(1)) - ...
                num_neurons;
            this_dat = self.dat_without_control - ...
                mean(self.dat_without_control,2);
            this_ctr_sparse = ...
                self.dat_with_control((num_neurons+1):(2*num_neurons),:);
            this_ctr_global = ...
                self.dat_with_control((2*num_neurons+1):end,:);
            modes_3d = self.L_sparse_modes(:,1:3).';
            if show_reconstruction
                this_reconstruction = ...
                    self.AdaptiveDmdc_obj.calc_reconstruction_control(...
                    [], [], true);
                this_reconstruction = this_reconstruction(neuron_ind,:);
                all_arrow_bases = modes_3d * this_reconstruction;
                all_dat_projected = modes_3d * this_dat;
            else
                all_arrow_bases = modes_3d * this_dat;
            end
            
            for i=1:self.original_sz(2)
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
                arrow_length_sparse = (this_ctr_sparse(:,i)*arrow_factor).';
                self.plot_colored_control_arrow(...
                    neuron_ind, arrow_base, 20*arrow_length_sparse, fig,...
                    false, 'mean', 'r');
                % Global controller
                if attractor_view
                    label_ind = this_ctr_global(1,i);
                    label_str = self.state_labels_key{label_ind};
                    self.plot_colored_fixed_point(label_str, true, fig);
                else
                    arrow_length_global = (this_ctr_global(:,i)*arrow_factor).';
                    self.plot_colored_control_arrow(...
                        global_ind, arrow_base, arrow_length_global, fig,...
                        false, true, 'k');
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
                
                if use_legend
                    leg = legend;
                    leg.String(end-3:end) = ...
                        {'Intrinsic','Sparse','Global','Current Point'};
                end
                if ~isempty(movie_filename)
                    frame = getframe(fig);
                    writeVideo(video_obj, frame);
                end
                pause(pause_time)
                
                c = get(gca,'children');
                if show_reconstruction
                    delete(c(1:5));
                else
                    delete(c(1:4));
                end
            end
            
            if ~isempty(movie_filename)
                close(video_obj);
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
        
        function plot_pareto_front(self)
            assert(~isempty(self.pareto_struct.lambda_vec),...
                'No pareto front data saved')
            
            figure('DefaultAxesFontSize',14);
            y_val = self.pareto_struct.all_errors;
            plot(self.pareto_struct.lambda_vec, y_val,...
                'LineWidth',2)
            xlabel('\lambda value for sparse signals')
            ylabel('L2 error')
            ylim([0, min(10*y_val(1,1),max(max(y_val)))])
            title('Pareto front')
            legend(self.pareto_struct.global_signal_modes,...
                'Interpreter','None')
            
        end
    end
    
    methods (Access=private)
        
        function set_defaults(self)
            defaults = struct(...
                'verbose',true,...
                ...% Getting the control signal
                'lambda_global', 0.0065,...
                'max_rank_global', 4,...
                'lambda_sparse', 0.05,...
                'global_signal_mode', 'RPCA',...
                ...% Data processing
                'filter_window_dat', 3,...
                'filter_window_global', 10,...
                'AdaptiveDmdc_settings', struct(),...
                'augment_data', 0,...
                'to_subtract_mean',false,...
                'to_subtract_mean_sparse', true,...
                'to_subtract_mean_global', true,...
                'dmd_mode', 'naive',...
                ...% Data importing
                'use_deriv', false,...
                'use_only_deriv', false,...
                'to_normalize_deriv', false);
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
            
            self.A_old = [];
            self.dat_old = [];
        end
        
        function import_from_struct(self, Zimmer_struct)
            warning('Assumes struct of Zimmer type')
            % First, get data and maybe derivatives
            self.raw = Zimmer_struct.traces.';
            if self.use_deriv || self.use_only_deriv
                self.raw_deriv = Zimmer_struct.tracesDif.';
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
            if ~any(ismember(fieldnames(self.AdaptiveDmdc_settings),...
                    {'id_struct','sort_mode','x_indices','dmd_mode',...
                    'to_plot_nothing'}))
                if ~self.use_deriv
                    x_ind = 1:size(self.raw,1);
                else
                    x_ind = 1:(2*size(self.raw,1));
                end
                id_struct = struct('ID',{Zimmer_struct.ID},...
                    'ID2',{Zimmer_struct.ID2},'ID3',{Zimmer_struct.ID3});
                self.AdaptiveDmdc_settings.id_struct = id_struct;
                self.AdaptiveDmdc_settings.sort_mode = 'user_set';
                self.AdaptiveDmdc_settings.x_indices = x_ind;
                self.AdaptiveDmdc_settings.dmd_mode = self.dmd_mode;
                self.AdaptiveDmdc_settings.to_plot_nothing = true;
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
                new_sz = [self.dat_sz(1)*aug, self.dat_sz(2)-aug];
                new_dat = zeros(new_sz);
                for j=1:aug
                    old_cols = j:(new_sz(2)+j-1);
                    new_rows = (1:self.dat_sz(1))+self.dat_sz(1)*(j-1);
                    new_dat(new_rows,:) = self.raw(:,old_cols);
                end
                self.dat_sz = new_sz;
                self.raw = new_dat;
                self.AdaptiveDmdc_settings.x_indices = 1:size(new_dat,1);
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
            
            self.pareto_struct = struct();
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
            
            switch self.global_signal_mode
                case 'RPCA'
                    % Gets VERY low-rank signal
                    %   Loop to check for rank convergence if 
                    %   self.max_rank_global > 0
                    iter_max = 10;
                    this_lambda = self.lambda_global;
                    for i=1:iter_max
                        if self.max_rank_global==0
                            break;
                        end
                        [~, ~,...
                            self.L_global_rank, ~] = ...
                            RobustPCA(self.dat, this_lambda, 10*this_lambda,...
                            1e-6, 70);
                        if self.L_global_rank <= self.max_rank_global
                            fprintf("Reached target rank (%d); restarting with more accuracy\n",...
                                    self.max_rank_global);
                            self.lambda_global = this_lambda;
                            break
                        else
                            % A smaller penalty means more data in the sparse
                            % component, less in the low-rank
                            if self.verbose
                                fprintf("Didn't reach target rank (%d); decreasing lambda\n",...
                                    self.max_rank_global);
                            end
                            if i==iter_max
                                warning("Didn't converge on maximum rank")
                            end
                            this_lambda = this_lambda*0.9;
                        end
                    end
                    % Get a more accurate decomposition (even if we didn't converge
                    % to the proper rank)
                    [self.L_global_raw, self.S_global_raw,...
                        self.L_global_rank, self.S_global_nnz] = ...
                        RobustPCA(self.dat, self.lambda_global);
                    % Smooth the modes out
                    self.L_global = self.flat_filter(...
                        self.L_global_raw.', self.filter_window_global).';
                    tmp_dat = self.L_global - mean(self.L_global,2);
                    [u, ~] = svd(tmp_dat(:,self.filter_window_global+5:end).');
                    x = 1:rank(self.L_global);
                    self.L_global_modes = u(:,x);

                    self.S_global = self.S_global_raw;
                
                case 'ID'
                    self.L_global_modes = [self.state_labels_ind_raw;...
                        ones(size(self.state_labels_ind_raw));
                        log(1:length(self.state_labels_ind_raw))].';
                    
                case 'ID_binary'
                    tmp = self.state_labels_ind_raw;
                    all_states = unique(tmp);
                    sz = [length(all_states), 1];
                    binary_labels = zeros(sz(1),size(tmp,2));
                    for i = 1:sz(1)
                        binary_labels(i,:) = (tmp==all_states(i));
                    end
                    self.L_global_modes = [binary_labels;...
                        ones(1,size(binary_labels,2));
                        log(1:size(tmp,2))].';
                    
                case 'ID_simple'
                    tmp = self.state_labels_ind_raw;
                    states_dict = containers.Map(...
                        {1,2,3,4,5,6,7,8},...
                        {-1,-1,0,0,0,0,1,0});
                    for i=1:length(tmp)
                        tmp(i) = states_dict(tmp(i));
                    end
                    self.L_global_modes = [tmp; ones(size(tmp))].';
                    
                case 'ID_and_ID_simple'
                    tmp = self.state_labels_ind_raw;
                    states_dict = containers.Map(...
                        {1,2,3,4,5,6,7,8},...
                        {-1,-1,0,0,0,0,1,0});
                    for i=1:length(tmp)
                        tmp(i) = states_dict(tmp(i));
                    end
                    self.L_global_modes = ...
                        [tmp,...
                        self.state_labels_ind_raw];
                    
                    self.L_global_modes = self.L_global_modes.';
                    
                case 'ID_and_binary'
                    self.global_signal_mode = 'ID_binary';
                    self.calc_global_signal();
                    self.L_global_modes = [self.L_global_modes, ...
                        self.state_labels_ind_raw.'];
                    self.global_signal_mode = 'ID_and_binary';
                    
                case 'ID_and_offset'
                    tmp = self.state_labels_ind_raw;
                    self.L_global_modes = tmp;
                    for i=1:length(unique(tmp))
                        self.L_global_modes = ...
                            [self.L_global_modes;
                            tmp - i];
                    end
                    self.L_global_modes = [self.L_global_modes; 
                        log(1:size(tmp,2))].';
                    
                otherwise
                    error('Unrecognized method')
            end
        end
        
        function calc_sparse_signal(self)
            % Calculates very sparse signal
            [self.L_sparse_raw, self.S_sparse_raw,...
                self.L_sparse_rank, self.S_sparse_nnz] = ...
                RobustPCA(self.dat, self.lambda_sparse);
            % For now, no processing
            self.L_sparse = self.L_sparse_raw;
            self.S_sparse = self.S_sparse_raw;
        end
        
        function calc_dat_and_control_signal(self)
            % Uses results from 2 different Robust PCA runs
            
            % Data to be reconstructed is everything EXCEPT the sparse
            % control signals
            if self.to_subtract_mean_sparse
                this_dat = self.L_sparse - mean(self.L_sparse,2);
                sparse_signal = self.S_sparse - mean(self.S_sparse,2);
            else
                this_dat = self.L_sparse;
                sparse_signal = self.S_sparse;
            end
            % Sparse signal with thresholding
%             tol = 1e-2;
%             sparse_signal = sparse_signal(max(abs(sparse_signal),[],2)>tol,:);
            % Use top svd modes for the low-rank component
            if self.to_subtract_mean_global
                L_low_rank = self.L_global_modes - mean(self.L_global_modes,1);
            else
                L_low_rank = self.L_global_modes;
            end
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
        
        function callback_plotter(self, ~, evt)
            % On left click:
            %   Plots the original data and the reconstruction
            % On right click:
            %   Prints neuron name
            this_neuron = round(evt.IntersectionPoint(2));
            if evt.Button==1
                self.AdaptiveDmdc_obj.plot_reconstruction(true, ...
                    true, true, this_neuron);
            else
                self.AdaptiveDmdc_obj.get_names(this_neuron);
            end
        end
        
    end
    
    methods (Static)
        function dat = flat_filter(dat, window_size)
            w = window_size;
            dat = filter(ones(w,1)/w,1,dat);
        end
    end
    
end

