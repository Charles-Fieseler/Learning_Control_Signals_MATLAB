classdef AdaptiveDmdc < AbstractDmd
    % Adaptive DMD with control
    %   In a heuristic way, learns which data entries are acting as controllers
    %   for the rest of the system, and separates them out as 'u_indices'
    %
    % INPUTS
    %   INPUT1 -
    %   INPUT2 -
    %
    % OUTPUTS -
    %   OUTPUT1 -
    %   OUTPUT2 -
    %
    %
    % Dependencies
    %   .m files, .mat files, and MATLAB products required:(updated on 21-Feb-2020)
    %         MATLAB (version 9.4)
    %         Signal Processing Toolbox (version 8.0)
    %         checkModes.m
    %         func_DMD.m
    %         func_DMDc.m
    %         sparse_dmd_fast.m
    %         AbstractDmd.m
    %         SettingsImportableFromStruct.m
    %         optimal_truncation.m
    %         svd_truncate.m
    %         plotSVD.m
    %         plot_2imagesc_colorbar.m
    %         optimal_SVHT_coef.m
    %
    %
    %   See also: OTHER_FUNCTION_NAME
    %
    %
    %
    % Author: Charles Fieseler
    % University of Washington, Dept. of Physics
    % Email address: charles.fieseler@gmail.com
    % Website: coming soon
    % Created: 11-Mar-2018
    %========================================
    
    properties (SetAccess={?SettingsImportableFromStruct}, Hidden=true)
        % Detection of control neurons (can be set manually)
        sort_mode
        u_sort_ind
        x_indices
        id_struct
        % External matrix of dynamics
        external_A_orig
        % Plotting options
        to_plot_nothing
        to_plot_cutoff
        to_plot_data
        to_plot_A_matrix_svd
        which_plot_data_and_filter
        to_plot_data_and_outliers
        
        % DMD options
        truncation_rank
        truncation_rank_control
        dmd_mode
        dmd_offset
        what_to_do_dmd_explosion
        oscillation_threshold
        sparsity_goal % if dmd_mode==sparse
        initial_sparsity_pattern
        sparsity_mode
        to_print_error
        % Outlier calculation settings
        error_tol
        min_number_outliers
        cutoff_multiplier
        % Preprocessing settings
        to_normalize_envelope
        filter_window_size
        outlier_window_size
        % Augmentation settings
        data_already_augmented
        to_augment_error_signals
        % Tolerance for sparsification
        sparse_tol
        sparse_tol_factor
        
        % For memory management
        to_save_raw_data
        
        % Cross validation settings
        hold_out_fraction
        cross_val_window_size_percent
    end
    
    properties (SetAccess=public, Hidden=false)
        u_indices
        sep_error
        original_error
        error_outliers
        neuron_errors
        % DMD propagators
        A_original
        A_separate
        % Sparsified versions of A
        A_thresholded
        use_A_thresholded
        
        % Control signal properties
        x_len
        u_len
        
        % Cross validation
        dat_cross_val
    end
    
    properties (Dependent)
        error_mat
    end
    
    methods
        function self = AdaptiveDmdc( file_or_dat, settings )
            % Creates adaptive_dmdc object using the filename or data
            % matrix (neurons=rows, time=columns) in file_or_dat.
            % The settings struct can have many fields, as explained in the
            % full help command.
            %
            % This initializer runs the following functions:
            %   import_settings_to_self(settings);
            %   preprocess();
            %   calc_data_outliers();
            %   calc_outlier_indices();
            %   calc_dmd_and_errors();
            %   plot_using_settings();
            %
            % And optionally:
            %   augment_and_redo();
            
            %% Set defaults and import settings
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
            self.use_A_thresholded = false;
            if self.to_plot_nothing
                self.turn_plotting_off(); %Overrides other options
            end
            %==========================================================================
            
            %---------------------------------------------
            %% Import data
            %---------------------------------------------
            if ischar(file_or_dat)
                %     self.filename = file_or_dat;
                self.raw = importdata(file_or_dat);
                %     if isstruct(tmp_struct)
                %         self.import_from_struct(tmp_struct);
                %     else
                %         error('Filename must contain a struct')
                %     end
            elseif isnumeric(file_or_dat)
                self.raw = file_or_dat;
            else
                error('Must pass data matrix or filename')
            end
            self.preprocess();
            
            %---------------------------------------------
            %% Get outlier indices
            %---------------------------------------------
            % Note: sets the variable 'x_indices' (not logical)
            % Also reorders the data so that the outliers are on the bottom
            % rows
            self.calc_data_outliers();
            self.calc_outlier_indices();
            
            %---------------------------------------------
            %% Do normal DMD and 'separated DMD'; plot
            %---------------------------------------------
            self.calc_dmd_and_errors();
            self.plot_using_settings();
            if self.sparse_tol > 0
                self.set_A_thresholded(self.sparse_tol);
            end
            if self.to_augment_error_signals
                self.augment_and_redo();
            end
            
            if self.verbose
                fprintf('Finished analyzing\n')
            end
        end
        
        function set_defaults(self)
            
            defaults = struct(...
                'sort_mode', 'DMD_error_outliers',...
                'to_plot_nothing', true,...
                'to_plot_cutoff', true,...
                'to_plot_data', true,...
                'to_plot_A_matrix_svd', false,...
                'which_plot_data_and_filter', 0,...
                'to_plot_data_and_outliers', false,...
                'truncation_rank', 0,...
                'truncation_rank_control', 0,...
                'dmd_mode', 'naive',...
                'dmd_offset', 1,...
                'what_to_do_dmd_explosion','project',...
                'oscillation_threshold', 0,...
                'external_A_orig', [],...
                'sparsity_goal', 0.6,...
                'initial_sparsity_pattern', logical([]),...
                'sparsity_mode', 'threshold',...
                'cutoff_multiplier', 1.0,...
                'to_print_error', false, ...
                'error_tol', 1e-8,...
                'min_number_outliers', 2,...
                'to_subtract_mean', false,...
                'to_normalize_envelope', false,...
                'to_augment_error_signals', false,...
                'data_already_augmented', 0, ...
                'filter_window_size', 10,...
                'outlier_window_size', 2000,...
                'id_struct', struct(),...
                'sparse_tol', 0,...
                'sparse_tol_factor', 1,...
                'to_save_raw_data', true,...
                'hold_out_fraction', 0,...
                'cross_val_window_size_percent', 1);
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
        end
        
        function calc_data_outliers(self)
            % Calculate which neurons have nonlinear features and should be
            % separated out as controllers by various methods:
            %
            %   sparsePCA: take out the nodes that have high loading on the
            %       first 15 pca modes
            %
            % The following methods all solve the DMD problem (x2=A*x1),
            % subtract off the resulting linear fit, and then calculate
            % features based on the residuals
            %   DMD_error: choose neurons with the highest L2 error
            %   DMD_error_exp: choose neurons with the highest error
            %       calculated using an exponential (length scale,
            %       'lambda', set in the initial options)
            %   DMD_error_normalized: choose neurons with high L2 error,
            %       normalized by their median activity
            %   DMD_error_outliers: (default) choose neurons with high
            %       'outlier error,' which is the L2 norm of only outlier
            %       points (>=3 std dev away) weighted by 1/distance to
            %       neighbors (i.e. clusters are weighted more strongly)
            %
            %   random: randomly selects neurons; for benchmarking
            %   user_set: The user sets the 'x_indices' setting manually
            %       (Note: also requires setting 'sort_mode')
            %
            % Note: if the setting 'to_plot_cutoff' is true, then this
            % function plots the errors of each neuron and the cutoffs used
            % as an interactive graph
            
            X = self.dat;
            self.error_outliers = [];
            
            switch self.sort_mode
                case 'sparsePCA'
                    %---------------------------------------------
                    % Do sparse PCA to sort nodes
                    %---------------------------------------------
                    cardinality = 50;
                    num_modes = 15;
                    [modes, loadings] = sparsePCA(X',...
                        cardinality, num_modes, 2, 0);
                    loading_log_cutoff = 4;
                    loading_cutoff_index = find(...
                        log(abs(diff(loadings)))<loading_log_cutoff,1) - 1;
                    
                    % Plot the cutoff as sanity check
                    if self.to_plot_cutoff
                        figure;
                        plot(loadings)
                        hold on;
                        plot(loading_cutoff_index, loadings(loading_cutoff_index), 'r*')
                        title('sparsePCA cutoff')
                    end
                    
                    x_ind = [];
                    for j=1:loading_cutoff_index
                        x_ind = [x_ind; find(modes(:,j)>0)]; %#ok<AGROW>
                    end
                    x_ind = unique(x_ind);
                    
                case 'DMD_error'
                    X1_original = X(:,1:end-1);
                    X2_original = X(:,2:end);
                    A = X2_original / X1_original;
                    
                    self.neuron_errors = mean((A*X1_original-X2_original).^2,2);
                    cutoff_val = (mean(self.neuron_errors) + ...
                        std(self.neuron_errors)*self.cutoff_multiplier);
                    x_ind = find(self.neuron_errors < cutoff_val);
                    
                    % Plot the cutoff as sanity check
                    %                     if self.to_plot_cutoff
                    %                         figure;
                    %                         plot(self.neuron_errors)
                    %                         hold on;
                    %                         vec = ones(size(self.neuron_errors'));
                    %                         plot(cutoff_val*vec, 'r')
                    %                         title('DMD reconstruction error cutoff')
                    %                     end
                    
                case 'DMD_error_exp'
                    lambda = 0.05; % For Zimmer data
                    X1_original = X(:,1:end-1);
                    X2_original = X(:,2:end);
                    A = X2_original / X1_original;
                    
                    self.neuron_errors = sum(exp(...
                        abs(A*X1_original-X2_original)/lambda),2);
                    
                    cutoff_val = (mean(self.neuron_errors) + ...
                        std(self.neuron_errors)*self.cutoff_multiplier);
                    x_ind = find(self.neuron_errors < cutoff_val);
                    
                    % Plot the cutoff as sanity check
                    %                     if self.to_plot_cutoff
                    %                         figure;
                    %                         plot(self.neuron_errors)
                    %                         hold on;
                    %                         vec = ones(size(self.neuron_errors'));
                    %                         plot(cutoff_val*vec, 'r')
                    %                         title('DMD reconstruction error cutoff')
                    %                     end
                    
                case 'DMD_error_normalized'
                    X1_original = X(:,1:end-1);
                    X2_original = X(:,2:end);
                    A = X2_original / X1_original;
                    
                    neuron_errors_tmp = mean((A*X1_original-X2_original).^2,2);
                    self.neuron_errors = neuron_errors_tmp./(mean(X,2)+mean(mean(X)));
                    
                    cutoff_val = ( mean(self.neuron_errors) + ...
                        std(self.neuron_errors)*self.cutoff_multiplier );
                    x_ind = find(self.neuron_errors < cutoff_val);
                    % Plot the cutoff as sanity check
                    %                     if self.to_plot_cutoff
                    %                         figure;
                    %                         plot(self.neuron_errors)
                    %                         hold on;
                    %                         vec = ones(size(self.neuron_errors'));
                    %                         plot(cutoff_val*vec, 'r')
                    %                         title('DMD reconstruction error cutoff')
                    %                     end
                    
                case 'DMD_error_outliers'
                    X1_original = X(:,1:end-1);
                    X2_original = X(:,2:end);
                    A = X2_original / X1_original;
                    self.neuron_errors = A*X1_original-X2_original;
                    
                    if max(max(self.neuron_errors)) < self.error_tol
                        warning('Fitting errors lower than tolerance; no control signals can be reliably identified')
                        x_ind = true(size(self.neuron_errors,1),1);
                        self.error_outliers = zeros(size(x_ind));
                    else
                        self.error_outliers = self.calc_error_outliers(self.neuron_errors, ...
                            self.filter_window_size, self.outlier_window_size);
                        x_ind = find(~isoutlier(self.error_outliers,...
                            'ThresholdFactor', self.cutoff_multiplier));
                    end
                    
                case 'DMD_error_outliers_sparse'
                    % External function that uses cvx
                    error('Option not supported')
                    %                     error('Need to update the syntax here')
                    %                     [ A_sparse, this_error ] = ...
                    %                         sparse_dmd( X, min_tol, max_error_mult );
                    %
                    %                     self.neuron_errors = this_error;
                    %
                    %                     if max(max(self.neuron_errors)) < self.error_tol
                    %                         warning('Fitting errors lower than tolerance; no control signals can be reliably identified')
                    %                         x_ind = true(size(self.neuron_errors,1),1);
                    %                         self.error_outliers = zeros(size(x_ind));
                    %                     else
                    %                         self.error_outliers = self.calc_error_outliers(self.neuron_errors, ...
                    %                             self.filter_window_size, self.outlier_window_size);
                    %                         x_ind = find(~isoutlier(self.error_outliers,...
                    %                             'ThresholdFactor', self.cutoff_multiplier));
                    %                     end
                    
                case 'random'
                    tmp = randperm(size(X,1));
                    x_ind = tmp(1:round(size(X,1)-...
                        self.min_number_outliers*self.cutoff_multiplier));
                    
                case 'user_set'
                    x_ind = self.x_indices;
                    
                otherwise
                    error('Sort mode not recognized')
            end
            
            self.x_len = length(find(x_ind));
            
            if self.to_plot_cutoff
                self.plot_data_and_outliers(self.error_outliers,[],true);
                title('Error signal detected and threshold')
            end
            
            if isempty(self.error_outliers)
                % Just for later plotting
                X1_original = X(:,1:end-1);
                X2_original = X(:,2:end);
                A = X2_original / X1_original;
                self.neuron_errors = A*X1_original-X2_original;
                self.error_outliers = self.calc_error_outliers(self.neuron_errors, ...
                    self.filter_window_size, self.outlier_window_size);
            end
            
            self.x_indices = x_ind;
        end
        
        function calc_outlier_indices(self)
            % Given indices that should be used for the data, calculate and
            % save the controller indices as well (the complement of the
            % data indices).
            %   Important: also sorts the data matrix so that the
            %   controllers are the last rows
            
            x_ind = self.x_indices;
            X = self.dat;
            %---------------------------------------------
            % Finish analyzing the x_indices variable
            %---------------------------------------------
            u_ind = true(size(X,1),1);
            u_ind(x_ind) = false;
            
            u_length = length(find(u_ind));
            
            % Sort the data: control signals in last columns
            [~, self.u_sort_ind] = sort(u_ind);
            X = X(self.u_sort_ind,:);
            self.dat = X;
            
            self.u_len = u_length;
            self.u_indices = u_ind;
            
        end
        
        function calc_dmd_and_errors(self)
            % Actually performs dmd and calculates errors
            %
            % The basic DMD algorithm solves for _A_ in the equation:
            %   $$ x2 = A*x1 $$
            % where the original data matrix, _X_, has been split:
            %   $$ x2 = X(:,2:end) $$
            %   $$ x1 = X(:,1:end-1) $$
            %
            % DMD with control adds an additional layer, and solves for _A_
            % and _B_ in the equation:
            %   $$ x2 = A*x1 + B*u $$
            % where _u_ is the control signal and _B_ is the connection
            % between the control signal and the dynamics in the data. In
            % this class, the control signal is taken to be certain rows
            % of the data as identified by the setting 'sort_mode'
            %
            % TODO: implement better and less biased algorithms for DMD
            
            x_length = self.x_len;
            u_length = self.u_len;
            X = self.dat;
            X1_original = X(:,1:end-self.dmd_offset);
            X2_original = X(:,(self.dmd_offset+1):end);
            
            switch lower(self.dmd_mode)
                case 'naive'
                    A_orig = X2_original / X1_original;
                    
                    % Set points corresponding to u on the LHS to 0
                    % X2_sep(u_indices,:) = 0;
                    % Easier with sorted data
                    sz = size(X2_original);
                    X2_sep = [X2_original(1:x_length,:);
                        zeros(u_length,sz(2)) ];
                    A_sep = X2_sep / X1_original;
                    
                case 'sparse'
                    error("Option 'sparse' is deprecated; use 'sparse_fast'")
                    % MUCH stronger sparsification on the control matrix
                    %                     p.min_tol = {{8e-3}, {5e-2}};
                    %                     p.tol2column_cell = {...
                    %                         {1:x_length}, ...
                    %                         {(x_length+1):(x_length+u_length)}};
                    %                     p.max_error_mult = 1.25;
                    %                     p.column_mode = true;
                    %                     p.rows_to_predict = 1:self.x_len;
                    %                     p.verbose = true;
                    %                     p.error_func = @(A, ~, ~, ~) ...
                    %                         self.update_and_reconstruction_error(A,'Inf');
                    %                     p.sparsity_goal = self.sparsity_goal;
                    %                     p.initial_sparsity_pattern = ...
                    %                         self.initial_sparsity_pattern;
                    %                     p.sparsity_mode = self.sparsity_mode;
                    %                     p.max_iter = 10;
                    %                     if self.truncation_rank == 0
                    %                         % Do not want to SVD reconstruct the control signal
                    %                         Upsilon = X((x_length+1):end,:);
                    %                         X_no_ctr = X(1:x_length,:);
                    %                         [~, ~, X_no_ctr] = optimal_truncation(X_no_ctr);
                    %                         X = [X_no_ctr; Upsilon];
                    %                     end
                    %
                    %                     [ A_orig, ~ ] = ...
                    %                         sparse_dmd(X, p);
                    %
                    %                     % Set points corresponding to u on the LHS to 0
                    %                     % X2_sep(u_indices,:) = 0;
                    %                     % Easier with sorted data
                    %                     sz = size(X2_original);
                    %                     X2_sep = [X2_original(1:x_length,:);
                    %                         zeros(u_length,sz(2)) ];
                    % %                     [ A_sep, ~ ] = ...
                    % %                         sparse_dmd( {X1_original, X2_sep},...
                    % %                         min_tol, max_error_mult, column_mode );
                    %                     A_sep = [A_orig(1:x_length,:);...
                    %                         zeros(self.u_len, size(A_orig,2))];
                    
                case 'sparse_fast'
                    Upsilon = X((x_length+1):end,1:end-1);
                    X_no_ctr = X(1:x_length,:);
                    dat = {[X_no_ctr(:,1:end-1);Upsilon], ...
                        X_no_ctr(:,2:end)};
                    
                    A = dat{2}/dat{1};
                    if self.data_already_augmented > 0
                        x = x_length/self.data_already_augmented;
                        partial_A = A(1:x,1:x);
                    else
                        partial_A = A(1:x_length,1:x_length);
                    end
                    settings = struct( 'threshold', ...
                        median(median(abs(partial_A))) *...
                        self.sparse_tol_factor );
                    [A_orig] = sparse_dmd_fast(dat, settings);
                    A_sep = [A_orig(1:x_length,:);...
                        zeros(self.u_len, size(A_orig,2))];
                    
                case 'optdmd'
                    error('Not implemented')
                    % Note: default rank is high, so it takes a while
                    %                     t = (1:size(X,2))';
                    %                     if self.truncation_rank == 0
                    %                         r = optimal_truncation(X2_original(1:x_length,:));
                    %                     else
                    %                         r = self.truncation_rank;
                    %                     end
                    %                     opt = varpro_opts('maxiter', 500);
                    %                     [~,~,~,~,~,A_orig] = optdmd(X,t,r, 1, opt);
                    %                     A_sep = [A_orig(1:x_length,:);...
                    %                         zeros(u_length, size(A_orig,2))];
                    
                case 'func_dmdc'
                    % Uses Zhe's code
                    %   Note: have to decide on a truncation rank for both
                    %   the dynamics and the control signal
                    X1 = X1_original(1:x_length,:);
                    X2 = X2_original(1:x_length,:);
                    Upsilon = X1_original((x_length+1):end,:);
                    if self.truncation_rank == 0
                        r = optimal_truncation(X2);
                        self.truncation_rank = r;
                        rtilde = self.truncation_rank_control;
                    elseif self.truncation_rank == -1
                        r = x_length;
                        rtilde = u_length + x_length;
                    else
                        r = self.truncation_rank;
                        rtilde = self.truncation_rank_control;
                    end
                    [~, ~, Bhat, ~, Uhat, ~, ~, ~, ~, ~, Atilde] = ...
                        func_DMDc(X1, X2, Upsilon, r, rtilde);
                    
                    U = Uhat(:,1:r);
                    A = U*Atilde*U';
                    % Concatenating them is the format this object expects
                    A_orig = [A Bhat];
                    A_orig = [A_orig; zeros(u_length, size(A_orig,2))];
                    A_sep = A_orig;
                    
                case 'truncated_dmdc'
                    % Uses Zhe's code
                    %   Note: have to decide on a truncation rank for both
                    %   the dynamics and the control signal
                    X1 = X1_original(1:x_length,:);
                    X2 = X2_original(1:x_length,:);
                    Upsilon = X1_original((x_length+1):end,:);
                    if self.truncation_rank == 0
                        r = optimal_truncation(X2);
                        self.truncation_rank = r;
                        rtilde = self.truncation_rank_control;
                    elseif self.truncation_rank == -1
                        r = x_length;
                        rtilde = u_length;
                    else
                        r = self.truncation_rank;
                        rtilde = self.truncation_rank_control;
                    end
                    U_truncate = svd_truncate(Upsilon, rtilde);
                    X1 = [svd_truncate(X1, r); U_truncate];
                    X2 = [svd_truncate(X2, r); U_truncate];
                    
                    A_orig = X2/X1;
                    A_sep = [A_orig(1:x_length,:); zeros(u_length, size(A_orig,2))];
                    
                case 'no_dynamics'
                    % Uses Zhe's code
                    %   Note: have to decide on a truncation rank for both
                    %   the dynamics and the control signal
                    X2 = X2_original(1:x_length,:);
                    X1 = zeros(size(X2));
                    Upsilon = X1_original((x_length+1):end,:);
                    if self.truncation_rank == 0
                        r = optimal_truncation(X2);
                        self.truncation_rank = r;
                        rtilde = self.truncation_rank_control;
                    elseif self.truncation_rank == -1
                        r = x_length;
                        rtilde = u_length;
                    else
                        r = self.truncation_rank;
                        rtilde = self.truncation_rank_control;
                    end
                    [~, ~, Bhat, ~, Uhat, ~, ~, ~, ~, ~, Atilde] = ...
                        func_DMDc(X1, X2, Upsilon, r, rtilde);
                    
                    A = zeros(size(Uhat));
                    % Concatenating them is the format this object expects
                    A_orig = [A Bhat];
                    A_orig = [A_orig; zeros(u_length, size(A_orig,2))];
                    A_sep = A_orig;
                    
                case 'no_dynamics_sparse'
                    warning('Assuming the initial sparsity enforces no dynamics')
                    self.dmd_mode = 'sparse';
                    self.calc_dmd_and_errors();
                    self.dmd_mode = 'no_dynamics_sparse';
                    return
                    
                case 'tdmd'
                    % Use Total DMD (aka total least squares dmd), sending
                    % in the entire data + controllers
                    if self.truncation_rank == 0
                        r = optimal_truncation(X1_original);
                    else
                        r = self.truncation_rank;
                    end
                    X1 = X1_original(1:x_length,:);
                    X2 = X2_original(1:x_length,:);
                    Upsilon = X1_original((x_length+1):end,:);
                    [A,B] = tdmdc(X1,X2,r, Upsilon);
                    A_orig = [A, B];
                    A_sep = [A_orig; zeros(self.u_len, size(A_orig,2))];
                    %                     error('Not working... I think I need to separate out the controller')
                    
                case 'external'
                    % Uses externally defined dynamics
                    A_orig = self.external_A_orig;
                    %                     self.x_len = size(A_orig,1);
                    %                     self.u_len = size(self.dat,1) - self.x_len;
                    %                     x_length = self.x_len;
                    %                     u_length = self.u_len;
                    A_sep = [A_orig(1:x_length,:);...
                        zeros(u_length, size(A_orig,2))];
                    assert(size(A_orig,1)==size(A_orig,2),...
                        'External intrinsic dynamics matrix must be square')
                    assert(size(A_orig,1)==size(self.dat,1),...
                        'External intrinsic dynamics matrix must be able to multiply the data')
                    
                otherwise
                    error('Unrecognized dmd mode')
            end
            
            % Calculate the eigenvalues (need all because they might not
            % converge)
            A = A_orig(1:x_length,1:x_length);
            [V, D] = eig(A, 'vector');
            if self.oscillation_threshold > 0
                % Extremely slow oscillations are probably numerical
                % instabilities
                for i = 1:length(D)
                    if ~isreal(D(i)) && ...
                            abs(angle(D(i))) < self.oscillation_threshold
                        D(i) = abs(D(i));
                    end
                end
            end
            if max(abs(D)) > 1.0
                % i.e. there are unstable eigenvalues!
                switch self.what_to_do_dmd_explosion
                    case 'ignore'
                        warning('Ignoring eigenvalue>1; not recommended')
                        
                    case 'error'
                        error('DMD eigenvalue>1; no useful prediction possible')
                        
                    case 'project'
                        warning('Projecting an unstable eigenvalue onto unit circle')
                        D = D./(max(abs(D)+1e-4,ones(size(D))));
                        
                    case 'shrink'
                        warning('Shrinking an unstable eigenvalue to 0')
                        tmp = abs(D);
                        D(tmp>1) = 0;
                        
                    otherwise
                        error('Unrecognized method for dealing with eigenvalue > 1')
                end
                % Implement eigenvalue changes, keeping sparsity pattern
                sparsity_pattern = abs(A) < 1e-10;
                A = V*diag(D)/V;
                A(sparsity_pattern) = 0;
            end
            % Don't change the control matrix, B
            A_orig(1:x_length,1:x_length) = A;
            A_sep(1:x_length,1:x_length) = A;
            
            self.A_original = A_orig;
            self.A_separate = A_sep;
            
            % Note that the separated DMD is attempting to reconstruct a smaller set of
            % data
            if ~exist('X2_sep','var')
                sz = size(X2_original);
                X2_sep = [X2_original(1:x_length,:);
                    zeros(u_length,sz(2)) ];
            end
            self.original_error = norm(self.error_mat)/numel(X2_original);
            self.sep_error = ...
                norm(A_sep*X1_original-X2_sep)/(x_length*size(X2_sep,2));
        end
        
        function augment_and_redo(self)
            % Augments the data with the error signals from the first
            % run-through, and then redoes the analysis
            
            % Current dat has no pseudo-neurons... set it as the new data
            % to reconstruct
            self.x_indices = 1:size(self.dat,1);
            self.sort_mode = 'user_set';
            self.min_number_outliers = self.u_len;
            augmented_dat = [self.dat(:,2:end);...
                self.error_mat(self.x_len+1:end,:)];
            
            % Set the new data, and redo all initializer steps
            self.raw = augmented_dat;
            self.preprocess();
            self.calc_data_outliers();
            self.calc_outlier_indices();
            self.calc_dmd_and_errors();
            self.plot_using_settings();
        end
        
        function error_outliers = calc_error_outliers(self, neuron_errors,...
                filter_window_size, outlier_window_size)
            
            error_outliers = zeros(size(neuron_errors,1),1);
            tspan = 1:size(neuron_errors,2);
            
            for i = 1:size(neuron_errors,1)
                this_error = neuron_errors(i,:)';
                this_error = filtfilt(ones(filter_window_size,1)/filter_window_size,...
                    1, this_error);
                %             this_error = this_error/var(this_error);
                if self.to_normalize_envelope
                    [up_env,low_env] = envelope(this_error);
                    this_error = this_error /...
                        mean(abs([mean(up_env) mean(low_env)]));
                end
                outlier_indices = isoutlier(this_error,...
                    'movmedian',outlier_window_size,'SamplePoints',tspan);
                % 2 factors increasing the importance of outliers:
                %   magnitude (i.e. L2 error... first term)
                %   clusters (i.e. shorter distance between neighbors... second term)
                this_pts = this_error(outlier_indices);
                if length(find(outlier_indices))<self.min_number_outliers
                    error_outliers(i) = 0;
                    continue
                end
                if length(find(outlier_indices))>1
                    n_dist = diff(find(outlier_indices));
                    neighbor_weights = [n_dist; n_dist(end)] + [n_dist(1); n_dist];
                    
                    error_outliers(i) = sum( (this_pts .* (1./neighbor_weights)).^2 );
                else
                    error_outliers(i) = sum( 2*(this_pts.^2) / length(tspan) );
                end
            end
        end
        
        function pruned_ad_obj = prune_outliers(self, error_tol)
            % Prunes control signals from the list greedily as long as the
            % reconstruction error is below tolerance
            initial_error = self.calc_reconstruction_error();
            if ~exist('error_tol','var')
                error_tol = 2*initial_error;
            elseif error_tol<initial_error
                warning('Error tolerance below current error; no pruning possible')
                pruned_ad_obj = [];
                return
            end
            
            fprintf('Initial error is %f\n', initial_error);
            test_settings = self.settings;
            test_settings.sort_mode = 'user_set';
            test_settings.to_plot_nothing = true;
            
            new_errors = zeros(self.u_len,1);
            for i=1:self.u_len
                % Remove current control nodes 1 by 1; get new error
                %   Note: self.dat is sorted with control signals at
                %   the bottom
                test_settings.x_indices = [1:self.x_len self.x_len+i];
                this_ad_obj = adaptive_dmdc(self.dat, test_settings);
                new_errors(i) = this_ad_obj.calc_reconstruction_error();
                fprintf('New error is %f\n', new_errors(i));
            end
            [~, sorted_error_ind] = sort(new_errors);
            
            % Definitely remove the least error-contributing signal
            test_x_indices = [1:self.x_len ...
                self.x_len+sorted_error_ind(1)];
            for i = 2:self.u_len
                this_indices = [test_x_indices ...
                    self.x_len+sorted_error_ind(i)];
                test_settings.x_indices = this_indices;
                this_ad_obj = adaptive_dmdc(self.dat, test_settings);
                
                this_error = this_ad_obj.calc_reconstruction_error();
                if this_error > error_tol
                    fprintf('Final error: %f\n',...
                        pruned_ad_obj.calc_reconstruction_error())
                    fprintf('Final control set size: %d\n',...
                        pruned_ad_obj.u_len);
                    break
                else
                    test_x_indices = this_indices;
                    pruned_ad_obj = this_ad_obj;
                end
            end
            
            
        end
        
        function names = get_names(self, neuron_ind, ...
                use_original_order, print_names, print_warning, ...
                to_parse_names)
            % Gets and optionally prints the names of the passed neuron(s).
            % Calls recursively if a list is passed
            %   Note: many will not be identified uniquely; the default is
            %   to just concatenate the names
            if isempty(fieldnames(self.id_struct))
                disp('Names not stored in this object; aborting')
                names = 'Neuron names not stored';
                return
            elseif isempty(self.id_struct.ID)
                disp('Names not stored in this object; aborting')
                names = 'Neuron names not stored';
                return
            end
            if ~exist('neuron_ind','var') || isempty(neuron_ind)
                neuron_ind = 1:self.x_len;
            end
            if ~exist('use_original_order','var')|| isempty(use_original_order)
                use_original_order = true;
            end
            if ~exist('print_names','var')
                print_names = true;
            end
            if ~exist('print_warning','var')
                print_warning = true;
            end
            if ~exist('to_parse_names','var')
                to_parse_names = true;
            end
            
            % Call recursively if a list is input
            if ~isscalar(neuron_ind)
                names = cell(size(neuron_ind));
                for n=1:length(neuron_ind)
                    this_neuron = neuron_ind(n);
                    names{n} = ...
                        self.get_names(this_neuron, ...
                        use_original_order, print_names, print_warning);
                end
                return
            end
            
            % The data might be sorted, so get the new index
            if ~use_original_order
                neuron_ind = self.u_sort_ind(neuron_ind);
            end
            
            % Actually get the name
            if neuron_ind > length(self.id_struct.ID)
                if print_warning
                    warning('Index outside the length of names; assuming derivatives')
                end
                neuron_ind = neuron_ind - length(self.id_struct.ID);
                if neuron_ind > length(self.id_struct.ID)
                    names = '';
                    return
                end
            end
            names = {self.id_struct.ID{neuron_ind},...
                self.id_struct.ID2{neuron_ind},...
                self.id_struct.ID3{neuron_ind}};
            if print_names
                fprintf('Identifications of neuron %d: %s, %s, %s\n',...
                    neuron_ind, names{1},names{2},names{3});
            end
            if to_parse_names
                names = self.parse_names({names});
                if length(names)==1
                    % i.e. a single neuron
                    names = names{1};
                end
            end
            
        end % function
        
        function preprocess(self)
            % First call superclass basic preprocessor
            preprocess@AbstractDmd(self);
            
            % Now separate out validation and training data
            if self.hold_out_fraction > 0
                assert(self.hold_out_fraction < 1,...
                    'Hold-out fraction should be between 0 and 1')
                
                ind = round(self.sz(2)*(1-self.hold_out_fraction));
                self.dat_cross_val = self.dat(:, (ind+1):end);
                self.dat = self.dat(:, 1:ind);
            end
        end
    end % methods
    
    methods % Cross-validation
        function err_vec = calc_baseline_error(self,...
                num_starts, test_length, to_return_matrix, error_func, seed)
            % Calculates the error for a random set of start points for the
            % same length as the test set
            % Input:
            %   num_starts (200) - number of restarts within the training
            %       dataset
            %   test_length - percentage of the cross validation window to
            %       use for the reconstruction errors
            %   to_return_matrix (false) - to break down the errors by
            %       neuron; default just averages them
            %   error_func (L2) - what error function to use; default uses
            %       the L2 norm for each neuron, i.e. the matrix 'fro' norm
            %   seed - for rng
            if ~exist('num_starts','var') || isempty(num_starts)
                num_starts = 200;
            end
            if ~exist('test_length','var') || isempty(test_length)
                test_length = round(size(self.dat_cross_val,2) * ...
                    self.cross_val_window_size_percent);
            end
            if ~exist('to_return_matrix', 'var') || isempty(to_return_matrix)
                to_return_matrix = false;
            end
            if ~exist('error_func', 'var')
                error_func = @(x,y) norm(x-y,'fro')/numel(y);
            end
            if ~exist('seed','var')
                seed = 1;
            end
            
            rng(seed);
            max_start = size(self.dat,2) - test_length;
            t_starts = randi(max_start, [num_starts,1]);
            if to_return_matrix
                err_vec = zeros(self.x_len, num_starts);
            else
                err_vec = zeros(num_starts,1);
            end
            for i=1:num_starts
                t_span = t_starts(i):(t_starts(i)+test_length);
                dat_true = self.dat(1:self.x_len, t_span);
                dat_approx = self.calc_reconstruction_control(...
                    dat_true(:,1), t_span, false);
                if to_return_matrix
                    for i2 = 1:size(dat_true, 1)
                        err_vec(i2, i) = ...
                            error_func(dat_true(i2,:), dat_approx(i2,:));
                    end
                else
                    err_vec(i) = error_func(dat_true, dat_approx);
                end
            end
        end
        
        function err = calc_test_error(self, ...
                dat_test, to_return_matrix, error_func)
            % If a cross-validation test set has been set aside, use that
            % as test data
            % Input:
            %   dat_test - defaults to a predetermined set-aside fraction
            %       of the input dataset, but could be other data of the
            %       same size
            %
            % See also: calc_baseline_error (used to calculate the
            % distribution)
            if ~exist('dat_test','var') || isempty(dat_test)
                dat_test = self.dat_cross_val;
            end
            if ~exist('to_return_matrix', 'var') || isempty(to_return_matrix)
                to_return_matrix = false;
            end
            if ~exist('error_func', 'var')
                error_func = @(x,y) norm(x-y,'fro')/numel(y);
            end
            assert(~isempty(dat_test),'No test data')
            
            % Switch out the 'dat' property so we can use default functions
            original_dat = self.dat;
            self.dat = dat_test;
            if self.cross_val_window_size_percent < 1
                % Use many random restarts
                err = self.calc_baseline_error([], [],...
                    to_return_matrix, error_func);
            else
                err = self.calc_reconstruction_error();
            end
            self.dat = original_dat;
        end
        
    end
    
    methods % Reconstruction
        function dat_approx = calc_reconstruction_original(self, x0, tspan)
            % Reconstructs the data using the full DMD propagator matrix
            % starting from the first data point
            if ~exist('x0','var') || isempty(x0)
                x0 = self.dat(:,1);
            end
            if ~exist('tspan','var')
                num_frames = size(self.dat,2);
                tspan = linspace(0,num_frames*self.dt,num_frames+1);
            end
            
            ind = 1:self.x_len;
            A = self.A_original(ind, ind);
            
            dat_approx = zeros(length(x0), length(tspan));
            dat_approx(:,1) = x0;
            for i=2:length(tspan)
                dat_approx(:,i) = A * dat_approx(:,i-1);
            end
        end
        
        function dat_approx = calc_reconstruction_control(self,...
                x0, t_ind, which_control_signals, return_control_signal)
            % Reconstructs the data using the partial DMD propagator matrix
            % starting from the first data point using control signals
            %
            % Input:
            %   x0 - the initial point. Default is the x0 for the entire
            %       dataset
            %   t_ind - a list of t points. Default is entire dataset
            %   which_control_signals - a list of which control signals to
            %       use. Default is all of them
            %   return_control_signal (false) - boolean for returning the
            %       control signals as part of the reconstruction
            if ~exist('x0','var') || isempty(x0)
                x0 = self.dat(:,1);
            end
            if ~exist('t_ind','var') || isempty(t_ind)
                t_ind = 1:size(self.dat,2);
                %                 tspan = linspace(0,num_frames*self.dt,num_frames);
            end
            if ~exist('which_control_signals', 'var')
                which_control_signals = []; % i.e. all of them
            end
            if ~exist('return_control_signal','var')
                return_control_signal = false;
            end
            
            if self.dmd_offset>1
                dat_approx = self.calc_reconstruction_offset(...
                    [], [], return_control_signal);
                return
            end
            
            ind = 1:self.x_len;
            if ~self.use_A_thresholded
                A = self.A_separate(ind, ind);
                B = self.A_separate(ind, self.x_len+1:end);
            else
                A = self.A_thresholded(ind, ind);
                B = self.A_thresholded(ind, self.x_len+1:end);
            end
            
            if return_control_signal
                dat_approx = zeros(length(x0), length(t_ind));
                % bottom rows are not reconstructed; taken as given
                dat_approx(self.x_len+1:end,:) = ...
                    self.dat(self.x_len+1:end,:);
                dat_approx(:,1) = x0;
            else
                dat_approx = zeros(self.x_len, length(t_ind));
                dat_approx(:,1) = x0(1:self.x_len);
            end
            if isempty(which_control_signals)
                for i=2:length(t_ind)
                    u = self.dat(self.x_len+1:end, t_ind(i-1));
                    dat_approx(1:self.x_len, i) = ...
                        A*dat_approx(1:self.x_len, i-1) + B*u;
                end
            else
                B = B(:, which_control_signals);
                for i=2:length(t_ind)
                    u = self.dat(self.x_len+1:end, t_ind(i-1));
                    u = u(which_control_signals, :);
                    dat_approx(1:self.x_len, i) = ...
                        A*dat_approx(1:self.x_len, i-1) + B*u;
                end
            end
        end
        
        function dat_approx = calc_reconstruction_offset(self,...
                x0, t_ind, include_control_signal)
            % Reconstructs the data using the partial DMD propagator matrix
            % starting from the first data point using the 'outlier' rows
            % as control signals
            %   Modification: for use when dmd_offset>1
            if ~exist('x0','var') || isempty(x0)
                x0 = self.dat(:, 1:self.dmd_offset);
            end
            if ~exist('t_ind','var') || isempty(t_ind)
                t_ind = 1:(size(self.dat,2)-self.dmd_offset+1);
            end
            if ~exist('include_control_signal','var')
                include_control_signal = false;
            end
            
            ind = 1:self.x_len;
            A = self.A_separate(ind, ind);
            B = self.A_separate(ind, self.x_len+1:end);
            
            if include_control_signal
                dat_approx = zeros(length(x0), length(t_ind));
                % bottom rows are not reconstructed; taken as given
                dat_approx(self.x_len+1:end, :) = ...
                    self.dat(self.x_len+1:end, :);
                dat_approx(:, 1:self.dmd_offset) = x0;
            else
                dat_approx = zeros(self.x_len, length(t_ind));
                dat_approx(:, 1:self.dmd_offset) = x0(1:self.x_len,:);
            end
            for i = 2:length(t_ind)
                u = self.dat(self.x_len+1:end, t_ind(i-1));
                dat_approx(1:self.x_len, i+self.dmd_offset-1) = ...
                    A*dat_approx(1:self.x_len, i-1) + B*u;
            end
        end
        
        function dat_approx = calc_reconstruction_manual(self, x0, u)
            % Calculates a single step forward using the saved dynamics and
            % manually input state and control signal
            
            ind = 1:self.x_len;
            dat_approx = ...
                self.A_separate(ind, ind)*x0 + ...
                self.A_separate(ind, self.x_len+1:end)*u;
        end
        
        function error_approx = calc_reconstruction_error(self, ...
                which_norm, use_persistence, varargin)
            % Does not include the control signal
            % User can specify a particular mode (2-norm, etc)
            %   If that norm has parameters, those can be passed in after
            if ~exist('which_norm','var') || isempty(which_norm)
                which_norm = '2norm';
            end
            if ~exist('use_persistence','var') || isempty(use_persistence)
                use_persistence = false;
            end
            % Does not include the control signal (which has 0 error)
            dat_original = self.dat(1:self.x_len,:);
            if ~use_persistence
                dat_approx = self.calc_reconstruction_control([],[],false);
            else
                % Null model comparison
                dat_approx = repmat(dat_original(:,1), ...
                    [1, size(dat_original,2)]);
            end
            
            switch which_norm
                case '2norm'
                    error_approx = ...
                        norm(dat_approx-dat_original, 'fro')/numel(dat_approx);
                    
                case 'Inf'
                    error_approx = ...
                        norm(dat_approx-dat_original,Inf)/numel(dat_approx);
                    
                case 'expnorm'
                    lambda = varargin{1};
                    assert(lambda>0, 'The length scale should be positive')
                    error_approx = sum(sum(...
                        exp((dat_approx-dat_original)./lambda) )) / ...
                        numel(dat_approx);
                    
                case 'flat_then_2norm'
                    error_approx = dat_approx-dat_original;
                    if nargin < 3
                        lambda = mean(var(error_approx));
                    else
                        lambda = varargin{1};
                    end
                    set_to_zero_ind = abs(error_approx)<lambda;
                    error_approx(set_to_zero_ind) = 0;
                    error_approx = norm(error_approx) / ...
                        length(find(set_to_zero_ind));
                    
                case 'residual_vector'
                    error_approx = dat_approx-dat_original;
                    
                otherwise
                    error('Unknown error metric')
            end
        end
        
        function A = set_A_thresholded(self, tol)
            % Thresholded A_original and changes the reconstruction
            % plotters to use that matrix
            self.use_A_thresholded = true;
            self.sparse_tol = tol;
            
            A = self.A_original;
            A(abs(A)<tol) = 0;
            self.A_thresholded = A;
        end
        
        function reset_threshold(self)
            % Resets settings back to original dynamics matrix
            self.use_A_thresholded = false;
        end
    end
    
    methods (Access=private)
        function error_approx = update_and_reconstruction_error(self,...
                A, which_norm, varargin)
            % Updates the A_sep matrix (meant to be temporary, e.g. in the
            % midst of converging on a final A_sep), then calculates the
            % error and returns it (scalar)
            if ~exist('which_norm','var') || isempty(which_norm)
                which_norm = '2norm';
            end
            if ~exist('varargin','var')
                varargin = {};
            end
            self.A_separate = A;
            error_approx = ...
                self.calc_reconstruction_error(which_norm, varargin);
        end
        
        function postprocess(self)
            % Gets rid of some raw data if to_save_raw_data==false
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
        end
    end
    
    methods % Plotting
        
        function plot_using_settings(self)
            % Plots several things after analysis is complete
            %
            % There are several plotting possibilities, which are all set
            %       in the original settings struct:
            % 'to_plot_data': the sorted data set (control signals on
            %   bottom)
            % 'to_plot_A_matrix_svd': the svd of the matrix which solves
            %   the equation $$ x_(t+1) = A*x_t $$ will be plotted
            % 'which_plot_data_and_filter': plots certain neurons with the
            %   filter used to determine error outliers
            % 'to_plot_data_and_outliers': plots individual neuron errors
            %   with outliers marked (method depends on 'sort_mode');
            %   interactive
            %
            % And printing options:
            % 'to_print_error': prints L2 error of normal fit vs. fit using
            %   the control signals
            % 'use_optdmd': also prints L2 error of an alternative dmd
            %   algorithm
            
            x_length = self.x_len;
            u_length = self.u_len;
            
            %---------------------------------------------
            %% Plotting options
            %---------------------------------------------
            if self.to_plot_data
                self.plot_data_and_control();
            end
            
            if self.to_plot_A_matrix_svd
                A_x = self.A_separate(1:x_length,1:x_length);
                A_u = self.A_separate(1:x_length,x_length+1:end);
                
                plotSVD(A_x);
                title(sprintf('Max possible rank (data): %d',x_length))
                
                plotSVD(A_u);
                title(sprintf('Max possible rank (ctr): %d',min(u_length,x_length)))
            end
            
            %---------------------------------------------
            %% Different per-neuron error metrics
            %---------------------------------------------
            
            if self.which_plot_data_and_filter > 0
                self.plot_data_and_filter(...
                    self.error_mat(self.which_plot_data_and_filter,:)',...
                    self.filter_window_size, self.outlier_window_size)
            end
            if self.to_plot_data_and_outliers
                sorted_error_outliers = self.calc_error_outliers(self.error_mat, ...
                    self.filter_window_size, self.outlier_window_size);
                self.plot_data_and_outliers(sorted_error_outliers,...
                    @(x,y) self.callback_plotter(x,y, ...
                    self.error_mat, self.filter_window_size, self.outlier_window_size),...
                    false)
            end
            
            if self.to_print_error
                fprintf('Error in original dmd is %f\n',self.original_error)
                fprintf('Error in separated dmd is %f\n',self.sep_error)
            end
            
            %             if strcmp(self.dmd_mode, 'optdmd')
            %                 optdmd_error = norm(w*diag(b)*exp(e*t)-X)/numel(X);
            %                 fprintf('Error in optdmd is %f\n',optdmd_error)
            %             end
            %==============================================================
        end
        
        function plot_data_and_control(self)
            % Plots the raw data on the left and the data with the control
            % signal set to 0 on the right
            X2_sep = ...
                [self.dat(1:self.x_len,:);...
                zeros(self.u_len,size(self.dat,2))];
            plot_2imagesc_colorbar(self.dat, X2_sep, '1 2',...
                'Original X2 data (sorted)',...
                'X2 data with u set to 0');
        end
        
        function [dat_approx, fig] = plot_reconstruction(self, ...
                use_control, include_control_signal, to_compare_raw,...
                neuron_ind, use_sorted_order)
            % Plots a reconstruction of the data using the stored linear
            % model. Options (defaults in parentheses):
            %   use_control (false): reconstruct using control, i.e.
            %       $$ x_(t+1) = A*x_t + B*u $$
            %       or without control (the last _B_*_u_ term)
            %   include_control_signal (false): plot the control signal as
            %       well as the reconstruction
            %   to_compare_raw (true): also plot the raw data
            %   neuron_ind (0): if >0, plots a single neuron instead of the
            %       entire dataset
            
            if ~exist('use_control','var')
                use_control = false;
            end
            if ~exist('include_control_signal','var') || isempty(include_control_signal)
                include_control_signal = false;
            end
            if ~exist('to_compare_raw', 'var')
                to_compare_raw = true;
            end
            if ~exist('neuron_ind','var')
                neuron_ind = 0;
            end
            if ~exist('use_sorted_order','var')
                use_sorted_order = false;
            end
            
            if neuron_ind>0
                if ~use_sorted_order
                    sorted_neuron_ind = find(self.u_sort_ind==neuron_ind);
                else
                    sorted_neuron_ind = neuron_ind;
                    neuron_ind = self.u_sort_ind(sorted_neuron_ind);
                end
                if isempty(sorted_neuron_ind)
                    error('Attempted to plot a neuron outside of the dataset (might be in the controller)')
                end
            else
                sorted_neuron_ind = 0;
            end
            
            if use_control
                if ~include_control_signal
                    assert(sorted_neuron_ind<=self.x_len,...
                        'If you really meant to plot a controller neuron, set include_control_signal=true')
                    full_dat = self.dat(1:self.x_len,:);
                    title_str = 'Reconstructed data (with control; signal not shown)';
                else
                    title_str = sprintf(...
                        'Reconstructed data (control signal = rows %d-%d)',...
                        self.x_len+1, size(self.dat,1));
                    full_dat = self.dat;
                end
                dat_approx = self.calc_reconstruction_control(...
                    [],[],[],include_control_signal);
            else
                full_dat = self.dat;
                dat_approx = self.calc_reconstruction_original();
                title_str = 'Reconstructed data (no control)';
            end
            if to_compare_raw
                if sorted_neuron_ind < 1
                    fig = plot_2imagesc_colorbar(full_dat, real(dat_approx),...
                        '2 1', 'Original data', title_str);
                    
                    names = self.get_names([],[],false,false);
                    subplot(2,1,1);
                    yticks(1:size(dat_approx,1))
                    yticklabels(names)
                    subplot(2,1,2);
                    yticks(1:size(dat_approx,1))
                    yticklabels(names)
                else
                    title_str = [title_str ...
                        sprintf('; neuron %d (name=%s)',...
                        neuron_ind, self.get_names(neuron_ind))];
                    fig = figure('DefaultAxesFontSize',16);
                    hold on
                    plot(full_dat(sorted_neuron_ind,:), 'LineWidth', 2)
                    plot(real(dat_approx(sorted_neuron_ind,:)), 'LineWidth', 3)
                    legend({'Original data','Reconstructed trajectory'})
                    ylabel('Amplitude')
                    xlabel('Time')
                    if sorted_neuron_ind < self.x_len+1
                        title(title_str)
                    else
                        title('This neuron taken as is; no reconstruction')
                    end
                end
            else
                fig = figure('DefaultAxesFontSize',12);
                if sorted_neuron_ind < 1
                    imagesc(dat_approx);
                    colorbar
                else
                    title_str = [title_str ...
                        sprintf('; neuron %d (name=%s)',...
                        sorted_neuron_ind, self.get_names(sorted_neuron_ind))];
                    plot(dat_approx(sorted_neuron_ind,:))
                    ylabel('Amplitude')
                    xlabel('Time')
                end
                title(title_str)
            end
            
            xlim([0, size(dat_approx,2)]);
            %             if neuron_ind==0
            %                 yticks(1:size(dat_approx,1))
            %                 yticklabels(self.get_names([],[],false,false))
            %             end
        end
        
        function fig = plot_data_and_filter(~, dat, filter_window, outlier_window)
            fig = figure('DefaultAxesFontSize',12);
            x = 1:size(dat,1);
            x_delay = x - (filter_window-1)/(2);
            plot(x, dat);
            hold on;
            dat_filter = filtfilt(ones(1,filter_window)/filter_window,1,dat);
            TF = isoutlier(dat_filter,'movmedian',outlier_window,'SamplePoints',x);
            plot(x_delay,dat_filter, x_delay(TF),dat_filter(TF),'x','LineWidth',3)
            legend('Raw Data','Moving Average','Outlier');
        end
        
        function fig = plot_data_and_outliers(self, ...
                dat, callback_func, use_original_order)
            % Plots the error signals with their outliers, using a GUI to
            % explore the individual neurons
            if ~exist('dat','var') || isempty(dat)
                dat = self.error_outliers;
            end
            if ~exist('callback_func','var') || isempty(callback_func)
                callback_func = @(x,y) self.callback_plotter(x,y, ...
                    self.neuron_errors,...
                    self.filter_window_size,...
                    self.outlier_window_size);
            end
            if ~exist('use_original_order','var')
                use_original_order = true;
            end
            
            fig = figure('DefaultAxesFontSize',12);
            x = 1:size(dat,1);
            vec = ones(1,length(x));
            hold on;
            [TF,lower,upper,center] = isoutlier(dat,...
                'ThresholdFactor', self.cutoff_multiplier);
            plot(x(TF),dat(TF),'x',...
                x,lower*vec,x,upper*vec,x,center*vec)
            plot(dat, 'o',...
                'ButtonDownFcn',callback_func);
            legend('Outlier',...
                'Lower Threshold','Upper Threshold','Center Value',...
                'Original Data')
            ylabel(sprintf('Error (measured using %s)', self.sort_mode),...
                'Interpreter','None')
            % Use neuron names for xticklabels (many will be empty)
            if use_original_order
                num_ticks = length(dat);
            else
                num_ticks = self.x_len;
            end
            xticks(1:num_ticks)
            xticklabels( self.get_names(1:num_ticks,...
                use_original_order, false, false));
            xtickangle(90)
            if strcmp(self.sort_mode, 'user_set')
                title('Calculated outliers (user set ones used instead)')
            else
                title('Calculated outliers')
            end
        end
        
        function fig = plot_data_and_exp_filter(~, dat, alpha, outlier_window)
            fig = figure('DefaultAxisFontSize',12);
            x = 1:size(dat,1);
            x_delay = x-1;
            plot(x, dat);
            hold on;
            dat_filter = filtfilt(alpha, 1-alpha, dat);
            TF = isoutlier(dat_filter,'movmedian',outlier_window,'SamplePoints',x);
            plot(x_delay,dat_filter, x_delay(TF),dat_filter(TF),'x','LineWidth',3)
            legend('Raw Data','Moving Average','Outlier');
        end
        
        function callback_plotter(self, ~, evt, dat, filter_window, outlier_window)
            % On left click:
            %   Plots the original data minus the linear component (i.e.
            %   the error signal, interpreted as important for control)
            % On other (e.g. right) click:
            %   Displays the neuron name, if identified
            this_neuron = evt.IntersectionPoint(1);
            if evt.Button==1
                self.plot_data_and_filter(dat(this_neuron,:)',...
                    filter_window, outlier_window);
                this_name = self.get_names(this_neuron);
                if isempty(this_name)
                    this_name = 'N/A';
                end
                title(sprintf('Residual for neuron %d (name=%s)',...
                    this_neuron, this_name))
                xlabel('Time')
                ylabel('Error')
            else
                self.plot_reconstruction(true, true, true, this_neuron);
            end
        end
        
        function turn_plotting_off(self)
            % Sets all plotting settings to 'false'
            self.to_plot_cutoff = false;
            self.to_plot_data = false;
            self.to_plot_A_matrix_svd = false;
            self.to_plot_data_and_outliers = false;
            
            self.to_print_error = false;
        end
    end
    
    methods % For dependent variables
        function out = get.error_mat(self)
            X1 = self.dat(:,1:end-1);
            X2 = self.dat(:,2:end);
            out = self.A_separate*X1-X2;
            out = out(1:self.x_len,:);
        end
    end
    
    methods(Static)
        function parsed_name_list = parse_names(name_list, keep_ambiguous)
            % Input: a cell array of 3x1 cell arrays, containing 3 possible
            % names for each neuron
            % Output: a cell array containing the strings of either a) only
            % unambiguously identified neurons or b) compound names when
            % the id has multiple names
            if ~exist('keep_ambiguous','var')
                keep_ambiguous = true;
            end
            
            parsed_name_list = cell(size(name_list));
            for jN=1:length(name_list)
                this_neuron = name_list{jN};
                new_name = '';
                for jID=1:length(this_neuron)
                    check_name = this_neuron{jID};
                    if isempty(check_name)
                        continue
                    elseif strcmp(check_name,new_name)
                        continue
                    else
                        % Different, non-empty ID name
                        if keep_ambiguous
                            new_name = [new_name check_name]; %#ok<AGROW>
                        else
                            new_name = 'Ambiguous';
                            break
                        end
                    end % if
                end % for
                parsed_name_list{jN} = new_name;
            end % for
        end % function
        
        function A = threshold_matrix(A, tol)
            % Sets all values in the matrix A with abs()<tol to 0
            A(abs(A)<tol) = 0;
        end
        
        function [err, A, B] = calc_one_step_error(X1, X2, u)
            % Calculates the simple L2 error from the traditional DMD
            % one-step fit
            r = optimal_truncation(X2);
            rtilde = 0; % Added adaptive discovery of truncation
            
            [~, ~, B, ~, Uhat, ~, ~, ~, ~, ~, Atilde] = ...
                func_DMDc(X1, X2, u, r, rtilde);
            
            U = Uhat(:,1:r);
            A = U*Atilde*U';
            
            % Calculate error
            err = norm(X2 - (A*X1+B*u))/numel(X1);
        end
    end % methods
end % class

