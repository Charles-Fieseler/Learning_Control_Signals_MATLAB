function [ControlSignalPath_object, my_model_base] = ...%[all_U, all_A, all_B, my_model_base] = ...
    learn_control_signals(file_or_obj, s)
%% learn_control_signals
% Analyzes the residuals of a DMD model in order to learn control signals
% from data
%
% INPUTS
%   file_or_obj - the file for the data matrix OR a model with the data
%       note: rows are channels and columns are time
%   s - settings struct, with values and defaults as follows:
%         verbose (true)
%         to_use_L1 (false) - to use LASSO, i.e. an L1 penalty. Default is
%           iterative least squares with hard thresholding
%         num_iter (80) - Number of iterations
%         iter_removal_fraction (0.05) - Fraction of entries to remove in
%           each iteration
%         iter_re_up_fraction (0) - Fraction of largest entries to add back
%           in each iteration; experimental
%         r_ctr (15) - Number of control signals to search for
%         to_use_model_U (false) - If a simulation model is passed, whether
%           to initialize this search using that object.
%         to_threshold_total_U (false) - When applying hard threshold, do
%           the entire matrix. Default is row-by-row. This is suggested
%           because the rows are only defined up to a scaling determined by
%           the matrix B
%         only_positive_U (true) - Enforce positivity in the initialization
%           of the control signal, as well as each iteration. Sometimes
%           helps interpretability.
%         to_smooth_controller (false) - Apply a Gaussian filter to the
%           output control signal during the algorithm
%         to_smooth_initialization (false) - Apply a moving average and
%           Gaussian filter to the initialization
%         initialization_smoothing_windows ([3, 6]) - windows for the
%           moving average and Gaussian filters, respectively. Note: 
%           to_smooth_initialization must be 'true' to use these.
%         custom_initialization ([]) - custom initialization for the
%           control signals; overrides filtering and positivity options
%           above. Checks for correct number of rows
%         seed (13) - the initialization uses rng
%
% OUTPUTS
%   ControlSignalPath_object - A class which contains:
%       all_U - The control signal matrices in a cell array, across sparsity
%           iterations. Each approximately solves: X2 = A*X1 + B*U
%       all_A - The A matrices corresponding the control signals U above
%       all_B - The B matrices corresponding the control signals U above
%   my_model_base - The model used to preprocess the data. Note that in the
%       future I should definitely separate out the data processing from
%       the model object
%

%% Dependencies
%   .m files, .mat files, and MATLAB products required:(updated on 26-Feb-2020)
%         MATLAB (version 9.4)
%         Signal Processing Toolbox (version 8.0)
%         Statistics and Machine Learning Toolbox (version 11.3)
%         ControlSignalPath.m
%         SimulationPlottingObject.m
%         aic_2step_dmdc.m
%         calc_nstep_error.m
%         convertKato2Zimmer.m
%         checkModes.m
%         func_DMD.m
%         func_DMDc.m
%         AdaptiveDmdc.m
%         exact_dmdc.m
%         sparse_dmd_fast.m
%         AbstractDmd.m
%         SettingsImportableFromStruct.m
%         top_ind_then_sort.m
%         optimal_truncation.m
%         calc_contiguous_blocks.m
%         calc_snr.m
%         svd_truncate.m
%         cmap_white_zero.m
%         plotSVD.m
%         plot_2imagesc_colorbar.m
%         plot_colored.m
%         plot_gray_lines.m
%         plot_std_fill.m
%         brewermap.m
%         RobustPCA.m
%         acf.m
%         optimal_SVHT_coef.m
%         TVRegDiff.m
%         
%
%   See also: ControlSignalPath.m
%
%
% Author: Charles Fieseler
% University of Washington, Dept. of Physics
% Email address: charles.fieseler@gmail.com
% Website: coming soon
% Created: 13-Feb-2019
%========================================

%---------------------------------------------
%% Set up defaults
%---------------------------------------------
if ~exist('s','var')
    s = struct();
end
defaults = struct(...
    'verbose', true, ...
    'to_use_L1', false,...
    'num_iter', 80,...
    'iter_removal_fraction', 0.05,...
    'iter_re_up_fraction', 0,...
    'r_ctr', 15,...
    'to_use_model_U', false,...
    'to_threshold_total_U', false,...
    'only_positive_U', true,...
    'to_smooth_controller', false, ...
    'to_smooth_initialization', false, ...
    'initialization_smoothing_windows', [3, 6],...
    'custom_initialization', [],...
    'seed', 13);
for key = fieldnames(defaults).'
    k = key{1};
    if ~isfield(s, k)
        s.(k) = defaults.(k);
    end
end
rng(s.seed);

%---------------------------------------------
%% Create model for preprocessing
%---------------------------------------------
if ischar(file_or_obj)
    dat_struct = importdata(file_or_obj);
    
    if isstruct(dat_struct)
        % Process into a model
        settings = struct(...
            'to_subtract_mean',false,...
            'to_subtract_mean_global',false,...
            'add_constant_signal',false,...
            'use_deriv',false,...
            'filter_window_dat', 1,...
            'dmd_mode','func_DMDc');
        settings.global_signal_mode = 'ID_binary_transitions';

        my_model_base = SimulationPlottingObject(dat_struct, settings);
    else
        % Assume this is just the data
        my_model_base = file_or_obj;
    end
elseif isnumeric(file_or_obj)
    % Assume this is just the data
    my_model_base = file_or_obj;
else
    assert(isa(file_or_obj, 'SimulationPlottingObject'), 'Wrong object type')
    my_model_base = file_or_obj;
end

%---------------------------------------------
%% Initialize the control signal
%---------------------------------------------
if isa(file_or_obj, 'SimulationPlottingObject')
    X1 = my_model_base.dat(:,1:end-1);
    X2 = my_model_base.dat(:,2:end);
else
    X1 = my_model_base(:,1:end-1);
    X2 = my_model_base(:,2:end);
end

[n, m] = size(X1);

all_err = zeros(s.num_iter,2);
if ~isempty(s.custom_initialization)
    U = s.custom_initialization;
    assert(size(U,1)==s.r_ctr,...
        'Custom initialization should have %d rows', s.r_ctr)
elseif s.to_use_model_U
    U = my_model_base.control_signal(1:s.r_ctr,1:m);
else
    % Initialize with the raw errors from a naive DMD fit
    err = real(X2 - (X2/X1)*X1);
    if s.to_smooth_initialization
        err = smoothdata(err, 2, 'movmean', ...
            s.initialization_smoothing_windows(1));
        err = smoothdata(err, 2, 'gaussian', ...
            s.initialization_smoothing_windows(2));
    end
    
    % Process raw error into modes
    if s.only_positive_U
        % Default: Initialize with non-negative factorization        
        [~, U0] = nnmf(err, s.r_ctr);
        U = U0(1:s.r_ctr,:);
        U = U ./ max(U,[],2);
    else
        [~, ~, U0] = svd(err);
        U = U0(:,1:s.r_ctr)';
    end
end

%---------------------------------------------
%% Iteratively sparsify the controller
%---------------------------------------------
sparsity_pattern = false(size(U));
all_U = cell(s.num_iter, 1);
all_A = cell(s.num_iter, 1);
all_B = cell(s.num_iter, 1);

if s.only_positive_U
    sparse_func = @(x) x;
else
    sparse_func = @(x) abs(x);
end

tstart = tic;
% Main loop
for i = 1:s.num_iter
    if s.verbose
        fprintf('Iteration %d/%d\n', i, s.num_iter)
    end
    % Step 1:
    %   Get A and B matrix, given U
    AB = X2/[X1; full(U)];
    A = AB(:,1:n);
    B = AB(:,(n+1):end);
    
    all_err(i, 1) = norm(A*X1 + B*U - X2, 2);
    if i > 1
        all_A{i-1} = A;
        all_B{i-1} = B;
    end

    % Step 2:
    %   Sequential LS thresholding on U
    U_raw = B\(X2 - A*X1);
    U = U_raw;
    if s.to_threshold_total_U
        % Normalize U by the actual effect it has on the data, i.e. include B
        U_effective = zeros(size(U));
        for i2 = 1:s.r_ctr
            U_effective(i2,:) = sum(abs(B(:,i2))*U(i2,:), 1);
        end
        U_effective_nonsparse = U_effective;
        U_effective_nonsparse(~sparsity_pattern) = 0;
        U_effective(sparsity_pattern) = 0;
        % Get rid of bottom 5% of BU nonzeros
        tmp = reshape(sparse_func(U_effective), [m*s.r_ctr, 1]);
        threshold = quantile(tmp(tmp>0),s.iter_removal_fraction);
        sparsity_pattern = sparse_func(U_effective) < threshold;

        % Add entries back in for the top 1% of the ignored matrix, if they
        % are above the median of the entries that are left
        if s.iter_re_up_fraction > 0
            tmp2 = reshape(sparse_func(U_effective_nonsparse), [m*s.r_ctr, 1]);
            threshold_top = max([quantile(tmp2(tmp2>0),1-s.iter_re_up_fraction), median(tmp)]);
            re_up_pattern = sparse_func(U_effective_nonsparse) > threshold_top;
        %     U(re_up_pattern) = U_effective_nonsparse(re_up_pattern);
        end
    else
        % Threshold per row of U
        U_nonsparse = U;
        U(sparsity_pattern) = 0;
        U_nonsparse(~sparsity_pattern) = 0;
        re_up_pattern = false(size(sparsity_pattern));
        has_stalled = true;
        for i2 = 1:s.r_ctr
            tmp = U(i2,:);
            threshold = max(quantile(tmp(tmp>0),s.iter_removal_fraction),...
                min(tmp(tmp>0)));
            if isempty(threshold)
                % i.e. all values are 0
                threshold = 0;
            else
                has_stalled = false;
            end
            sparsity_pattern(i2,:) = sparse_func(tmp) <= threshold;

            if s.iter_re_up_fraction > 0
                tmp2 = U_nonsparse(i2,:);
                threshold_top = max([quantile(tmp2(tmp2>0),1-s.iter_re_up_fraction), median(tmp)]);
                re_up_pattern(i2,:) = sparse_func(tmp2) > threshold_top;
            end
        end
    end
    
    if s.iter_re_up_fraction > 0
        sparsity_pattern = (sparsity_pattern - re_up_pattern)==1;
    end
    U(sparsity_pattern) = 0;
    % Smooth the controller slightly; keep the max at 1.0
    if s.to_smooth_controller
%         U = 1.02*smoothdata(U, 2, 'gaussian', [1 0]);
        U = smoothdata(U, 2, 'gaussian', [1 0]);
        U = U ./ max(U, [], 2);
    end
    
    % Save for output
    all_U{i} = U;
    
    if s.verbose
        fprintf('Number of nonzero control signals: %d\n', nnz(U))
        if s.iter_re_up_fraction > 0
            fprintf('Number of signals added back in: %d\n',...
                nnz(re_up_pattern))
        end
    end
    
    all_err(i, 2) = norm(A*X1 + B*U - X2, 'fro');
    if has_stalled
        disp('All control signals are 0. Stopping early (this is nothing to worry about)')
        break
    end
end
% Fill rest of output to keep output consistent
if has_stalled
    for i2 = i:s.num_iter
        all_U{i2} = all_U{i-1};
        all_A{i2} = all_A{i-1};
        all_B{i2} = all_B{i-1};
    end
else
    [all_A{end}, all_B{end}] = exact_dmdc([X1, X2(:, end)], all_U{end});
end
if s.verbose
    toc(tstart)
end

% Final output packaging
ControlSignalPath_object = ControlSignalPath(...
    file_or_obj, s, all_A, all_B, all_U);

end