function [ A_sparse, all_errors ] = sparse_dmd(X, prams)
% Do sparse DMD

%% Set defaults
if ~exist('prams','var')
    prams = struct();
end
defaults = struct(...
    'min_tol', {{{0}}},...
    'tol2column_cell',{{{1:size(X,2)}}},...
    'error_func', @(A, x, b, rows) norm(A*x-b(rows,:)),...
    'initial_sparsity_pattern', [],...
    'max_error_mult', 2,...
    'column_mode', false,...
    'rows_to_predict', 1:size(X,1),...
    'sparsity_goal', 0.5,...
    'max_iter', 50,...
    'truncation_rank', -1,...
    'verbose', true,...
    'cull_n_lowest_values', 1,...
    'sparsity_mode', 'threshold');
fnames = fieldnames(prams);
for key = fieldnames(defaults).'
    k = key{1};
    if ~ismember(k,fnames)
        prams.(k) = defaults.(k);
    end
end

if isnumeric(X)
    X1 = X(:,1:end-1);
    X2 = X(:,2:end);
elseif iscell(X)
    X1 = X{1};
    X2 = X{2};
else
    error('Unrecognized data format')
end

if prams.min_tol{1}{:} == 0
    prams.min_tol = cell(length(prams.tol2column_cell),1);
    for i = 1:length(prams.min_tol)
        prams.min_tol{i} = {0};
    end
end
%==========================================================================

%% Set up the convex optimization problem and solve with cvx
% Reduce the rank of the data
if prams.truncation_rank == 0
    [~, ~, X1] = optimal_truncation(X1);
    [~, ~, X2] = optimal_truncation(X2);
elseif prams.truncation_rank > 0
    ind = 1:prams.truncation_rank;
    [U1, D1, V1] = svd(X1);
    X1 = U1(:,ind)*D1(ind,ind) * V1(ind,:)';
    [U2, D2, V2] = svd(X2);
    X2 = U2(:,ind)*D2(ind,ind) * V2(ind,:)';
else
    assert(prams.truncation_rank == -1,...
        'Invalid truncation rank')
end
% Initialize with MATLAB backslash
A_sparse = X2 / X1;
A_sparse_old = A_sparse;
% Set all rows we don't care about to 0
if ~prams.column_mode
    assert(isequal(prams.rows_to_predict, 1:size(A_sparse,1)),...
        'If only predicting a subset of rows, set column_mode=true')
end
A_sparse = A_sparse(prams.rows_to_predict,:);
n = length(A_sparse); %#ok<NASGU>
% Use cvx to set up a sequential thresholding loop
total_num_elem = numel(A_sparse);
% Intelligently set the threshold
for i = 1:length(prams.tol2column_cell)
    if prams.min_tol{i}{:} == 0
        if prams.column_mode
            tmp = prams.tol2column_cell{i}{:};
            A_block = abs(A_sparse(:,tmp));
        else
            A_block = abs(A_sparse);
        end
        ind = ceil(numel(A_block) * prams.sparsity_goal);
        A_sort = sort(A_block(:), 'descend');
        % Set tolerance based on sparsity goal
        prams.min_tol{i}{:} = A_sort(ind) + 1e-8;
    end
end
% Set up intitial sparsity structure
A0 = prams.initial_sparsity_pattern;
if ~isempty(A0)
    assert(islogical(A0),...
        'Sparsity structure must be a logical indexing matrix')
    assert(isequal(size(A0), size(A_sparse)),...
        'Sparsity structure must be the same size as the dynamics')
    sparsity_pattern = prams.initial_sparsity_pattern;
    A_sparse(sparsity_pattern) = 0;
    A_sparse_old(sparsity_pattern) = 0;
else
    sparsity_pattern = zeros(size(A_sparse));
end
num_nnz = zeros(prams.max_iter,1);
num_nnz(1) = nnz(A_sparse);
all_errors = zeros(prams.max_iter,1);
all_errors(1) = prams.error_func(A_sparse, X1, X2, prams.rows_to_predict);
error_max = all_errors(1)*prams.max_error_mult;
did_it_abort_early = false;

for i=2:prams.max_iter
    % Threshold to enforce sparsity; may have a different threshold for
    % different columns
    if prams.column_mode
        for i3 = 1:length(prams.tol2column_cell)
            tmp = prams.tol2column_cell{i3}{:};
            sparsity_pattern(:,tmp) = ...
                abs(A_sparse(:,tmp)) < prams.min_tol{i3}{:};
        end
    else
        sparsity_pattern = abs(A_sparse) < prams.min_tol{1}{:};
    end
    sparsity_pattern = logical(sparsity_pattern);
    
    num_nnz(i) = nnz(A_sparse(prams.rows_to_predict,:));
    all_errors(i) = prams.error_func(A_sparse, X1, X2, prams.rows_to_predict);
    if prams.verbose
        fprintf('Iteration %d:\n', i-1)
        fprintf('  %d nonzero-entries (goal: %d)\n',...
            num_nnz(i), round(total_num_elem*prams.sparsity_goal))
        fprintf('  Current error: %.4f (max=%.4f)\n',...
            all_errors(i), error_max)
    end
    % Check convergence etc
    if i>2
        current_sparsity = num_nnz(i) / total_num_elem;
        if strcmp(prams.sparsity_mode, 'threshold')
            if all_errors(i) > error_max
                % If we have already halved the threshold, abort early instead
                % of bouncing around the threshold
                A_sparse = A_sparse_old;
                if length(find(diff(all_errors>error_max))) < 2
                    % Note: diff() because we want to continue if:
                    %   all_errors>error_max = [0 0 0 1 1 1]
                    tol_factor = abs(all_errors(i)-error_max) / ...
                        (all_errors(i)+error_max);
                    fprintf('  Error exceeded max; multiplying threshold by %.2f\n',...
                        tol_factor)
                    for i2 = 1:length(prams.min_tol)
                        prams.min_tol{i2}{:} = ...
                            prams.min_tol{i2}{:} * tol_factor;
                    end
                else
                    disp('  Error exceeded maximum again; aborting')
                    did_it_abort_early = true;
                    break
                end
            elseif  current_sparsity < prams.sparsity_goal
                fprintf('Achieved sparsity goal (%f percent)\n',current_sparsity)
                break
            elseif (num_nnz(i-1)-num_nnz(i)) < max([0.02*num_nnz(i), 5])
                % Measure what would be needed to get another 50% to the
                % sparseness goal and set that as the threshold
                nnz_goal = round(total_num_elem*prams.sparsity_goal);
                f = @(x) abs(length(A_sparse(A_sparse>x))-nnz_goal);
                goal_tol = fminbnd(f, prams.min_tol{1}{:}, max(max(A_sparse)));
                tol_factor = abs(all_errors(i)-error_max) / ...
                    (all_errors(i)+error_max);
                if isempty(goal_tol)
                    warning('Failed to update tolerance; exiting')
                    break
                end
                % Move a percentage towards the ideal goal in relation to how
                % much error we can increase
                for i2 = 1:length(prams.min_tol)
                    prams.min_tol{i2}{:} = ...
                        prams.min_tol{i2}{:} + tol_factor*goal_tol;
                end

                fprintf('  Stall predicted; new tolerance is %.2f\n',...
                    prams.min_tol{1}{:})
            else
                A_sparse_old = A_sparse;
            end
            for i3 = 1:length(prams.tol2column_cell)
                tmp = prams.tol2column_cell{i3}{:};
                sparsity_pattern(:,tmp) = ...
                    abs(A_sparse(:,tmp)) < prams.min_tol{i3}{:};
            end
            sparsity_pattern = logical(sparsity_pattern);
        elseif strcmp(prams.sparsity_mode, 'cull')
            if current_sparsity < prams.sparsity_goal
                fprintf('Achieved sparsity goal (%f percent)\n',current_sparsity)
                break
            end
            sz = size(A_sparse);
            tmp = reshape(abs(A_sparse),[sz(1)*sz(2),1]);
            sort_tmp = sort(tmp(tmp>0));
            sparsity_pattern = abs(A_sparse) <= ...
                sort_tmp(prams.cull_n_lowest_values);
        else
            error('Unrecognized sparsity mode')
        end
    else
        A_sparse(sparsity_pattern) = 0;
    end
    % Actually solve, either all at once or row by row
    if ~prams.column_mode
        cvx_begin quiet
            variable A_sparse(n,n)
            minimize( ...
                norm(A_sparse*X1-X2,2) )
            A_sparse(sparsity_pattern) == 0 %#ok<NOPRT,EQEFF>
        cvx_end
    else
        for i2 = prams.rows_to_predict
            if prams.verbose && (i2==1 || mod(i2,20)==0)
                fprintf('Solving row %d/%d...\n',i2,prams.rows_to_predict(end))
            end
            X2_row = X2(i2,:);
            sparsity_pattern_row = sparsity_pattern(i2,:);
            
            cvx_begin quiet
                variable A_sparse_row(1,n)
                minimize( ...
                    norm(A_sparse_row*X1-X2_row,2) )
                A_sparse_row(sparsity_pattern_row) == 0 %#ok<NOPRT,EQEFF>
            cvx_end
            
            A_sparse(i2,:) = A_sparse_row;
        end
    end
end

if prams.verbose
    if did_it_abort_early
        final_index = i-1;
    else
        final_index = i;
    end
    fprintf('Number of nonzero entries decreased from %d to %d\n',...
        num_nnz(1), num_nnz(final_index));
    fprintf('Error increased from %.4f to %.4f\n',...
        all_errors(1), all_errors(final_index));
end

end

