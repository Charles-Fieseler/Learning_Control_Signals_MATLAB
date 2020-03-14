function [err, all_err] = dmdc_cross_val(X, U, num_folds, err_steps, ...
    cross_val_mode, is_inclusive)
% Does 'num_folds' cross validation with a DMDc framework
%   Can use error steps that are greater than 1
%
% Input:
%   X - data (columns are time slices)
%   U - control signals
%   num_folds - number of k-folds
%   err_steps (1) - number of steps in the future to calculate the error
%   cross_val_mode ('chaining') - by default does a "chained" or "rolling"
%       cross-validation, which uses as test data the block exactly after
%       the training block(s). This is recommended for time-series data
%   is_inclusive (true) - Includes the error for multiple time steps, if 
%       err_steps>1... otherwise does nothing
if ~exist('err_steps', 'var')
    err_steps = 1;
end
if ~exist('cross_val_mode', 'var') || isempty(cross_val_mode)
    cross_val_mode = 'chaining';
end
if ~exist('is_inclusive', 'var')
    is_inclusive = true;
end
all_err = zeros(num_folds, 1);

m = size(X, 2);
window_starts = round(linspace(1, m, num_folds+1));
if strcmp(cross_val_mode, 'chaining')
    window_starts(1) = [];
    num_folds = num_folds - 1;
end
all_ind = 1:(m-err_steps);
n = size(X, 1);
for i = 1:num_folds
    % Get test and training indices
    test_ind = window_starts(i):(window_starts(i+1)-err_steps);
    
    if strcmp(cross_val_mode, 'chaining')
        train_ind = 1:test_ind(i) - err_steps;
    else
        train_ind = all_ind;
        train_ind(test_ind) = [];
    end
    
    % Get test/training data for DMDc
    X1 = X(:, train_ind);
    X2 = X(:, train_ind + 1);
    U1 = U(:, train_ind);
    X1_t = X(:, test_ind);
    X_end_t = X(:, test_ind+err_steps);
    U1_t = U(:, test_ind(1):(test_ind(end)+err_steps-1));
    
    % Build DMDc model
    AB = X2/[X1; U1];
    A = AB(:, 1:n);
    B = AB(:, (n+1):end);
    
    % Calculate error
%     all_err(i) = norm(A*X1_t + B*U1_t - X2_t,'fro');
    X_hat = X1_t;
    for i2 = 1:err_steps
        X_hat = A*X_hat + B*U1_t(:, i2:(end-(err_steps-i2)));
        if is_inclusive
            all_err(i) = all_err(i) + ...
                norm(X(:, test_ind+i2) - X_hat, 'fro') / err_steps;
        end
    end
    if ~is_inclusive
        all_err(i) = norm(X_end_t - X_hat, 'fro');
    end
end

err = mean(all_err);
end

