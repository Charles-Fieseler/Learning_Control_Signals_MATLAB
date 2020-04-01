function [err, all_err] = dmdc_cross_val(X, U, num_folds, err_steps, ...
    cross_val_mode, is_inclusive)
% Does 'num_folds' cross validation with a DMDc framework
%   Can use error steps that are greater than 1
if ~exist('err_steps', 'var')
    err_steps = 1;
end
if ~exist('cross_val_mode', 'var') || isempty(cross_val_mode)
    cross_val_mode = 'chaining';
end
if ~exist('is_inclusive', 'var')
    is_inclusive = true;
    err_steps = 1:err_steps; % Save each step; average at the end
end


m = size(X, 2);
window_starts = round(linspace(1, m, num_folds+1));
if strcmp(cross_val_mode, 'chaining')
    window_starts(1) = [];
    num_folds = num_folds - 1;
end

% Return a set of error evaluations
if length(err_steps) > 1
    err_steps_to_save = err_steps;
    err_steps = err_steps(end); % This should be a scalar
    all_err = zeros(length(err_steps_to_save), num_folds);
else
    all_err = zeros(1, num_folds);
    err_steps_to_save = err_steps;
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
%     X_end_t = X(:, test_ind+err_steps);
%     U1_t = U(:, test_ind(1):(test_ind(end)+err_steps-1));
    U1_t = U(:, test_ind(1):(test_ind(end)-1));
    
    % Build DMDc model
    AB = X2/[X1; U1];
    A = AB(:, 1:n);
    B = AB(:, (n+1):end);
    
    % Calculate error
    
    all_err = calc_nstep_error(X1_t, A, B, U1_t, ...
        err_steps_to_save, false);
%     X_hat = X1_t;
%     for i2 = 1:err_steps
%         X_hat = A*X_hat + B*U1_t(:, i2:(end-(err_steps-i2)));
%         if ismember(i2,err_steps_to_save)
%             save_ind = ismember(err_steps_to_save, i2);
%             all_err(save_ind, i) = norm(X(:, test_ind+i2) - X_hat, 'fro');
%         end
%     end
end
if is_inclusive
    % Average over saved steps
    all_err = mean(all_err,1);
end

err = mean(all_err);
end

