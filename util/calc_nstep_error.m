function all_err = calc_nstep_error(X, A, B, U, num_steps, is_inclusive)
% Calculates the n-step error for a DMDc model
%   NOTE: this returns the error for all steps if num_steps is a vector,
%   but uses the amount of data for the last step, i.e. less than if you
%   just had one step error
if ~exist('is_inclusive', 'var')
    is_inclusive = false;
end

% Return a set of error evaluations
if length(num_steps) > 1
    err_steps_to_save = num_steps;
    num_steps = num_steps(end); % This should be a scalar
    all_err = zeros(length(err_steps_to_save), 1);
else
    all_err = 0;
    err_steps_to_save = num_steps;
end

% Set up model and initial conditions
if isnumeric(X)
    start_ind = 1:size(X,2) - num_steps;
    end_ind = 2:size(X,2) - num_steps + 1;
    X1 = X(:, start_ind);
    X2 = X(:, end_ind);
% elseif iscell(X)
%     [X1, X2] = X{:};
% %     start_ind = 1:size(X1,2) - num_steps + 1;
%     end_ind = 2:size(X1,2) - num_steps;
else
%     error('Must pass a data matrix or cell array of two matrices for X')
    error('Must pass a data matrix for X')
end
if ~exist('A', 'var')
    AB = X2/[X1; U];
    A = AB(:, 1:n);
    B = AB(:, (n+1):end);
end

X_hat = X1;
for iStep = 1:num_steps
    X_hat = A*X_hat + B*U(:, iStep:(end-(num_steps-iStep)));
    if ismember(iStep,err_steps_to_save)
        save_ind = ismember(err_steps_to_save, iStep);
        all_err(save_ind) = norm(X_hat - X(:, end_ind+iStep-1), 'fro');
    end
end

if is_inclusive
    % Average over saved steps
    all_err = mean(all_err,1);
end



% X1 = X(:, 1:(end-num_steps));
% X_end = X(:, (num_steps+1):end);
% X_hat = X1;
% RSS = 0;
% for i = 1:num_steps
%     X_hat = A*X_hat + B*U(:, i:(end-(num_steps-i)));
%     if is_inclusive
%         RSS = RSS + norm(X_hat - X(:, (i+1):(end-num_steps+i)),'fro');
%     end
% %    ((A^3)*X1 +...
% %         A*A*B*U(:, 1:end-2) + A*B*U(:, 2:end-1) + A*B*U(:, 3:end)), 'fro');
% end
% if ~is_inclusive
%     RSS = norm(X_end - X_hat, 'fro');
% end

end

