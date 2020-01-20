function [err_vec, rel_err, thresh_vec] = ...
    calc_U_separation(X, B, U, err_steps)
% Given a control signal matrix U, attempt to separate it out into two
% pieces:
%   noise, i.e.:
%       X1_true = X1 - B_n*U0_n
%       X2_true = A*X1_true + B_n*U1_n
%   true controllers, i.e.:
%       X1_true = X1
%       X2_true = A*X1 + B*U1_true
%
% Algorithmically, this means trying different hard thresholds to separate
% out the given U into U_true=U_t and U_noise=U_n, and comparing the L2
% error:
%   err = (X2 - B_n*U1_n) - (A*(X1 - B_n*U0_n) + B*U1_true)
%
%   Note: for now, I'm doing the test on the entire matrix without worrying
%   about return U_true and B or B_n

if ~exist('B', 'var') || isempty(B)
    % Do EXACT, 1-step DMDc to get both A and B
    X2 = X(:, 2:end);
    n = size(X2, 1);
    
    AB = X2/[X(:, 1:end-1); U];
%     A = AB(:, 1:n);
    B = AB(:, (n+1):end);
end

max_val = max(max(U));
min_val = min(min(U(U>0)));
n = 20;
thresh_vec = linspace(min_val, max_val, n);
% thresh_vec = linspace(min_val, max_val, n+2);
% thresh_vec = thresh_vec(2:end-1);
err_vec = zeros(size(thresh_vec));
rel_err = err_vec;

% Skip the first step
X1 = X(:, 2:end-1);
X2 = X(:, 3:end);

% Work with the full matrices... TODO
U_mat = B*U;

for i = 1:n
    U_n = U_mat;
    U_n(abs(U_n)>thresh_vec(i)) = 0;
    U_t = U_mat;
    U_t(abs(U_t)<thresh_vec(i)) = 0;
    
    U0_n = U_n(:, 1:end-1);
    U1_n = U_n(:, 2:end);
    U1_true = U_t(:, 2:end);

    X2_true = X2 - U1_n; % Includes the matrix B, so U_n is propagated
%     X1_true = X1 - U0_n;
    X1_true = X1;
    this_A = (X2_true - U1_true) / X1_true;
    
    if err_steps == 1
        
        err_vec(i) = norm(X2_true - (this_A*X1_true + U1_true),...
            'fro');
        rel_err(i) = err_vec(i) / norm(X2_true, 'fro');
    else
        
        X_true = X(:, 2:end) - U_n;
        
        X_hat = X_true(:, 1:end-err_steps);
        X_end = X_true(:, (err_steps+1):end);
        for i2 = 1:err_steps
            X_hat = this_A*X_hat + U1_true(:, i2:(end-(err_steps-i2)));
        end
        err_vec(i) = norm(X_end - X_hat, 'fro');
        rel_err(i) = err_vec(i) / norm(X_end, 'fro');
    end
end
end

