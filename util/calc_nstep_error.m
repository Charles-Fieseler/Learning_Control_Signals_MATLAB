function RSS = calc_nstep_error(X, A, B, U, num_steps, is_inclusive)
% Calculates the n-step error for a DMDc model
if ~exist('is_inclusive', 'var')
    is_inclusive = false;
end

X1 = X(:, 1:(end-num_steps));
X_end = X(:, (num_steps+1):end);
X_hat = X1;
RSS = 0;
for i = 1:num_steps
    X_hat = A*X_hat + B*U(:, i:(end-(num_steps-i)));
    if is_inclusive
        RSS = RSS + norm(X_hat - X(:, (i+1):(end-num_steps+i)),'fro');
    end
%    ((A^3)*X1 +...
%         A*A*B*U(:, 1:end-2) + A*B*U(:, 2:end-1) + A*B*U(:, 3:end)), 'fro');
end
if ~is_inclusive
    RSS = norm(X_end - X_hat, 'fro');
end

end

