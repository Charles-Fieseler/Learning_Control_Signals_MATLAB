function RSS = calc_nstep_error_old(X, A, B, U, num_steps, is_inclusive)
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
end
if ~is_inclusive
    RSS = norm(X_end - X_hat, 'fro');
end

end

