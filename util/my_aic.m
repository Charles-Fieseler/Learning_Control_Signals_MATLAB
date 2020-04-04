function [aic_out] = my_aic(formula_mode, do_aicc, RSS, k, n, num_signals,...
    A, B, U, X1)
% Calculates AIC or AICc using various formulas. The main difference
% between the formulas is whether the variance of the error is explicitly
% calculated
if ~exist('formula_mode', 'var')
    formula_mode = 'standard';
end

if strcmp(formula_mode, 'standard')
%     n = numel(X1);
%     aic_out = 2*num_signals*(k + n) + n*log(RSS); % From wikipedia
    aic_out = 2*num_signals*k + 2*n + n*log(RSS); % The 2*n usually doesn't matter
elseif strcmp(formula_mode, 'multivariate')
    % Note: AICc probably won't work here
    n = size(X1, 1);
    aic_out = 2*num_signals*k + 2*n + n*log(RSS); % The 2*n usually doesn't matter
elseif strcmp(formula_mode, 'stanford')
    % X2 = dat(:, 2:end);
    % err_cov = norm(calc_nstep_error(dat, X2/dat(:, 1:end-1),...
    %     zeros(size(X2,1),1), zeros(1, size(X2,2)), num_steps),'fro') / n; % For now, use a one-step model
    % err_cov = norm(calc_nstep_error(dat, X2/dat(:, 1:end-1),...
    %     zeros(size(X2,1),1), zeros(1, size(X2,2)), 1),'fro') / n;

    % Error covariance guess: 1/2 the no-control version
    % err_cov = norm(calc_nstep_error(dat, X2/dat(:, 1:end-1),...
    %     zeros(size(X2,1),1), zeros(1, size(X2,2)), num_steps),'fro') / n;
    % err_cov = 0.005;

    % Error covariance guess: the svd truncated modes
    [~, ~, dat_noise] = calc_snr(dat);
    err_cov = norm(dat_noise, 'fro') / n;
    aic_out = 2*k*num_signals + RSS/err_cov; % From Stanford notes
    % aic = 2*(k*num_signals + size(X1,1)*num_signals) + RSS/err_cov; % From Stanford notes
    
elseif strcmp(formula_mode, 'stanford2')
    % Same as above formula, but I think they missed a factor of 2...
    [~, ~, dat_noise] = calc_snr(dat);
    err_cov = norm(dat_noise, 'fro') / n;
    aic_out = 2*k*num_signals + 2*RSS/err_cov; % From Stanford notes
    
elseif strcmp(formula_mode, 'one_step')
    % Assume that these parameters are different from the regular AIC ones,
    % and should only count as a "degree of freedom" / number of snapshots
    
    aic_out = 2*num_signals*(k/n + n) + n*log(RSS); % From wikipedia
elseif strcmp(formula_mode, 'one_step_stanford')
    % Error covariance guess: the svd truncated modes
    [~, ~, dat_noise] = calc_snr(dat);
    err_cov = norm(dat_noise, 'fro') / n;
    aic_out = 2*k*num_signals/n + RSS/err_cov; % From Stanford notes
else
    error('Unrecognized mode')
end

%---------------------------------------------
% Corrections; experimental
%---------------------------------------------
% First, AICc
if do_aicc
    k_t = nnz(U) + nnz(A) + nnz(B); % Total number of parameters
%     sz = size(X1);
%   d = sz(2) / (sz(2) - (a/num_signals + a/sz(2) + 1));
    correction = (2*k_t.^2 + 2*k_t) / abs(n - k_t - 1);
%     correction = (2*k_t.^2 + 2*k_t) / (numel(X1) - k_t - 1);
    aic_out = aic_out + correction;
    if k_t > n
        warning('Number of parameters is larger than data; AICc may be difficult to interpret')
    end
    % TODO: k_t is usually much larger than n!!
end

end

