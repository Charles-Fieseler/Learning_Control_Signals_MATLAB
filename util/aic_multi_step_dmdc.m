function [aic_vec] = aic_multi_step_dmdc(dat, U, A, B, num_steps, do_aicc, ...
    formula_mode)
% Calculates the NEGATIVE aic for a DMDc model given data and basing the 
% number of parameters on the number of nonzero terms in U, and follows:
%   https://en.wikipedia.org/wiki/Akaike_information_criterion#Comparison_with_least_squares
%
%   Note: this uses the 2-step error as the residual term
% X1 = dat(:, 1:end-2);
% X3 = dat(:, 3:end);
%
% Input:
%   dat - the data to analyze
%   U - the control signal
%   A - the intrinsic dynamics matrix. Note: (along with matrix B) can be
%       calculated directly from data using MP inverse
%   B - the mapping from control signal to data space. Note: can be
%       calculated directly from data using MP inverse
%   num_steps (2) - the number of steps to predict in order to calculate the
%       error
%   do_aicc (false) - to calculate AICc, a second order correction
%   formula_mode ('stanford') - I've found several different formulas,
%       which don't seem to be entirely equivalent...
if ~exist('num_steps', 'var') || isempty(num_steps)
    num_steps = 2;
end
if ~exist('formula_mode', 'var')
    formula_mode = 'stanford';
end
X1 = dat(:, 1:(end-num_steps));
if ~exist('A', 'var') || isempty(A)
    % Do EXACT, 1-step DMDc to get both A and B
    X2 = dat(:, 2:end);
    n = size(X2, 1);
    
    AB = X2/[dat(:, 1:end-1); U];
    A = AB(:, 1:n);
    B = AB(:, (n+1):end);
end
if ~exist('do_aicc', 'var') || isempty(do_aicc)
    do_aicc = false;
end

RSS = calc_nstep_error(dat, A, B, U, num_steps, true);

num_signals = size(U, 1);
k = nnz(U) / num_signals;
n = size(X1, 2);

for i = 1:length(num_steps)
    aic_vec(i) = my_aic(formula_mode, do_aicc, {});
end

% 
% Note: used reduced k originally because it doesn't change the difference

% p = size(A,1);
% k_d = (k + p + size(B,2))/p; % i.e. per dimension, including A and B matrices
% Formula using p=dim(A) 
% multi_aicc = aic + (2*n*(num_signals*k + p*(p + 1)/2)) / (n - (k + p + 1));
% Formula using p=dim(U)
% p = num_signals;
% multi_aicc = aic + (2*n*(p*k + p*(p + 1)/2)) / (n - (k + p + 1));
end

