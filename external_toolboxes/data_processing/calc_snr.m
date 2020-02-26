function [snr_vec, dat_signal, dat_noise] = calc_snr(dat, r)
% Uses the optimal truncation rank for the SVD decomposition of the data to
% get a rough separation between noise and signal; the snr is then the
% ratio of these two variances
%
% Input:
%   dat - the data (row = data channel, column = time)
%   r (optimal_truncation function) - the truncation rank
%
% Output:
%   snr_vec - the vector of signal to noise ratios for each row
%   dat_signal - the matrix determined to be signal
%   dat_noise - the residual (noise) matrix
if exist('r', 'var')
    [dat_signal, U, S, V] = svd_truncate(dat, r);
else
    [r, ~, dat_signal, U, S, V] = optimal_truncation(dat);
end

ind = (r+1):size(U,2);
dat_noise = U(:,ind) * S(ind,ind) * V(:,ind)';

snr_vec = var(dat_signal-mean(dat_signal,2), 0, 2) ./...
    var(dat_noise-mean(dat_noise,2), 0, 2);
end

