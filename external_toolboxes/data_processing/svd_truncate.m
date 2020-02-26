function [dat, U, S, V] = svd_truncate(dat, r)
% Truncates data to rank r using SVD
%
% Input:
%   dat - the data matrix to be truncated
%   r - the truncation rank; if 0 then this function returns
%
% Output:
%   dat - the truncated dat
%   [U, S, V] - the output of economical svd()

if r == 0
    return
end

[U, S, V] = svd(dat,'econ');
dat = U(:,1:r)*S(1:r,1:r)*V(:,1:r)';
end

