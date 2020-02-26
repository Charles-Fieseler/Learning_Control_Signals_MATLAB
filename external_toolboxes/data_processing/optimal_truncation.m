function r = optimal_truncation(Y)
% "Optimal" hard truncation, from this paper:
%   https://arxiv.org/pdf/1305.5870.pdf

[m, n] = size(Y);
[U, D, V] = svd(Y); 
d = diag(D);
val = optimal_SVHT_coef(m/n,false);

r = floor(val*median(d));

% y = diag(Y); 
% y( y < (optimal_SVHT_coef_sigma_unknown(m/n,0) * median(y)) ) = 0; 
% Xhat = U * diag(y) * V';

end