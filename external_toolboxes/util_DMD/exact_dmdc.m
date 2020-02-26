function [A, B] = exact_dmdc(dat, U)
% Least squares DMDc
X2 = dat(:, 2:end);
n = size(X2, 1);

AB = X2/[dat(:, 1:end-1); U];
A = AB(:, 1:n);
B = AB(:, (n+1):end);
end

