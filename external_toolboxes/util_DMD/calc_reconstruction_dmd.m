function x = calc_reconstruction_dmd(x0, m, A, B, u)
% Calculates a reconstructed dataset from linear dynamics 
%   Note: uses matrix multiplication
if ~exist('u', 'var')
    u = nan;
    assert(~isempty(m), 'Must have controller or number of frames set')
    assert(~exist('B', 'var'), ...
        'Control matrix set without passing control signal...')
elseif ~exist('m', 'var') || isempty(m)
    m = size(u,2) + 1;
end
n = length(x0);
x = zeros(n, m);
x(:,1) = x0;

% Looops brother
if ~isnan(u)
    for i = 2:m
        x(:, i) = A*x(:, i-1) + B*u(:, i-1);
    end
else
    for i = 2:m
        x(:, i) = A*x(:, i-1);
    end
end
end

