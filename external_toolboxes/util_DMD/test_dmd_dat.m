function [X, A] = test_dmd_dat(n, m, noise, eigenvalue_min, seed)
% Produces test data generated according to a linear system

%% Defaults
assert(mod(n,2)==0, ...
    "Must pass a size that is even")
if ~exist('noise', 'var')
    noise = 0.0;
end
if ~exist('eigenvalue_min', 'var')
    eigenvalue_min = 0.99;
end
if ~exist('seed', 'var')
    seed = 13;
end

rng(seed);
%==========================================================================

%% Generate Dynamics matrix

%---------------------------------------------
% Generate random eigenvalues
%---------------------------------------------
%   From: https://www.mathworks.com/matlabcentral/answers/294-generate-random-points-inside-a-circle
% [V, D] = eig(rand(n), 'vector');
theta = rand(1,n/2)*(2*pi);
annulus = 1.0 - eigenvalue_min;
r = sqrt(eigenvalue_min + annulus*rand(1,n/2));
x = r.*cos(theta);
y = r.*sin(theta);
% Eigenvalues should be complex conjugates
D = zeros(n,1);
sign = [-1, 1];
for i = 1:n/2
    for i2 = 1:2 % Plus or minus
        D(2*i+i2-2) = x(i) + 1i*sign(i2)*y(i);
    end
end
% D = [x + 1i*y, x - 1i*y];

%---------------------------------------------
% Generate random eigenvectors
%---------------------------------------------
% Random matrix, for eigenvectors
% [V, ~] = eig(rand(n), 'vector');
% Clamp eigenvalues to be stable, but not too small
% D = D./min( max( x,eigenvalue_min ),1.0 );
tmp = rand(n);
[V, ~] = eig(tmp - tmp');

%---------------------------------------------
% Finally, build the full matrix
%---------------------------------------------
A = real(V*diag(D)/V);
%==========================================================================

%% Produce data
% Random starting point
x0 = rand(n,1);
X = calc_reconstruction_dmd(x0, m, A);

% Add noise
X = X + noise*randn(size(X));
%==========================================================================

end

