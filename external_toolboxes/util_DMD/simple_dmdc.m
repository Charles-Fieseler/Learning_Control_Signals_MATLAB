function [A, B, dat_dmd] = simple_dmdc(dat, U)
% Simple L2 DMDc, i.e. multiple linear regression


% Calculate the A and B matrices with the 'ideal' signal
X2 = dat(:, 2:end);
n = size(X2, 1);

AB = X2/[dat(:, 1:end-1); U];
A = AB(:, 1:n);
B = AB(:, (n+1):end);

% Reconstruction
if nargout > 2
    dat_dmd = zeros(size(dat));
    dat_dmd(:,1) = dat(:, 1);
    for i = 2:size(dat, 2)
        dat_dmd(:, i) = A*dat_dmd(:, i-1) + B*U(:, i-1);
    end
end

end