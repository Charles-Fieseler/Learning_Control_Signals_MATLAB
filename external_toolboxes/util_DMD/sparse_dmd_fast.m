function [A, varargout] = sparse_dmd_fast(dat, s)
% Uses regular backslash with iterative least squares thresholding
%
% Input:
%   dat - the data; either a matrix or a cell array of {X1, X2}; rows are
%       channels and columns are time
%   settings - struct of settings

%---------------------------------------------
% Set options
%---------------------------------------------
if ~exist('s', 'var')
    s = struct();
end
if isnumeric(dat)
    X1 = dat(:,1:end-1);
    X2 = dat(:,2:end);
elseif iscell(dat)
    X1 = dat{1};
    X2 = dat{2};
end
defaults = struct(...
    'verbose', true,...
    'threshold', NaN,...
    'max_iter', 2);
for key = fieldnames(defaults).'
    k = key{1};
    if ~isfield(s, k)
        s.(k) = defaults.(k);
    end
end

%---------------------------------------------
% Loop through and sparsify
%---------------------------------------------
A = X2/X1;
if isnan(s.threshold)
    s.threshold = median(median(abs(A)));
end
all_A = cell(s.max_iter,1);
if nargout > 1
    all_err = zeros(s.max_iter,1);
end
sz = size(A);

for i = 1:s.max_iter
    A(abs(A)<1e-10) = 0;
    all_A{i} = A;
    sparsity_pattern = abs(A)>s.threshold;
    A = zeros(size(A));
    if nargout > 1
        all_err(i) = calc_reconstruction_error(A, X1);
    end
    
    for row = 1:sz(1)
        ind = sparsity_pattern(row,:);
        A(row,ind) = X2(row,:) / X1(ind,:);
    end
end

%---------------------------------------------
% Post-process for export
%---------------------------------------------
A(abs(A)<1e-10) = 0;
all_A{i} = A;
varargout{1} = all_A;
if length(varargout)>1
    all_err(i) = calc_reconstruction_error(A, X1);
    varargout{2} = all_err;
end

%---------------------------------------------
% Internal function for reconstruction error
%---------------------------------------------
    function err = calc_reconstruction_error(A, dat)
        approx = zeros(dat);
        approx(1,:) = dat(1,:);
        for i2 = 2:size(dat,2)
            approx(i2,:) = A*approx(i2-1,:);
        end
        err = norm(approx-dat,2);
    end

end

