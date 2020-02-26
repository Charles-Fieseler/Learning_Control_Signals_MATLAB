function cmap = cmap_white_zero(dat, cmap_func, num_colors)
% Uses the maximum and minimum values of 'dat' and the base 'cmap_func' to
% return an asymmetric colormap with 0 set to be the original middle
% color... designed for use with a base 'cmap_func' with white as the
% middle but not necessary
%   Note: compresses the side (positive or negative) to be the same color
%   range as the original
if ~exist('num_colors', 'var')
    num_colors = 10;
end
if ~exist('cmap_func', 'var')
    cmap_func = @(x) brewermap(2*x,'RdBu');
end

cmax = max(max(dat));
cmin = min(min(dat));
more_on_positive = abs(cmax/cmin) > abs(cmin/cmax);
if more_on_positive
    ratio = abs(cmax/cmin);
else
    ratio = abs(cmin/cmax);
end

one_side_extra_colors = floor(num_colors*ratio);
extra_colormap = cmap_func(one_side_extra_colors);
cmap = ones(one_side_extra_colors+num_colors+1,3);
if more_on_positive
    % Downsample the side with lower magnitude (positive)
    downsample = floor(linspace(1, one_side_extra_colors, num_colors));
    cmap(1:num_colors, :) = ...
        extra_colormap(downsample, :);
    % The side with higher magnitude data
    cmap((num_colors+2):end, :) = ...
        extra_colormap((one_side_extra_colors+1):end, :);
else
    % The side with higher magnitude data (positive)
    cmap(1:one_side_extra_colors, :) = ...
        extra_colormap(1:one_side_extra_colors, :);
    % Downsample the side with lower magnitude
    downsample = floor(linspace(1, one_side_extra_colors, num_colors));
    cmap(one_side_extra_colors+2:end, :) = ...
        extra_colormap(one_side_extra_colors+downsample, :);
end

end

