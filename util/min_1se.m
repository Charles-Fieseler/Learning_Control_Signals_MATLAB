function [min_1se_ind] = min_1se(dat, se)
% Calculates the minimum value that is at most one standard deviation above
% the minimum error, but simpler. Simpler is defined as a larger index in
% the 'dat' vector

[min_val, min_i] = min(dat);
min_plus = min_val + se(min_i);

% second_half_dat = dat(min_i:end);
% min_plus_second_half_ind = find(second_half_dat > min_plus, 1);
min_1se_ind = find(dat < min_plus, 1, 'last');
% if isempty(min_plus_second_half_ind)
%     % May not be any values greater than 1se above the min
%     min_plus_second_half_ind = length(second_half_dat);
% else
%     min_plus_second_half_ind = min_plus_second_half_ind - 1;
% end
% % Put back into original index coordinates
% min_1se_ind = min_plus_second_half_ind + min_i;

end

