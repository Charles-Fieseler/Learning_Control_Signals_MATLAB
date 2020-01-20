function [this_U] = get_signal_actuating_neuron(...
    neur, text_names, registered_lines, all_U1, not_neur)
% Helper function to get the actual signal that is actuating a neuron from
% a neuron name. Checks across ranks for the highest quality signal (as a
% default, quality is measured by autocorrelation)

registration_ind = contains(text_names, neur);
if exist('not_neur', 'var')
    registration_ind = logical(registration_ind.*...
        ~contains(text_names, not_neur));
end
registration_ind = find(registration_ind);
    
max_y = 0;
rank_ind = 0;
max_line = [];
for i = 1:length(registration_ind)
    this_line = registered_lines{registration_ind(i)};
    [y, ind] = max(this_line{:, 'this_y'}); % Max over ranks AND sparsity
    if y > max_y
        max_y = y;
        max_line = this_line;
        rank_ind = ind;
    end
end
line_ind = max_line{rank_ind, 'which_line_in_rank'};
rank_ind = max_line{rank_ind, 'which_rank'};
this_U = all_U1{rank_ind}(line_ind, :);
end