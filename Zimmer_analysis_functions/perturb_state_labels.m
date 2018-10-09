function u_new = perturb_state_labels(u,...
    can_overlap, max_perturb, num_perturb_single_u, num_draws, seed)
% Given a control signal 'u', perturb the lengths of the states
% randomly (up to 'max_perturb' number of frames) 'num_draws' number of
% times
% Input:
%   u - control signal; format: rows = channels; columns = time slices
%   can_overlap (true) - if one control signal is extended, should the
%       signal that was previously on at that time be on as well?
%   max_perturb - max num frames to perturb
%   num_perturb_single_u (1) - number of times each returned u_new signal
%       should be perturbed 
%
% Output:
%   u_new - a cell array of all the new control signals
sz = size(u);
if ~exist('can_overlap', 'var')
    can_overlap = false;
end
if ~exist('max_perturb', 'var')
    max_perturb = round(sz(2)/1000);
end
if ~exist('num_perturb_single_u', 'var')
    num_perturb_single_u = 1;
end
if ~exist('num_draws', 'var')
    num_draws = round(sz(1)*max_perturb);
end
if ~exist('seed', 'var')
    seed = 13;
end
rng(seed);

u_new = cell(num_draws,1);
u_diff = diff(u, [], 2);
u_diff_ind = find(u_diff);
for i = 1:num_draws
    ind = u_diff_ind(randperm(length(u_diff_ind)));
    this_diff = u_diff;
    for i2 = 1:num_perturb_single_u
        this_ind = ind(i2);
        % Get the number of + or - frames, and check if we're on the edges
        pert = randi(max_perturb)*(2*randi(2)-3);
        if pert + this_ind > sz(2) || pert + this_ind < 1
            pert = -pert;
        end
        % If we extend it forward, we want to add zeros; if we are passing
        % another transition, we simply absorb it and the intermediate
        % state is gone (or multiple intermediate states)
        if pert > 0
            sweep_ind = this_ind:(this_ind+pert);
            this_diff(this_ind+pert) = sum(this_diff(sweep_ind));
            this_diff(sweep_ind(1:end-1)) = 0;
        else
            sweep_ind = (this_ind+pert):this_ind;
            this_diff(this_ind+pert) = sum(this_diff(sweep_ind));
            this_diff(sweep_ind(2:end)) = 0;
        end
    end
    u_new{i} = cumsum([u(1) this_diff]);
end

end

