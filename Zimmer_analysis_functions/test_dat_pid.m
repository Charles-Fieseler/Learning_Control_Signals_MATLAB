function [dat, ctr_signal, state_vec] = ...
    test_dat_pid(sz, kp, ki, kd, set_points, transition_mat,...
    perturbation_mat,...
    x0, z0, initial_condition_dependence, noise)
%Produces data for a pid-type system
%   Switches between different internal states via a HMM.
% Input:
%   sz - size of the data matrix [num channels, time steps]
%   kp - vector of proportional controller coefficients for EACH state
%   ki - vector of integral controller coefficients for EACH state
%   kd - vector of derivative controller coefficients for EACH state
%   set_points - vector of set points for EACH state
%   transition_mat - HMM transition matrix
%   perturbation_mat - vector for each channel perturbations
%   x0 - initial condition, size=sz(1) by 1
%   z0 - initial discrete state
%   initial_condition_dependence - whether any set points are dependent on
%               the state of the system at the time of initiation
%   noise - variance of Gaussian noise
%
% Output:
%   dat - matrix of size sz
%   ctr_signal - control signal, size=[1, sz(1)], i.e. the state vector

if ~exist('x0','var') || isempty(x0)
    x0 = rand([sz(1),1]);
end
if ~exist('z0','var') || isempty(z0)
    z0 = randi(size(transition_mat,1));
end
if ~exist('initial_condition_dependence','var')
    initial_condition_dependence = zeros(size(set_points));
end
if ~exist('noise','var')
    noise = 0;
end

assert(isempty(kd), 'Derivative control not implemented')
assert(size(set_points,1)==sz(1),...
    'Set points must be the same size as the system')
num_states = size(set_points,2);
assert(size(transition_mat,1)==num_states,...
    'Set points must be equal to the number of states')
assert(size(kp,1)==sz(1),...
    'Controller coefficients must be equal to the number of channels')
assert(size(ki,1)==sz(1),...
    'Controller coefficients must be equal to the number of channels')

seed = 1;
rng(seed);

dat = zeros(sz);
dat(:,1) = x0;
int_error = zeros(sz);
% int_error(:,1) = x0 - set_points(:,z0);

state_vec = zeros([1,sz(2)]);
state_vec(1) = z0;
this_set_point = set_points(:,z0) + x0.*initial_condition_dependence(:,z0);
for i = 2:sz(2)
    x = dat(:,i-1);
%     z = state_vec(i-1);
    % Add PID control signal
    % TODO: intrinsic dynamics as well
    [u, int_error(:,i)] = calc_pid_ctr_signal(...
        x, kp, ki, this_set_point, int_error(:,i-1));
    dat(:,i) = x + u + perturbation_mat(:,i-1);
    
    % Possibly transition states
    %   i.e. determine the state for the CURRENT timestep
    [state_vec(i), did_switch] = ...
        calc_new_state(transition_mat, num_states, state_vec(i-1));
    if did_switch
        int_error(:,i-1) = 0;
        z_i = state_vec(i);
        this_set_point = set_points(:,z_i) + ...
            dat(:,i).*initial_condition_dependence(:,z_i);
%         int_error(:,i) = x - set_points(:,z);
    end
end

ctr_signal = [int_error; perturbation_mat];

dat = dat + normrnd(0, noise, size(dat));

    function [u, int_error] = calc_pid_ctr_signal(x, kp, ki, sp, int_error)
        error = x - sp;
        int_error = int_error + error;
        u = -kp.*error - ki.*int_error;
    end

    function [new_state, did_switch] = ...
            calc_new_state(transition_mat, num_states, z)
        trans = rand();
        if trans >= transition_mat(z,z)
            current_prob = transition_mat(z,z);
            other_states = 1:num_states;
            other_states(z) = [];
            for i2 = other_states
                % Format: row=from, col=to
                current_prob = current_prob + transition_mat(z,i2);
                if trans < current_prob
                    new_state = i2;
                    break
                end
            end
            did_switch = true;
        else
            new_state = z;
            did_switch = false;
        end
    end
end

