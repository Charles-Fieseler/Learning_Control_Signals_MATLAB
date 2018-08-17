function [dat, ctr_signal, state_vec] = ...
    test_dat_pid(sz, kp, ki, kd, set_points, transition_mat,...
    perturbation_mat,...
    x0, z0)
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
%
% Output:
%   dat - matrix of size sz
%   ctr_signal - control signal, size=[1, sz(1)], i.e. the state vector

if ~exist('x0','var')
    x0 = rand([sz(1),1]);
end
if ~exist('z0','var')
    z0 = randi(size(transition_mat,1));
end

assert(isempty(kd), 'Derivative control not implemented')
assert(size(set_points,1)==sz(1),...
    'Set points must be the same size as the system')
num_states = size(set_points,2);
assert(size(transition_mat,1)==num_states,...
    'Set points must be equal to the number of states')
assert(size(kp,2)==num_states,...
    'Controller coefficients must be equal to the number of states')
assert(size(ki,2)==num_states,...
    'Controller coefficients must be equal to the number of states')

seed = 1;
rng(seed);

dat = zeros(sz);
dat(:,1) = x0;
int_error = zeros(sz);

state_vec = zeros([1,sz(2)]);
state_vec(1) = z0;
for i = 2:sz(2)
    z = state_vec(i-1);
    x = dat(:,i-1);
    % Add PID control signal
    % TODO: intrinsic dynamics as well
    [u, int_error(:,i)] = calc_pid_ctr_signal(...
        x, kp(z), ki(z), set_points(:,z), int_error(:,i-1));
    dat(:,i) = x + u + perturbation_mat(:,i-1);
    % Transition states
    [state_vec(i), did_switch] = ...
        calc_new_state(transition_mat, num_states, z);
    if did_switch
        int_error(:,i) = 0;
    end
end

ctr_signal = [int_error; perturbation_mat];

    function [u, int_error] = calc_pid_ctr_signal(x, kp, ki, sp, int_error)
        error = x-sp;
        int_error = int_error + error;
        u = -kp*error - ki*int_error;
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
                end
            end
            did_switch = true;
        else
            new_state = z;
            did_switch = false;
        end
    end
end

