function [all_err, out] = calc_control_realness(my_model)
% Calculates two different L2 errors for a model with control signals:
%   A) the control signals are real
%   B) the control signals are actually noise, and should be subtracted
%
% Note: made to work with CElegansModel objects
out = struct();

num_tests = my_model.dat_sz(2) - 2;
all_err = zeros(num_tests, 2);
obj = my_model.AdaptiveDmdc_obj;
B = obj.A_separate(1:obj.x_len, obj.x_len+1:end);
for i = 1:num_tests
    % Calculate the prediction assuming u1 is a real controller, and that
    % x2 is really the starting point
    x2 = my_model.dat(:, i+1);
    u2 = my_model.control_signal(:, i+1);
    x3_ctr = obj.calc_reconstruction_manual(x2, u2);
    
    % Calculate the prediction assuming u1 is a noise term (mostly), and
    % that x2-B*u1 is the starting point
    u1 = my_model.control_signal(:, i);
    x3_noise = obj.calc_reconstruction_manual(x2-B*u1, u2);
    
    % Calculate errors
    x3 = my_model.dat(:, i+2);
    
    all_err(i, :) = [vecnorm(x3_ctr-x3), vecnorm(x3_noise-x3)];

%     delta_ctr = x3_ctr-x3;
%     delta_noise = x3_noise-x3;
%     [~, ind_ctr] = max(abs(delta_ctr));
%     val_ctr = delta_ctr(ind_ctr);
%     [~, ind_noise] = max(abs(delta_noise));
%     val_noise = delta_noise(ind_noise);
%     all_err(i, :) = [val_ctr, val_noise];
%     out.ind(i, 1:2) = [ind_ctr, ind_noise];
end

all_err = [all_err; [0, 0]; [0, 0]];

end

