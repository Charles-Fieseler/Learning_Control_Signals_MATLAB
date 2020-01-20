function [falloff_time, err_vec] = ...
    calc_falloff_time(my_model, dat_ind, func, max_step, tol, ind_to_plot)
% Calculates the falloff time for a model to tolerance 'tol' up to maximum
% step length 'max_step'
%
% The interface expected for 'my_model' is that it can be called with a
% method 'calc_reconstruction_control' to produce the reconstruction
if ~exist('func', 'var') || isempty(func)
    func = @calc_reconstruction_control;
end
if ~exist('max_step', 'var')
    max_step = 10;
end
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-2;
end
if ~exist('ind_to_plot', 'var')
    ind_to_plot = 0;
end

dat = my_model.dat(:, dat_ind);
x0 = dat(:,1);
% recon_ind = 1:max_step;
recon = real(func(my_model, x0, dat_ind));

% err_vec = vecnorm(dat(:, recon_ind) - recon, 2, 1) / size(dat, 1);
err_vec = vecnorm(dat - recon, 2, 1) / size(dat, 1);

err_ind = err_vec > tol;
falloff_time = find(err_ind, 1);
if isempty(falloff_time)
    falloff_time = max_step;
end

if ind_to_plot > 0
%     figure;
    plot(dat(ind_to_plot, :))
    hold on
    plot(recon(ind_to_plot, :))
    plot(falloff_time, mean([recon(ind_to_plot, falloff_time),...
        dat(ind_to_plot, falloff_time)]), '*', 'LineWidth', 3)
    legend('Data', 'Reconstruction', 'Falloff time')
    
end

end

