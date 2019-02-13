


%% Use cvx to do sparse DMD
% Import data
filename = '../../Collaborations/Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
dat5 = importdata(filename);
this_dat = dat5.traces';
X1 = this_dat(:,1:end-1);
X2 = this_dat(:,2:end);

% Do naive dmd as a check
A_original = X2/X1;

% solver variables
n = size(this_dat,1);
gamma = 0.1;

cvx_begin
    variable A_sparse(n,n)
    minimize( norm(A_sparse*X1-X2,2) + gamma*norm(A_sparse,1) )
cvx_end


%==========================================================================


%% cvx tests: sparsity in matrices
n = 20;
tol = 0.9;
rng default;
X1 = rand(n,5*n);
A_true = rand(n,n);
% Make the matrix and data sparse
X1(abs(X1)<tol) = 0;
A_true(abs(A_true)<tol) = 0;
spread = 3*(2*rand(size(A_true))-1);
spread(abs(spread)<0.5) = 0; % Make sure no entries are noise-level
A_true = A_true.*spread;
% This isn't necessarily sparse and is used as data
noise = 0.5;
min_tol = 2*noise + 1e-6;
X2 = A_true*X1 + normrnd(0.0,noise,size(X1));
gamma_list = linspace(0,noise*10,11);
A_sparse_nnz = zeros(size(gamma_list));
for i=1:length(gamma_list)
    % sparsity term
    gamma = gamma_list(i);
    % Actually solve
    cvx_begin quiet
        variable A_sparse(n,n)
        minimize( norm(A_sparse*X1-X2,2) + ...
            gamma*norm(A_sparse,1))
%          + (1-gamma)*norm(A_sparse,2)
    cvx_end
    % Postprocess a bit because there are some tiny terms that survive
    A_sparse(abs(A_sparse)<min_tol) = 0;
    
    A_sparse_nnz(i) = nnz(A_sparse);
    if  (A_sparse_nnz(i) == nnz(A_true)) ||...
            A_sparse_nnz(i) > A_sparse_nnz(1)
        A_sparse_nnz(i+1:end) = NaN;
        break
    end
    
end
% Also get backslash solution for comparison
A_matlab = X2/X1;
A_matlab(abs(A_matlab)<min_tol) = 0;
% Plot
figure;
plot(gamma_list, nnz(A_true)*ones(size(gamma_list)), 'LineWidth',2)
hold on
plot(gamma_list, A_sparse_nnz, 'o')
plot(gamma_list, nnz(A_matlab)*ones(size(gamma_list)), '--',...
    'LineWidth',2)
title('Number of nnz elements vs. sparsity penalty')
legend('nnz of fit', 'nnz of backslash')

figure;
subplot(2,1,1);
spy(A_sparse);
title(sprintf('Solved with gamma=%.1f',gamma))
subplot(2,1,2);
spy(A_true);
title('True matrix')
plot_2imagesc_colorbar(A_true,A_sparse,'2 1')
%figure;
%subplot(2,1,1);spy(X1);
%subplot(2,1,2);spy(X2);
%==========================================================================


%% cvx tests: sequential thresholding for sparsity in matrices
n = 50;
tol = 0.9;
rng default;
X1 = rand(n,5*n);
A_true = rand(n,n);
% Make the matrix and data sparse
X1(abs(X1)<tol) = 0;
A_true(abs(A_true)<tol) = 0;
spread = 3*(2*rand(size(A_true))-1);
spread(abs(spread)<0.5) = 0; % Make sure no entries are noise-level
A_true = A_true.*spread;
% This isn't necessarily sparse and is used as data
noise = 0.5;
% min_tol = 2*noise + 1e-6;
min_tol = 3e-1;
max_iter = 10;
X2 = A_true*X1 + normrnd(0.0,noise,size(X1));
% gamma_list = linspace(0,noise*10,2);
gamma_list = [0.1];
A_sparse_nnz = zeros(size(gamma_list));
% sparsity_pattern = abs(A_true)==0;
sparsity_pattern = false(size(A_true));
num_nnz = zeros(max_iter,1);
for i=1:length(gamma_list)
    % sparsity term
    gamma = gamma_list(i);
    for i2=1:max_iter
        num_nnz(i2) = numel(A_true) - length(find(sparsity_pattern));
        fprintf('Iteration %d; %d nonzero-entries\n',...
            i2, num_nnz(i2))
        if i2>1 && (num_nnz(i2-1)==num_nnz(i2))
            disp('Stall detected; quitting early')
            break
        end
        % Actually solve
        cvx_begin quiet
            variable A_sparse(n,n)
%             minimize( norm(A_sparse*X1-X2,2) + ...
%                 gamma*norm(A_sparse,1))
            minimize( norm(A_sparse*X1-X2,2) )
            A_sparse(sparsity_pattern) == 0
        cvx_end
        % Postprocess a bit because there are some tiny terms that survive
        sparsity_pattern = abs(A_sparse)<min_tol;
%         A_sparse(abs(A_sparse)<min_tol) = 0;

        A_sparse_nnz(i) = nnz(A_sparse);
        if  (A_sparse_nnz(i) == nnz(A_true)) ||...
                A_sparse_nnz(i) > A_sparse_nnz(1)
            A_sparse_nnz(i+1:end) = NaN;
            break
        end
    end
    A_sparse(abs(A_sparse)<1e-6) = 0;
    
end
% Also get backslash solution for comparison
A_matlab = X2/X1;
A_matlab(abs(A_matlab)<min_tol) = 0;
% Plot
figure;
plot(gamma_list, nnz(A_true)*ones(size(gamma_list)), 'LineWidth',2)
hold on
plot(gamma_list, A_sparse_nnz, 'o')
plot(gamma_list, nnz(A_matlab)*ones(size(gamma_list)), '--',...
    'LineWidth',2)
title('Number of nnz elements vs. sparsity penalty')
legend('nnz of fit', 'nnz of backslash')

figure;
subplot(2,1,1);
spy(A_sparse);
title(sprintf('Solved with gamma=%.1f',gamma))
subplot(2,1,2);
spy(A_true);
title('True matrix')
plot_2imagesc_colorbar(A_true,A_sparse,'2 1')
%figure;
%subplot(2,1,1);spy(X1);
%subplot(2,1,2);spy(X2);
%==========================================================================


%% Toy data with oscillations

% One neuron, 3 states
use_oscillation = true;
num_neurons = 5;
t_steps = 2500;
sz = [num_neurons, t_steps];
% kp = [0.5];
kp = [0.5, 0.0, 0.0, 0.0, 0.0]';
ki = [0];
kd = [];
% set_points = [0 0.5, 1.3];
set_points = [0 2.0, 1.3]; % Intermediate "begin escape" state
num_states = length(set_points);
if num_neurons > 1
    if use_oscillation
        set_points = [set_points; ...
            NaN*ones([num_neurons-size(set_points,1), num_states])];
    else
        set_points = [set_points; ...
            rand([num_neurons-size(set_points,1), num_states])];
    end
%     if num_neurons == 5
%         set_points(4,1) = 0;
%     end
end

% Some dependence on the initial condition
initial_condition_dependence = zeros(size(set_points));
noise = 0.00;

% transition_mat = [[0.99 0.01 0.00];
%                   [0.01 0.98 0.01];
%                   [0.00 0.01 0.99]];
transition_mat = [[0.99 0.00 0.01];
                  [0.01 0.90 0.00];
                  [0.00 0.10 0.99]];
perturbation_mat = zeros(sz);
perturbation_mat(1,100) = 0.1;
perturbation_mat(1,300) = -0.1;
perturbation_mat(1,500) = 0.2;
perturbation_mat(1,700) = -0.2;
perturbation_mat(1,1000:1050) = 0.1;
perturbation_mat(1,2000:2050) = -0.5;

% Set up the intrinsic dynamics
A = eye(num_neurons);
if use_oscillation
    period = 50;
    theta = 2*pi/period;
    rot_mat = [[cos(theta), -sin(theta)];[sin(theta) cos(theta)]];
    % Allow the oscillator to oscillate about the value of neuron 1 by
    % using translation matrices
    trans_mat = [[1 0 0];[1 1 0];[1 0 1]];
    inv_trans_mat = [[1 0 0];[-1 1 0];[-1 0 1]];
    big_rot_mat = eye(3);
    big_rot_mat(2:3,2:3) = rot_mat;
    full_mat = trans_mat*big_rot_mat*inv_trans_mat;
    
    if num_neurons == 3
        disp('Using oscillator as neurons 2-3')
        % Add intrinsic decay and a connection to the first neuron
        rot_mat = rot_mat - [[1e-3, 0];[1e-3, 0]];
        A(2:3, 2:3) = full_mat(2:3,2:3);
        A(2:3, 1) = full_mat(2:3,1);
    elseif num_neurons == 5 && use_oscillation
        disp('Using oscillator as neurons 3-4')
        % First set up a lag neuron, then a diff neuron
        A(2,2) = 0;
        A(2,1) = 1;
        
        A(3,1) = 1;
        A(3,2) = -1;
        A(3,3) = 0;
        % Add small intrinsic decay
        full_mat(2:3,2:3) = full_mat(2:3,2:3) - (0.01)*eye(2);
%         full_mat(2:3,2:3) = full_mat(2:3,2:3) - (0.05)*[[1 0];[0 -0.1]];
        % Connect the oscillator to the derivative neuron
        A(4,3) = 0;
        
        A(4:5, 4:5) = full_mat(2:3,2:3);
        A(4:5, 1) = full_mat(2:3,1); % All other entries of full_mat should be 0
    end
end

% Finally: produce the toy data
[dat, ctr_signal, state_vec] = ...
    test_dat_pid(sz, kp, ki, kd, set_points, transition_mat,...
    perturbation_mat, A, [], 1, initial_condition_dependence, noise);

% Now do dmd model
ID = cell(num_neurons,1);
for i = 1:num_neurons
    ID{i} = num2str(i);
end
grad = gradient(dat');
dat_struct = struct(...
    'traces', dat',...
    'tracesDif',grad,...
    'ID',{ID},...
    'ID2',{ID},...
    'ID3',{ID},...
    'TwoStates', state_vec,...
    'TwoStatesKey',{{'State 1','State 2','State 3'}},...
    'fps',1);

augment_data = 0;
use_deriv = false;
ctr_signal = ctr_signal(:,1:end-augment_data);
ad_settings = struct('truncation_rank', -1,...
    'truncation_rank_control', -1,...
    'what_to_do_dmd_explosion', 'project');
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    ...'filter_window_dat',0,...
    'filter_window_dat',2,...
    'filter_window_global',0,...
    'add_constant_signal',false,...
    ...'dmd_mode','func_DMDc',...
    'dmd_mode','sparse',...
    ...'dmd_mode','optdmd',...
    'use_deriv',use_deriv,...
    'AdaptiveDmdc_settings',ad_settings,...
    'custom_control_signal',ctr_signal((num_neurons+1):end,:),... % Not using the integral error, only perturbations
    'lambda_sparse',0); % Don't want a sparse signal here
settings.global_signal_mode = 'ID_binary';
my_model_optdmd = CElegansModel(dat_struct, settings);

my_model_optdmd.plot_reconstruction_interactive();
%==========================================================================


%% Test out fbDMD on toy data

% Produce data
noise = 0.01;
seed = 13;
n = 30;
m = 1000;
eigenvalue_min = 0.95;
[dat, A_true] = test_dmd_dat(n, m, noise, eigenvalue_min, seed);

% Truncation rank
r = 0;

% First get the naive model
useFB = false;
[ ~, ~, ~, romA_naive, ~, ~, ~ ] = dmd( dat, 1,...
    r, [], false, useFB);

% Now compare to the FB model
useFB = true;
[ ~, ~, ~, romA_FB, ~, ~, ~ ] = dmd( dat, 1,...
    r, [], false, useFB);

% Next do an exponential DMD model
[A_exp] = exponential_dmd(dat, 10, false);

% Now do a geometric series model
[A_geo] = exponential_dmd(dat, 150, true, 0.9);

% Now do a geometric series model with FB
% [A_geo_fb] = exponential_dmd(dat, 150, true, 0.9, true);

% Now do an exponential series model with FB
% [A_exp_fb] = exponential_dmd(dat, 20, false, [], true);

% Plot eigenvalues
figure;
plot(eig(romA_naive, 'vector'),'o')
hold on
plot(eig(romA_FB, 'vector'), '*')
plot(eig(A_exp, 'vector'), '*')
plot(eig(A_geo, 'vector'), '*')
plot(eig(A_true, 'vector'), 'sk')
% viscircles([0,0],1, 'Color', 'k');
% plot(eig(A_exp_fb, 'vector'), '*')
% plot(eig(A_geo_fb, 'vector'), '*')
% legend({'Naive', 'FB', 'Exp', 'Exp_fb', 'Geometric', 'FB_geo', 'True'})
legend({'Naive', 'FB', 'Exp', 'Geometric', 'True'})
title(sprintf('Noise level: %.4f',noise))


%==========================================================================


%% Test out fbDMD on toy data (test via reconstructions)

% Produce data
noise = 0.01;
seed = 13;
n = 30;
m = 1000;
eigenvalue_min = 0.95;
[dat, A_true] = test_dmd_dat(n, m, noise, eigenvalue_min, seed);

% Truncation rank
r = -1;

% First get the naive model
useFB = false;
[ ~, ~, ~, romA_naive, ~, ~, ~ ] = dmd( dat, 1,...
    r, [], false, useFB);

% Now compare to the FB model
useFB = true;
[ ~, ~, ~, romA_FB, ~, ~, ~ ] = dmd( dat, 1,...
    r, [], false, useFB);

% Next do an exponential DMD model
[A_exp] = exponential_dmd(dat, 10, false);

% Now do a geometric series model
[A_geo] = exponential_dmd(dat, 150, true, 0.9);

% Now do a geometric series model with FB
% [A_geo_fb] = exponential_dmd(dat, 150, true, 0.9, true);

% Now do an exponential series model with FB
% [A_exp_fb] = exponential_dmd(dat, 20, false, [], true);

% Get data reconstructions
dat_naive = zeros(size(dat));
dat_naive(:,1) = dat(:,1);
dat_fb = dat_naive;
dat_exp = dat_naive;
dat_geo = dat_naive;
for i = 2:m
    dat_naive(:,i) = romA_naive*dat_naive(:,i-1);
    dat_fb(:,i) = romA_FB*dat_fb(:,i-1);
    dat_exp(:,i) = A_exp*dat_exp(:,i-1);
    dat_geo(:,i) = A_geo*dat_geo(:,i-1);
end

t = 1:1000;
figure;
plot(real(dat(1,t)), 'k', 'LineWidth',2)
true_ylim = ylim();
% title('First neuron and reconstructions')
hold on
plot(real(dat_naive(1,t)))
plot(real(dat_fb(1,t)))
plot(real(dat_exp(1,t)))
plot(real(dat_geo(1,t)))
ylim(true_ylim);

% plot(eig(A_exp_fb, 'vector'), '*')
% plot(eig(A_geo_fb, 'vector'), '*')
% legend({'Naive', 'FB', 'Exp', 'Exp_fb', 'Geometric', 'FB_geo', 'True'})
legend({'True', 'Naive', 'FB', 'Exp', 'Geometric'})
title(sprintf('Noise level: %.4f',noise))


%==========================================================================


%% Test out fbDMD on BACKWARDS toy data

% Produce data
noise = 0.01;
seed = 13;
n = 30;
m = 1000;
eigenvalue_min = 0.95;
[dat, A_true] = test_dmd_dat(n, m, noise, eigenvalue_min, seed);
A_true = inv(A_true);
dat = dat(:, size(dat,2):-1:1);

% Truncation rank
r = 0;

% First get the naive model
useFB = false;
[ ~, ~, ~, romA_naive, ~, ~, ~ ] = dmd( dat, 1,...
    r, [], false, useFB);

% Now compare to the FB model
useFB = true;
[ ~, ~, ~, romA_FB, ~, ~, ~ ] = dmd( dat, 1,...
    r, [], false, useFB);

% Next do an exponential DMD model
[A_exp] = exponential_dmd(dat, 10, false, 2);

% Now do a geometric series model
[A_geo] = exponential_dmd(dat, 150, true, 0.9);

% Now do a geometric series model with FB
[A_geo_fb] = exponential_dmd(dat, 150, true, 0.9, true);

% Now do an exponential series model with FB
[A_exp_fb] = exponential_dmd(dat, 10, false, 2, true);

% Plot eigenvalues
figure;
plot(eig(romA_naive, 'vector'),'o')
hold on
plot(eig(romA_FB, 'vector'), '*')
plot(eig(A_exp, 'vector'), '*')
plot(eig(A_exp_fb, 'vector'), '*')
plot(eig(A_geo, 'vector'), '*')
plot(eig(A_geo_fb, 'vector'), '*')
plot(eig(A_true, 'vector'), 'sk')
% viscircles([0,0],1, 'Color', 'k');
legend({'Naive', 'FB', 'Exp', 'Exp_fb', 'Geometric', 'FB_geo', 'True'})
title(sprintf('Noise level: %.4f',noise))


%==========================================================================


%% Use MAF to look at a "better" low-dim mode structure
filename = '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
filename_both = {...
    'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\sevenStateColoring.mat',...
    'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\TS20141221b_THK178_lite-1_punc-31_NLS3_6eggs_1mMTet_basal_1080s.mat'};
dat_struct = importdata(filename);
dat_struct2 = importdata(filename_both{2});

% Use a model as a preprocessor
settings = struct(...
    'to_subtract_mean',false,...
    'to_subtract_mean_sparse',false,...
    'to_subtract_mean_global',false,...
    'dmd_mode','func_DMDc',...
    'add_constant_signal',false,...
    'filter_window_dat',0,...
    'use_deriv',false);
settings.global_signal_mode = 'ID_binary';
my_model_smooth = CElegansModel(filename_both, settings);

% Set the data to be used for MAF
dat = my_model_smooth.dat;
% dat = dat_struct2.deltaFOverF_deriv';
% dat = my_model_smooth.dat(130:end,:);

% First use heuristic for temporal patterns, which is very fast
[ Q1 , fQ1 , info1 ] = run_maf( dat , 20, 'heuristic');
figure;plot((Q1(:,1:3)'*(dat))')
% figure;plot((Q1(:,2)'*(dat_struct.traces'))')
% figure;plot((Q1(:,4:6)'*(dat))')
% figure;plot(Q1(:,1)'*(dat_struct.traces'))
% figure;plot((Q1(:,2:4)'*(dat_struct.traces'))')
% figure;plot((Q1(:,5:6)'*(dat_struct.traces'))')
% figure;plot((Q1(:,7:8)'*(dat_struct.traces'))')
% figure;plot((Q1(:,9:10)'*(dat_struct.traces'))')
plot_colored((Q1(:,1:3)'*(dat))', dat_struct.SevenStates,...
    dat_struct.SevenStatesKey)
% plot_colored((Q1(:,[1, 2, 4])'*(dat))', dat_struct.SevenStates,...
%     dat_struct.SevenStatesKey)
% plot_colored((Q1(:,10)'*(dat))', dat_struct.SevenStates,dat_struct.SevenStatesKey)

% Use heuristic, but for the spatial patterns 
[ Q1_space , fQ1_space , info1_space ] = run_maf( dat' , 20, 'heuristic');
figure;plot((Q1_space(:,1:3)'*(dat'))')
% figure;plot((Q1_space(:,2)'*(dat'))')
% figure;plot((Q1_space(:,4:6)'*(dat'))')
% figure;plot(Q1(:,1)'*(dat_struct.traces'))
% figure;plot((Q1(:,2:4)'*(dat_struct.traces'))')
% figure;plot((Q1(:,5:6)'*(dat_struct.traces'))')
% figure;plot((Q1(:,7:8)'*(dat_struct.traces'))')
% figure;plot((Q1(:,9:10)'*(dat_struct.traces'))')
% plot_colored((Q1_space(:,1:3)'*(dat))', dat_struct.SevenStates,...
%     dat_struct.SevenStatesKey)

% Second use the iterative method, which may be much slower
% [ Q2 , fQ2 , info2 ] = run_maf( dat_struct.traces' , 4);
% figure;plot(Q2(:,1)'*(dat_struct.traces'))
% figure;plot((Q2(:,2:4)'*(dat_struct.traces'))')


%==========================================================================


%% Plot exp_dmd reconstruction error as a function of noise
% Produce data
noise = 0.00;
seed = 13;
n = 3;
m = 500;
eigenvalue_min = 0.99;
[dat, A_true] = test_dmd_dat(n, m, noise, eigenvalue_min, seed);

% Project the data into a higher space and add noise
%   i.e. obscure the original basis
n_high = 10;
proj_high = rand([n_high, n]);
dat_high = proj_high*dat;
dat_high = dat_high - mean(dat_high,2);

noise_vec = linspace(0.01, 0.1, 2);
all_errs_exp = zeros(size(noise_vec));
all_errs_clean = zeros(size(noise_vec));

for i = 1:length(noise_vec)
    dat_high_noise = dat_high + normrnd(0, noise_vec(i), size(dat_high));
    dat_high_noise = dat_high_noise - mean(dat_high_noise,2);
    % Decrease the dimensionality of the data
%     [U,S,V] = svd(dat_high_noise);
%     r = n;
%     dat_high_noise = U(:,1:r)*S(1:r,1:r)*V(:,1:r)';
    % Get model
%     [A_exp] = exponential_dmd(dat_high_noise, 12, false);
%     [A_exp] = exponential_dmd(dat_high_noise, 40, false, 10, true);
    [A_exp] = exponential_dmd(dat_high_noise, 100, true, 0.9);
    % Get reconstruction
    dat_exp = zeros(size(dat_high));
    dat_exp(:,1) = dat_high_noise(:,1);
    for j = 2:m
        dat_exp(:,j) = A_exp*dat_exp(:,j-1);
    end
    % Get error
    all_errs_exp(i) = norm(dat_exp-dat_high_noise);
    all_errs_clean(i) = norm(dat_exp-dat_high);
end

% Plot errors
%figure;
%plot(noise_vec, all_errs_exp)
%hold on
%plot(noise_vec, all_errs_clean)
%legend({'Noisy data', 'Clean data'})
%xlabel('Noise level')
%ylabel('L2 reconstruction error')

% Plot the last reconstruction
figure
subplot(3,1,1)
imagesc(real(dat_high_noise))
cb_min = min(min(real(dat_high_noise)));
cb_max = max(max(real(dat_high_noise)));
colorbar
title('Noisy data')
subplot(3,1,2)
imagesc(real(dat_exp))
colorbar
caxis([cb_min cb_max]);
title('Reconstruction')
subplot(3,1,3)
imagesc(real(dat_high))
colorbar
caxis([cb_min cb_max]);
title('Clean data')

%==========================================================================


%% Dynamic component analysis (nuclear norm minimization)
% Also: Use cvx to solve for a "dynamics preserving" map 
% Produce data
noise = 0.00;
seed = 13;
n = 3;
m = 400;
eigenvalue_min = 0.999;
[dat, A_true] = test_dmd_dat(n, m, noise, eigenvalue_min, seed);

% Project the data into a higher space and add noise
%   i.e. obscure the original basis
n_high = 8;
proj_high = rand([n_high, n]);
noise_high = 0.01;

dat_high_clean = proj_high*dat;
dat_high = dat_high_clean + normrnd(0, noise_high, size(dat_high_clean));
dat_high = dat_high - mean(dat_high,2);

X1 = dat_high(:,1:end-1);
X2 = dat_high(:,2:end);

% Do exponential DMD for comparison, with PCA 
% [A_exp] = exponential_dmd(dat_high_clean, 10, false);

[U,S,V] = svd(X1);
r = optimal_truncation(X1);
Ur = U(:,1:r);
Sr = S(1:r,1:r);
Vr = V(:,1:r);
dat_high_r = Ur*Sr*Vr';
[A_exp] = exponential_dmd(dat_high_r, 10, false);
% [A_exp] = exponential_dmd(dat_high_r, 15, false, 2);
% [A_exp] = exponential_dmd(dat_high, 100, true, 0.85);
% [A_exp] = exponential_dmd(dat_high2, 300, true, 0.98);

% Now solve for the dynamics-preserving projection
%   Minimize the norm directly instead of PCA-reduced dynamics
num_D = 20;
D_cell = cell(num_D,1);
lambda_vec = linspace(0.00001, 0.01, num_D);
for i = 1:num_D
    D_cell{i} = exponential_dmd(dat_high, 10, false, 1, false,...
        lambda_vec(i));
end
D = D_cell{end};

% cvx_begin
%    variable D(n_high, n_high)
%    minimize( norm(D*X1 - X2, 2) + lambda*norm_nuc(D) )
% cvx_end

cvx_begin
   variable D(n_high, n_high)
   minimize( norm_nuc(D) )
   subject to 
    D*X1 == X2
cvx_end

% Get data reconstructions
dat_exp_matrix = zeros(size(dat_high));
dat_exp_matrix(:,1) = dat_high(:,1);
dat_cvx = dat_exp_matrix;
% dat_thresh = dat_naive;
% dat_opt_matrix = dat_naive;
for i = 2:m
%     dat_naive(:,i) = romA_naive*dat_naive(:,i-1);
    dat_cvx(:,i) = D*dat_cvx(:,i-1);
%     dat_thresh(:,i) = A_thresh*dat_thresh(:,i-1);
    dat_exp_matrix(:,i) = A_exp*dat_exp_matrix(:,i-1);
%     dat_opt_matrix(:,i) = A_opt*dat_opt_matrix(:,i-1);
end

% Second, simplified figure
figure;
subplot(4,1,1)
imagesc(real(dat_high))
title('Noisy data')
subplot(4,1,2)
imagesc(real(dat_high_clean))
title('Clean data')
subplot(4,1,3)
imagesc(real(dat_exp_matrix))
title('Reconstruction: Exponential DMD (matrix multiplication)')
subplot(4,1,4)
imagesc(real(dat_cvx))
title('Reconstruction: Nuclear norm minimization (matrix multiplication)')


% figure;
% subplot(6,1,1)
% imagesc(real(dat_high))
% title('Noisy data')
% subplot(6,1,2)
% imagesc(real(dat_high_clean))
% title('Clean data')
% subplot(6,1,3)
% imagesc(real(dat_exp))
% title('Reconstruction: Exponential DMD')
% subplot(6,1,4)
% imagesc(real(dat_exp_matrix))
% title('Reconstruction: Exponential DMD (matrix multiplication)')
% subplot(6,1,5)
% imagesc(real(dat_opt))
% title('Reconstruction: Optimized DMD')
% subplot(6,1,6)
% imagesc(real(dat_opt_matrix))
% title('Reconstruction: Optimized DMD (matrix multiplication)')

%==========================================================================


%% Reprise: 1-step errors (toy data)

use_oscillation = false;
num_neurons = 5;
t_steps = 2500;
sz = [num_neurons, t_steps];
kp = [0.5, 0.0, 0.0, 0.0, 0.0]';
ki = [0];
kd = [];
% set_points = [0 0.5, 1.3];
set_points = [0 2.0, 1.3]; % Intermediate "begin escape" state
num_states = length(set_points);
if num_neurons > 1
%     if use_oscillation
        set_points = [set_points; ...
            NaN*ones([num_neurons-size(set_points,1), num_states])];
%     else
%         set_points = [set_points; ...
%             rand([num_neurons-size(set_points,1), num_states])];
%     end
end
% Some dependence on the initial condition
initial_condition_dependence = zeros(size(set_points));

noise = 0.1;%1e-8;

transition_mat = [[0.99 0.00 0.01];
    [0.01 0.90 0.00];
    [0.00 0.10 0.99]];
perturbation_mat = zeros(sz);
% Set up the intrinsic dynamics
A = eye(num_neurons);
% Finally: produce the toy data
[dat, ctr_signal, state_vec] = ...
    test_dat_pid(sz, kp, ki, kd, set_points, transition_mat,...
    perturbation_mat, A, [], 1, initial_condition_dependence, noise);

X1 = dat(:,1:end-1);
X2 = dat(:,2:end);
this_dat = X2 - (X2/X1)*X1; %Naive DMD residual
plotSVD(this_dat');%, struct('PCA3d',true, 'PCA_opt', 'o'));
[~, ~, ~, proj3d] = plotSVD(this_dat);

plot_colored(proj3d, state_vec(1:end-1), {'1', '2', '3'});

%==========================================================================


