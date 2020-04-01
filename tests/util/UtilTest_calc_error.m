classdef UtilTest_calc_error < matlab.unittest.TestCase
    % CElegansModelTest tests inputs and basic processing properties for
    % Kato-type structs
    
    properties
        % Model settings
        all_err_nstep_new
        all_err_nstep_old
        num_err_steps = 1:5
        
        cross_val_new
        cross_val_old
    end
    
    methods(TestClassSetup)
        function calcErrors(testCase)
            %% Generate test data
            % Build example dynamics matrix
            seed = 17;
            n = 2;                  % Matrix size; should be even
            m = 1000;               % Number of data points
            eigenvalue_min = 0.95;  % Minimum eigenvalue; 1.0 = stable
            [X_dmd, A_true] = test_dmd_dat(n, m, 0, eigenvalue_min, seed);

            % Build random controller
            ctr_timing = 100:105;
            U = zeros(1,m-1);
            U(ctr_timing) = 1.0;

            % Generate controlled data
            x0 = X_dmd(:,1);
            B = ones(n,1);
            X_true = real(calc_reconstruction_dmd(x0, [], A_true, B, U));
            % Add noise
            noise = 0.05;
            X = X_true + noise*randn(size(X_true));
            
            
            % Calculate model
            X1 = X(:, 1:end-1);
            X2 = X(:, 2:end);
            n = size(X, 1);
            AB = X2/[X1; U];
            A = AB(:, 1:n);
            B = AB(:, (n+1):end);
            
            %% Calculate errors
            % Use new error function
            testCase.all_err_nstep_new = calc_nstep_error(...
                X, A, B, U, testCase.num_err_steps, false);
            % Do old error function
            testCase.all_err_nstep_old = ...
                zeros(size(testCase.all_err_nstep_new));
            final_step = testCase.num_err_steps(end);
            for i = 1:length(testCase.num_err_steps)
                this_step = testCase.num_err_steps(i);
                % The new function uses the same amount of data for all
                % error calculations
                this_subset = m - final_step + i;
                this_X = X(:,1:this_subset);
                testCase.all_err_nstep_old(i) = calc_nstep_error_old(...
                    this_X, A, B, U(:,1:this_subset-1), this_step, false);
            end
            
            %% Cross validation functions
%             k = 5;
%             % Old
%             [~, testCase.cross_val_old] = dmdc_cross_val_old(X, U, k, ...
%                 testCase.num_err_steps, [], false);
%             % New
%             [~, testCase.cross_val_new] = dmdc_cross_val(X, U, k, ...
%                 testCase.num_err_steps, [], false);
        end
    end
    
    methods (Test)
        function test_nstep(testCase)
            % Test that the new function is calculating all error steps
            % correctly
            testCase.verifyEqual(...
                testCase.all_err_nstep_old, testCase.all_err_nstep_new);
        end
        
        function test_crossval(testCase)
            % Test that the new function is calculating all error steps
            % correctly
%             testCase.verifyEqual(...
%                 testCase.cross_val_old, testCase.cross_val_new);
        end
        
    end
    
end