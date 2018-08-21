classdef CElegansModelTest_generated < matlab.unittest.TestCase
    % CElegansModelTest tests inputs and basic processing properties
    
    properties
        dat
        settings
        model
    end
    
    methods(TestClassSetup)
        function setFilename(testCase)
            %% Settings
            sz = [2, 3000];
            kp = [0.5; 0.25];
            ki = [0.01; 0.01];
            kd = [];
            set_points = [[0.5 1];
                [1 -1]];
            transition_mat = [[0.99 0.01];
                [0.01 0.99]];
            perturbation_mat = zeros(sz);
            perturbation_mat(1,100) = 0.1;
            perturbation_mat(1,300) = -0.1;
            perturbation_mat(1,500) = 0.2;
            perturbation_mat(1,700) = -0.2;
            perturbation_mat(1,1000:1050) = 0.1;
            perturbation_mat(1,2000:2050) = -0.5;
            
            %% Get data
            [testCase.dat, ctr_signal, state_vec] = ...
                test_dat_pid(sz, kp, ki, kd, set_points, transition_mat,...
                perturbation_mat);
            
            % Calculate the models
            ID = {{'1','2'}};
            grad = gradient([testCase.dat; ctr_signal]');
            dat_struct = struct(...
                ...'traces', {[dat; ctr_signal]'},...
                'traces', testCase.dat',...
                'tracesDif',grad(1:end-1,:),...
                'ID',ID,...
                'ID2',ID,...
                'ID3',ID,...
                'TwoStates', state_vec,...
                'TwoStatesKey',{{'State 1','State 2'}});
            
            use_deriv = false;
            augment_data = 0;
            ctr_signal = ctr_signal(:,1:end-augment_data);
            testCase.settings = struct(...
                'to_subtract_mean',false,...
                'to_subtract_mean_sparse',false,...
                'to_subtract_mean_global',false,...
                ...'dmd_mode','func_DMDc',...
                ...'dmd_mode','sparse',...
                'augment_data',augment_data,...
                'use_deriv',use_deriv,...
                ...'AdaptiveDmdc_settings',struct('x_indices',x_ind),...
                ...'custom_global_signal',ctr_signal,...
                'custom_control_signal',ctr_signal(3:end,:),... % Not using the integral error, only perturbations
                ...'lambda_sparse',0.022); % This gets some of the perturbations, and some of the transitions...
                'lambda_sparse',0); % Don't want a sparse signal here
            
            %% Now add an additional integral control signal for generating data
            % settings.global_signal_mode = 'ID_binary_and_cumsum_x_times_state';
            testCase.settings.global_signal_mode = ...
                'ID_binary_and_cumsum_x_times_state_and_length_count';
            
            % Define the table for the dependent row objects
            signal_functions = {SumXtimesStateDependentRow()};
            setup_arguments = {'normalize_cumsum_x_times_state'};
            signal_indices = {'cumsum_x_times_state'};
            dependent_signals = table(signal_functions, signal_indices, setup_arguments);
            testCase.settings.dependent_signals = dependent_signals;

            %% Actually calculate the model
            testCase.model = CElegansModel(dat_struct, testCase.settings);
        end
    end
    
    methods (Test)
        function testMetadata(testCase)
            % Control signal metadata
            mdat = testCase.model.control_signals_metadata;
            
            testCase.verifyEqual(...
                mdat{'ID_binary',:}{:}, 1:2);
            testCase.verifyEqual(...
                mdat{'constant',:}{:}, 3);
            testCase.verifyEqual(...
                mdat{'cumsum_x_times_state',:}{:}, 4:7);
            testCase.verifyEqual(...
                mdat{'length_count',:}{:}, 8:9);
            testCase.verifyEqual(...
                mdat{'user_custom_control_signal',:}{:}, 10:11);
        end
        
        function testControlSignalsBinary(testCase)
            % Control signal metadata
            ctr = testCase.model.control_signal(1:2,:);
            
            testCase.verifyEqual(unique(ctr), [0; 1]);
            testCase.verifyEqual(...
                find(abs(diff(ctr(1,:)))>0)',...
                [110; 360; 367; 520; 870; 907; 961; 1123; 1208; 1210;
                1218; 1371; 1406; 1737; 1898; 2205; 2646; 2723; 2839; 2988]);
        end
        
        function testControlSignalsConstant(testCase)
            % Control signal metadata
            ctr = testCase.model.control_signal(3,:);
            
            testCase.verifyEqual(ctr, ones(size(ctr)));
        end
        
        function testSparse(testCase)
            % Properties of the sparse signal
            testCase.verifyEqual(...
                testCase.model.S_sparse_nnz, 0);
            
            testCase.verifyEqual(...
                testCase.model.L_sparse_rank, 0);
        end
        
        function testGlobal(testCase)
            % Properties of the NOT SET global properties
            testCase.verifyEqual(...
                testCase.model.L_global, []);
            testCase.verifyEqual(...
                testCase.model.L_global_rank, []);
        end
        
        function testDat(testCase)
            % Properties of the data
            testCase.verifyTrue(isequal(...
                testCase.model.original_sz, [2,3000]));
            
            testCase.verifyTrue(isequal(...
                testCase.model.total_sz, [13,3000]));
            
            testCase.verifyTrue(isequal(...
                size(testCase.model.control_signal), [11,3000]));
        end
        
        function testDependentSignals(testCase)
            % Control signal metadata and dependent signal metadata
            mdat = testCase.model.control_signals_metadata;
            dsig = testCase.model.dependent_signals;
            
            testCase.verifyEqual(...
                mdat{'cumsum_x_times_state',:}{1},...
                dsig.signal_indices{:});
            
        end
    end
    
end