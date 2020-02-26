classdef SparseResidualAnalysisTest_interpretability < matlab.unittest.TestCase
    % SimulationPlottingObject tests inputs and basic processing properties for
    % Kato-type structs
    
    properties
        % Model settings
        all_filenames
        settings
        all_models
        n
        
        % Residual analysis settings
        num_iter
        rank
        
        % Residual analysis output
        all_U
        all_acf
        all_nnz
        
        correlation_table
    end
    
    methods(TestClassSetup)
        function setFilename(testCase)
            testCase.all_filenames = get_Zimmer_filenames();
            testCase.n = length(testCase.all_filenames);
            testCase.all_models = cell(testCase.n, 1);
            testCase.all_U = cell(testCase.n, 1);
            testCase.all_acf = cell(testCase.n, 1);
            % For the preprocessor model
            testCase.settings = define_ideal_settings();
            testCase.settings.augment_data = 0;
            testCase.settings.dmd_mode = 'no_dynamics'; % But we want the controllers
            f_smooth = @(x) smoothdata(x, 'gaussian', 3);
            % For the residual analysis
            testCase.rank = 10;
            testCase.num_iter = 50;
            s = struct('num_iter', testCase.num_iter, 'r_ctr', testCase.rank);
            
            % Get the control signals and acf
            for i = 1:testCase.n
                dat_struct = importdata(testCase.all_filenames{i});
                dat_struct.traces = f_smooth(dat_struct.traces);
                
                % First get a baseline model as a preprocessor
                testCase.all_models{i} = SimulationPlottingObject(dat_struct, ...
                    testCase.settings);
                
                [U, acf] = sparse_residual_analysis_max_acf(...
                    testCase.all_models{i}, s.num_iter, s.r_ctr);
                testCase.all_U{i} = U{1};
                testCase.all_acf{i} = acf{1};
            end
            
            % Set up interpretation table
            testCase.correlation_table = ...
                connect_learned_and_expert_signals(...
                testCase.all_U, testCase.all_models);
        end
    end
    
    methods (Test)
        
        function test_all_models(testCase)
            % Each model should be represented at least once
            t = testCase.correlation_table;
            testCase.assertEqual(...
                length(unique(t{:, 'model_index'})), testCase.n);
        end
        
        function test_reversals(testCase)
            % EACH model should have a good reversal signal
            %   Update: 2 of them don't! But they have either FWD or REV
            t = testCase.correlation_table;
            
            required_set = {'Reversal', 'REV1', 'REV2', 'REVSUS',...
                'Forward', 'FWD', 'SLOW'};
            
            for i = 1:testCase.n
                ind = t{:,'model_index'}==i;
                this_t = t{ind, 'experimental_signal_name'};
                testCase.assertTrue(...
                    any(contains(this_t, required_set)));
            end
        end
    end
    
end