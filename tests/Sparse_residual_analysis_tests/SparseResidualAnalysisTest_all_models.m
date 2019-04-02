classdef SparseResidualAnalysisTest_all_models < matlab.unittest.TestCase
    % CElegansModelTest tests inputs and basic processing properties for
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
            testCase.num_iter = 3;
            s = struct('num_iter', testCase.num_iter, 'r_ctr', testCase.rank);
            
            % Get the control signals and acf
            for i = 1:testCase.n
                dat_struct = importdata(testCase.all_filenames{i});
                dat_struct.traces = f_smooth(dat_struct.traces);
                
                % First get a baseline model as a preprocessor
                testCase.all_models{i} = CElegansModel(dat_struct, ...
                    testCase.settings);
                
                [U, acf] = sparse_residual_analysis_max_acf(...
                    testCase.all_models{i}, s.num_iter, s.r_ctr);
                testCase.all_U{i} = U{1};
                testCase.all_acf{i} = acf{1};
            end
        end
    end
    
    methods (Test)
        function test_all_U(testCase)
            % The learned control signals
            
            testCase.verifyEqual(...
                length(testCase.all_U), testCase.n);
                
            for i = 1:testCase.n
                U = testCase.all_U{i};
                m = testCase.all_models{i};

                % Only one rank tested here, not multiple
                testCase.verifyEqual(...
                    size(U,1), testCase.rank);
                testCase.verifyEqual(...
                    size(U,2), m.dat_sz(2)-1);
            end
        end
        
        function test_all_acf(testCase)
            % Properties of the autocorrelation
            testCase.verifyEqual(...
                length(testCase.all_acf), testCase.n);
            
            for i = 1:testCase.n
                a = testCase.all_acf{i};
                
                testCase.verifyEqual(...
                    size(a, 1), testCase.rank);
                testCase.verifyEqual(...
                    size(a, 2), testCase.num_iter);

                testCase.verifyTrue(...
                    isempty(find(a<-1, 1)));
                testCase.verifyTrue(...
                    isempty(find(a>1, 1)));
            end
        end
        
        function test_interpretable_signals(testCase)
            % Sets up an additional table of correspondance between expert
            % and learned signals, which should allow for 
        end
    end
    
end