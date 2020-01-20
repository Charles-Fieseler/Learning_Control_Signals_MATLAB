classdef SparseResidualAnalysisTest_one_model < matlab.unittest.TestCase
    % SignalLearningObjectTest tests inputs and basic processing properties for
    % Kato-type structs
    
    properties
        % Model settings
        filename
        settings
        model
        
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
            tmp = get_Zimmer_filenames();
            testCase.filename = tmp{1};
            % Calculate the model
            testCase.settings = define_ideal_settings();
            testCase.model = ...
                SignalLearningObject(testCase.filename, testCase.settings);
            % Run the sparse residual analysis function
            testCase.num_iter = 10;
            testCase.rank = 5;
            [testCase.all_U, testCase.all_acf, testCase.all_nnz] = ...
                sparse_residual_analysis_max_acf(...
                testCase.model, testCase.num_iter, testCase.rank);
        end
    end
    
    methods (Test)
        function test_all_U(testCase)
            % The learned control signals
            U = testCase.all_U;
            U1 = U{1};
            
            testCase.verifyEqual(...
                length(U), length(testCase.rank));
            
            % Only one rank tested here, not multiple
            testCase.verifyEqual(...
                size(U1,1), testCase.rank);
        end
        
        function test_all_acf(testCase)
            % Properties of the autocorrelation
            a = testCase.all_acf;
            
            testCase.verifyEqual(...
                size(a{1}, 1), testCase.rank);
            testCase.verifyEqual(...
                size(a{1}, 2), testCase.num_iter);
            
            testCase.verifyTrue(...
                isempty(find(a{1}<-1, 1)));
            testCase.verifyTrue(...
                isempty(find(a{1}>1, 1)));
        end
        
        function test_all_nnz(testCase)
            % Properties of the number of nonzeros in the control signals
            n = testCase.all_nnz{1};
            
            % Bounds
            testCase.verifyTrue(isempty(find(n<0, 1)))
            testCase.verifyTrue(...
                isempty(find(n>numel(testCase.all_U{1}),1)));
            % Decreasing, though it might stall
            testCase.verifyTrue(...
                isempty(find(diff(n)>0,1)));
        end
    end
    
end 