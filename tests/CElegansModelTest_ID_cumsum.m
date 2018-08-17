classdef CElegansModelTest_ID_cumsum < matlab.unittest.TestCase
    % CElegansModelTest tests inputs and basic processing properties
    
    properties
        filename
        settings
        model
    end
    
    methods(TestClassSetup)
        function setFilename(testCase)
            testCase.filename = ...
                '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
            % Calculate the model
            testCase.settings = struct(...
                'to_subtract_mean',false,...
                'to_subtract_mean_sparse',false,...
                'to_subtract_mean_global',false);
            testCase.settings.global_signal_mode = ...
                'ID_binary_and_cumsum_x_times_state';
            testCase.model = ...
                CElegansModel(testCase.filename, testCase.settings);
        end
    end
    
    methods (Test)
        function testMetadata(testCase)
            % Control signal metadata
            mdat = testCase.model.control_signals_metadata;
            
            testCase.verifyEqual(...
                mdat{'sparse',:}{:}, 1:129);
            testCase.verifyEqual(...
                mdat{'ID_binary',:}{:}, 130:137);
            testCase.verifyEqual(...
                mdat{'cumsum_x_times_state',:}{:}, 138:1169);
        end
        
        function testSparse(testCase)
            % Properties of the sparse signal
            testCase.verifyEqual(...
                testCase.model.S_sparse_nnz, 14485);
            
            testCase.verifyEqual(...
                testCase.model.L_sparse_rank, 129);
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
                testCase.model.original_sz, [129, 3021]));
            
            testCase.verifyTrue(isequal(...
                testCase.model.total_sz, [1298, 3021]));
            
            testCase.verifyTrue(isequal(...
                size(testCase.model.control_signal), [1169, 3021]));
        end
    end
    
end 