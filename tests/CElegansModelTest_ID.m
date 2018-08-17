classdef CElegansModelTest_ID < matlab.unittest.TestCase
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
            testCase.settings.global_signal_mode = 'ID';
            testCase.model = ...
                CElegansModel(testCase.filename, testCase.settings);
        end
    end
    
    methods (Test)
        function testMetadata(testCase)
            % Control signal metadata
            mdat = testCase.model.control_signals_metadata;
            
            testCase.verifyEqual(...
                mdat{'ID',:}{:}, 130);
            testCase.verifyEqual(...
                mdat{'sparse',:}{:}, 1:129);
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
                testCase.model.original_sz, [129,3021]));
            
            testCase.verifyTrue(isequal(...
                testCase.model.total_sz, [260,3021]));
            
            testCase.verifyTrue(isequal(...
                size(testCase.model.control_signal), [131,3021]));
        end
    end
    
end 