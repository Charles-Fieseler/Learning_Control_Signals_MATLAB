classdef SimulationPlottingObjectTest_Kato < matlab.unittest.TestCase
    % CElegansModelTest tests inputs and basic processing properties for
    % Kato-type structs
    
    properties
        filename
        settings
        model
    end
    
    methods(TestClassSetup)
        function setFilename(testCase)
            folder_name = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\wbdataKato2015\wbdata\';
            testCase.filename = ...
                {[folder_name 'sevenStateColoring.mat'],...
                [folder_name 'TS20141221b_THK178_lite-1_punc-31_NLS3_6eggs_1mMTet_basal_1080s.mat']};
            % Calculate the model
            testCase.settings = struct(...
                'to_subtract_mean',false,...
                'to_subtract_mean_sparse',false,...
                'to_subtract_mean_global',false,...
                'add_constant_signal',false,...
                'lambda_sparse',0);
            testCase.settings.global_signal_mode = 'ID_binary';
            testCase.model = ...
                CElegansModel(testCase.filename, testCase.settings);
        end
    end
    
    methods (Test)
        function testMetadata(testCase)
            % Control signal metadata
            mdat = testCase.model.control_signals_metadata;
            
            testCase.verifyEqual(...
                mdat{'ID_binary',:}{:}, 1:8);
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
                testCase.model.original_sz, [129,3021]));
            
            testCase.verifyTrue(isequal(...
                testCase.model.total_sz, [137,3021]));
            
            testCase.verifyTrue(isequal(...
                size(testCase.model.control_signal), [8,3021]));
        end
    end
    
end 