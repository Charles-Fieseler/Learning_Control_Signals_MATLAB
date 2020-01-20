classdef SignalLearningObjectTest_Stimulus < matlab.unittest.TestCase
    % CElegansModelTest tests inputs and basic processing properties for
    % Kato-type structs
    %   Note: the location of the data is hardcoded
    
    properties
        filename
        settings
        model
    end
    
    methods(TestClassSetup)
        function setFilename(testCase)
            folder_name = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\npr1_1_PreLet\AN20140730a_ZIM575_PreLet_6m_O2_21_s_1TF_47um_1330_\';
            testCase.filename = [folder_name 'wbdataset.mat'];
            % Calculate the model
            testCase.settings = struct(...
                'to_subtract_mean',false,...
                'to_subtract_mean_sparse',false,...
                'to_subtract_mean_global',false,...
                'add_constant_signal',false,...
                'lambda_sparse',0);
            testCase.settings.global_signal_mode = 'ID_binary';
            testCase.model = ...
                SignalLearningObject(testCase.filename, testCase.settings);
            % Align control signal naming
            testCase.model.set_simple_labels();
            testCase.model.remove_all_control();
            testCase.model.calc_all_control_signals();
        end
    end
    
    methods (Test)
        function testMetadata(testCase)
            % Control signal metadata
            mdat = testCase.model.control_signals_metadata;
            
            testCase.verifyEqual(...
                mdat{'ID_binary',:}{:}, 1:5);
            testCase.verifyEqual(...
                mdat{'Stimulus_O2',:}{:}, 6:7);
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
                testCase.model.original_sz, [114,4055]));
            
            testCase.verifyTrue(isequal(...
                testCase.model.total_sz, [121,4055]));
            
            testCase.verifyTrue(isequal(...
                size(testCase.model.control_signal), [7,4055]));
        end
    end
    
end 