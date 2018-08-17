classdef CElegansModelTest_ID < matlab.unittest.TestCase
    % CElegansModelTest tests inputs and basic processing properties
    
    properties
        filename
        settings
        model
    end
    
    methods(TestMethodSetup)
        function setFilename(testCase)
            testCase.filename = ...
                '../../Zimmer_data/WildType_adult/simplewt5/wbdataset.mat';
            % Calculate the model
            settings = struct(...
                'to_subtract_mean',false,...
                'to_subtract_mean_sparse',false,...
                'to_subtract_mean_global',false);
            settings.global_signal_mode = 'ID';
            obj = CElegansModel(testCase.filename, settings);
        end
    end
    
    methods (Test)
        function testMetadata(testCase)
            testCase.verifyEqual(...
                obj.control_signals_metadata.signal_indices, 130);
        end
    end
    
end 