classdef AbstractDependentRow < SettingsImportableFromStruct
    %Abstract class for Dependent data rows for DMDc
    %   Works with CElegansModel to produce new data rows as data is
    %   generated
    % Replaces a row of certain indices in the training data
    
    properties
        signal_name % For interacting with control signal metadata
    end
    
    methods
        function setup(~)
            error('Must implement a setup() method with arguments saved in field setup_arguments')
        end
        
        function calc_next_step(~)
            error('Must implement calc_next_step method with one output')
        end
    end
end

