classdef SumXtimesStateDependentRow < AbstractDependentRow
    % A dependent row for use in CElegansModel
    % Main method is calc_next_step:
    %   Input:
    %       x - Current state
    %       u - ctr_signal; only binary label is used
    %
    %   Output: 
    %       ctr_signal - large column of data multiplied by all binary
    %       state encodings (len = len(u)*len(x))
    
    properties (SetAccess={?SettingsImportableFromStruct})
    end
    
    properties
        XtimesState_obj
    end
    
    methods
        
        function self = SumXtimesStateDependentRow(settings)
            
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
            
            self.XtimesState_obj = XtimesStateDependentRow();
            
            self.signal_name = 'cumsum_x_times_state';
        end
        
        function set_defaults(self)
            defaults = struct();
            
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
            
        end
        
        function setup(~, ~)
            % No setup required
        end
        
        function ctr_signal = calc_next_step(self, x, u, metadata)
            % Inputs entire control signal but returns only this controller
            this_u = u(metadata{self.signal_name,:}{:});
            % Simple multiplication of the data by the binary control
            % signal (uses external object)
            ctr_signal = calc_next_step(self.XtimesState_obj, x, u, metadata);
            
            % Do a custom cumulative sum:
            %   A state change means no cumulative sum is necessary
            nonzero_ctr = abs(ctr_signal)>0;
            nonzero_u = abs(this_u)>0;
            if nonzero_ctr == nonzero_u
                ctr_signal = ctr_signal + this_u;
            end
        end
    end
end

