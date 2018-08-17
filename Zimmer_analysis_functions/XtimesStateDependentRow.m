classdef XtimesStateDependentRow < AbstractDependentRow
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
    end
    
    methods
        
        function self = XtimesStateDependentRow(settings)
            
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
        end
        
        function set_defaults(self)
            defaults = struct();
            
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
            
            self.signal_name = 'x_times_state';
            
        end
        
        function setup(~, ~)
            % No setup required
        end
        
        function ctr_signal = calc_next_step(~, x, u, metadata)
            % Inputs entire control signal but returns only this controller
            this_u = u(metadata{self.signal_name,:}{:});
            % Simple multiplication of the data by the binary control
            % signal
            sz_x = length(x);
            ctr_signal = zeros(length(this_u)*sz_x,1);
            for i = 1:length(this_u)
                ind = ((i-1)*sz_x+1):(i*sz_x);
                ctr_signal(ind) = this_u(i)*x;
            end
        end
    end
end

