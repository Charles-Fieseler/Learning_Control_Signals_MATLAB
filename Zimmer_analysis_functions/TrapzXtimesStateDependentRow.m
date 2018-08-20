classdef TrapzXtimesStateDependentRow < AbstractDependentRow
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
        normalization_factor
        
        previous_dat
    end
    
    methods
        
        function self = TrapzXtimesStateDependentRow(settings)
            
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
            
            self.XtimesState_obj = XtimesStateDependentRow();
            
            self.signal_name = 'cumtrapz_x_times_state';
            self.XtimesState_obj.signal_name = 'cumtrapz_x_times_state';
            self.previous_dat = [];
        end
        
        function set_defaults(self)
            defaults = struct();
            
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
            
        end
        
        function setup(self, varargin)
            self.normalization_factor = varargin{1}{1};
        end
        
        function ctr_signal = calc_next_step(self, x, u, metadata)
            % Inputs entire control signal but returns only this controller
            this_u = u(metadata{self.signal_name,:}{:});
            % Simple multiplication of the data by the binary control
            % signal (uses external object)
            binary_dat = calc_next_step(self.XtimesState_obj, x, u, metadata);
            if isempty(self.previous_dat)
                self.previous_dat = zeros(size(binary_dat));
            end
            
            % Do a custom cumulative sum:
            %   A state change means no cumulative integration is necessary
            nonzero_ctr = abs(binary_dat)>0;
            nonzero_u = abs(this_u)>0;
            binary_dat = binary_dat * self.normalization_factor;
            if nonzero_ctr == nonzero_u
                ctr_signal = this_u + ...
                    trapz([self.previous_dat, binary_dat], 2);
            else
                ctr_signal = zeros(size(binary_dat));
            end
            self.previous_dat = binary_dat;
        end
    end
end

