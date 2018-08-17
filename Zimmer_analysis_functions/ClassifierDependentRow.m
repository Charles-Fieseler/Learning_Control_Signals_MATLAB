classdef ClassifierDependentRow < AbstractDependentRow
    % A dependent row for use in CElegansModel
    % Main method is calc_next_step:
    %   Input:
    %       x - Current state
    %
    %   Output: 
    %       ctr_signal - a classification of the current system state
    
    properties (SetAccess={?SettingsImportableFromStruct})
        to_optimize_hyperparameters
    end
    
    properties
        cecoc_model
    end
    
    methods
        
        function self = ClassifierDependentRow(settings)
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
            
            self.signal_name = 'ID';
        end
        
        function set_defaults(self)
            defaults = struct(...
                'to_optimize_hyperparameters', false...
                );
            
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
            
        end
        
        function setup(self, varargin)
            % Trains the cecoc model... takes a long time!!!
            training_dat = varargin{1};
            training_labels = varargin{2};
            rng(1);
            if optimize_hyperparameters
                disp('Optimizing hyperparameters, may take >1 hour')
                self.cecoc_model = fitcecoc(...
                    training_dat',...
                    training_labels,...
                    'OptimizeHyperparameters','auto',...
                    'HyperparameterOptimizationOptions',...
                        struct('AcquisitionFunctionName',...
                        'expected-improvement-plus'));
            else
                self.cecoc_model = fitcecoc(...
                    training_dat',...
                    training_labels);
            end
        end
        
        function ctr_signal = calc_next_step(self, x, ~, ~)
            % Uses the trained classifier to predict the category of the
            % input brain state
            % Note: input should be one or more column vectors
            %
            % For this function, the external control signal 'u' isn't used
            
            % The model might have been trained with derivatives
            if size(self.cecoc_model.X,2) == 2*length(x)
                x = [x, gradient(x)];
            end
            
            % Only change the single entry that is the ID (any later
            % entries are constant)
            ctr_signal = predict(self.cecoc_model, x);
        end
    end
end

