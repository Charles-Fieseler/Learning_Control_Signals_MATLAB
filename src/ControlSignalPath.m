classdef ControlSignalPath < matlab.mixin.Copyable
    % Stores output of a learned control signal path, produced by the
    % function learn_control_signals.m
    %
    % This object can be recalculated using:
    %  >> newSelf = learn_control_signals(self.data, self.settings);
    %
    % Also contains the results of metric evaluations used to determine the
    % "best" or "true" control signals. These can be calculated using the
    % method self.calc_best_control_signal. For more information on these
    % helper functions:
    %   
    % See also: ControlSignalPath.calc_best_control_signal
    
    properties
        % Settings from original object
        data
        learn_control_signals_settings
        % Dynamics
        all_A
        all_B
        % Control signals
        all_U
        
        % Metric evaluations
        objective_function
        objective_values
        
        % Best values and model
        best_index
    end
    
    properties (Dependent)
        A
        B
        U
    end
    
    methods % User-facing
        function self = ControlSignalPath(...
                data, settings, all_A, all_B, all_U)
            % Imports data produced by learn_control_signals.m
            self.data = data;
            self.learn_control_signals_settings = settings;
            
            self.all_A = all_A;
            self.all_B = all_B;
            self.all_U = all_U;
        end
        
        function calc_best_control_signal(self, objective_function)
            % Calculates the best control signal, according to function
            % 'objective_function' which is saved in this object
            %
            % Input:
            %   objective_function - A string with a valid method name
            %       that calculates an objective function using the control
            %       signals and possibly the dynamics matrices and original
            %       data. Currently implemented metrics are:
            %
            %       'acf' - simple autocorrelation of the signal
            %       'aic' - Akaike Information Criteria (AIC)
            %
            % TODO: allow custom metrics
            % TODO: allow custom options
            assert(ischar(objective_function),...
                'Should pass string name of function')
            assert(ismethod(self, objective_function),...
                'Custom objective function not implemented (yet)')
            
            self.objective_function = objective_function;
            
            vals = zeros(size(self.all_U));
            for i = 1:length(self.all_U)
                vals(i) = self.(objective_function)(i);
            end
            vals(end) = vals(end-1);
            self.objective_values = vals;
            
            % Get maximum of above
            [~, i] = max(vals);
            self.best_index = i;
        end
        
    end
    
    methods % Cross-validation specific functions
        
        function all_err = calc_cross_validation_error(self,...
                k, num_error_steps)
            % Calculates the cross validation errors for specific
            % hyperparameters. Does not calculate the minumum
            if length(num_error_steps) > 1
                error('Use calc_cross_validation_error_vec() instead')
            end
            max_iter = length(self.all_U);
            X = self.data;
            
            all_err = zeros(max_iter, k-1);
            for i = 1:max_iter
                this_U = self.all_U{i};
                [~, all_err(i,:)] = ...
                    dmdc_cross_val(X, this_U, k, num_error_steps, [], false);
            end
        end
        
        function all_err = calc_cross_validation_error_vec(self,...
                k, num_error_steps)
            % Calculates the cross validation errors for specific
            % hyperparameters. Does not calculate the minumum
            assert(length(num_error_steps) > 1,...
                'Use calc_cross_validation_error() instead')
            % Evaluates multiple time steps at once
            
            max_iter = length(self.all_U);
            X = self.data;
            
            all_err = zeros(max_iter, length(num_error_steps), k-1);
            for i = 1:max_iter
                this_U = self.all_U{i};
                [~, all_err(i, :, :)] = ...
                    dmdc_cross_val(X, this_U, k, num_error_steps, [], false);
            end
        end
    end
    
    methods % Metrics for control signal quality
        function val = acf(self, i)
            % Simplest objective function: autocorrelation
            val = acf(self.all_U{i}', 1, false);
        end
        
        function val = aic(self, i)
            % Akaike Information Criteria (AIC)
            %   Uses the 2-step error by default
            val = -aic_2step_dmdc(self.data, self.all_U{i}, [], [], 2, ...
                [], 'standard');
        end
    end
    
    
    methods % For dependent variables
        function out = get.A(self)
            out = self.all_A{self.best_index};
        end
        
        function out = get.B(self)
            out = self.all_B{self.best_index};
        end
        
        function out = get.U(self)
            out = self.all_U{self.best_index};
        end
    end
end

