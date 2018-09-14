classdef AdaptiveSegmentation < ...
        AbstractWindowDmd & SettingsImportableFromStruct
    % Adaptively segments a time series using control signals
    % Two major assumptions:
    %   - The intrinsic dynamics (A) are the same throughout
    %   - The control signal is one dimensional at any given time, i.e. is
    %   dominated by a state value 
    %
    %
    % INPUTS
    %   INPUT1 -
    %   INPUT2 -
    %
    % OUTPUTS -
    %   OUTPUT1 -
    %   OUTPUT2 -
    %
    % EXAMPLES
    %
    %   EXAMPLE1
    %
    %
    %   EXAMPLE2
    %
    %
    %
    % Dependencies
    %   .m files, .mat files, and MATLAB products required:
    %
    %   See also: OTHER_FUNCTION_NAME
    %
    %
    %
    % Author: Charles Fieseler
    % University of Washington, Dept. of Physics
    % Email address: charles.fieseler@gmail.com
    % Website: coming soon
    % Created: 25-Aug-2018
    %========================================

    
    properties (SetAccess={?SettingsImportableFromStruct}, Hidden=true)
        verbose
        
        force_redo_every_window
        
        external_u
    end
    
    properties
        dat
        all_error_reductions
        
        all_B0
        segment_indices
    end
    
    properties (Access=private, Transient=true)
        raw
    end
    
    methods
        function self = AdaptiveSegmentation(...
                file_or_dat, settings, window_settings)
            %AdaptiveSegmentation Construct an instance of this class
            %   Takes in a data series and attempts to segment it
            
            %% Set defaults and import settings
            if ~exist('window_settings', 'var')
                window_settings = struct('window_step',1);
            end
            self@AbstractWindowDmd(window_settings);
            
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
            %==========================================================================
            
            %% Import data and preprocess
            if ischar(file_or_dat)
                %     self.filename = file_or_dat;
                tmp_dat = importdata(file_or_dat);
                if isstruct(tmp_dat)
                    self.import_from_struct(tmp_dat);
                elseif isnumeric(tmp_dat)
                    self.raw = tmp_dat;
                else
                    error('Filename must contain a struct or matrix')
                end
            elseif isnumeric(file_or_dat)
                self.raw = file_or_dat;
            elseif isstruct(file_or_dat)
                self.import_from_struct(file_or_dat);
            else
                error('Must pass data matrix or filename')
            end
            
            self.preprocess();
            self.setup_windows();
            %==========================================================================

            %% Do segmentation
            self.calc_all_segments();
            
            %==========================================================================


        end
        
        function setup_windows(self)
            % Set up windows and preprocess
            setup_windows@AbstractWindowDmd(self, size(self.dat));
            
            self.all_error_reductions = zeros(self.num_clusters, 1);
        end
        
    end
    
    methods % Segmentation
        function calc_all_segments(self)
            
            is_same_segment = false;
            which_segment = 0;
            j = 0;
            while true
                j = j+1;
                if j>self.num_clusters
                    break
                end
                if self.verbose
                    fprintf('Processing window %d\n',j)
                end
                if ~is_same_segment
%                     fprintf('Starting new segment at window %d\n',j)
                    [A0, B0] = self.initialize_segment(j);
                    is_same_segment = ~self.force_redo_every_window;
                    which_segment = which_segment + 1;
                else
                    is_same_segment = check_next_segment(self, j, A0, B0);
                    if ~is_same_segment
                        % Redo this window
                        j = j-1;
                    end
                end
                self.all_B0 = [self.all_B0, B0(:,1)];
                self.segment_indices(j) = which_segment;
            end
        end
        
        function [A, B] = initialize_segment(self, which_window)
            % Do basic DMDc on a window
            this_ind = self.window_ind(:,which_window);
            this_dat = self.dat(:,this_ind);
            u = ones(size(this_ind))';
            ex_u = [];
            if ~isempty(self.external_u)
                ex_u = self.external_u(:,this_ind);
            end
            u = [u; ex_u];
            u = u(:,1:end-1);
            [A, B] = self.do_dmdc(this_dat, u);
            % Also do dmd without control, to compare the error reduction
            [A_full, ~] = self.do_dmdc(this_dat, 0);
            
            if isempty(ex_u)
                err = self.calc_dmd_error(this_dat, A, 0);
%                 err_full = self.calc_dmd_error(this_dat, A_full, 0);
                err_full = self.calc_dmd_error(this_dat, A_full, u);
%                 error_reduction = (norm(err) - norm(err-B))/norm(err);
                error_reduction = (norm(err_full) - norm(err))/norm(err_full);
            else
                external_BU = B(:,2:end)*ex_u(:,1:end-1);
                err = self.calc_dmd_error(this_dat, A, external_BU);
                error_reduction = (norm(err) - norm(err-B(:,1)))/norm(err);
            end
            self.all_error_reductions(which_window) = 100*error_reduction;
        end
        
        function is_same = check_next_segment(self, which_window, A0, B0)
            % Use A0 from a previous segment to get an error matrix, then
            % check if the error is mostly explained by the same control
            % matrix (B0)
            this_ind = self.window_ind(:,which_window);
            this_dat = self.dat(:,this_ind);
            if isempty(self.external_u)
                err = self.calc_dmd_error(this_dat, A0, 0);
                error_reduction = (norm(err) - norm(err-B0))/norm(err);
            else
                external_BU = B0(:,2:end)*self.external_u(:,this_ind);
                external_BU = external_BU(:,2:end);
                err = self.calc_dmd_error(this_dat, A0, external_BU);
                error_reduction = (norm(err) - norm(err-B0(:,1)))/norm(err);
            end
            
            if ~self.force_redo_every_window
                is_same = error_reduction>0.5;
            else
                is_same = false;
            end
            self.all_error_reductions(which_window) = error_reduction;
        end
    end
    
    methods (Access=private)
        function preprocess(self)
            self.dat = self.raw - mean(self.raw,2);
            self.sz = size(self.dat);
            
            self.all_B0 = [];
            self.segment_indices = zeros(self.sz(1),1);
        end
        
        function set_defaults(self)
            defaults = struct(...
                'verbose',true,...
                'force_redo_every_window', false,...
                'external_u', []);
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
        end
        
        function import_from_struct(self, Zimmer_struct)
            warning('Assumes struct of Zimmer type')
            % First, get data
            self.raw = Zimmer_struct.traces.';
        end
        
    end
    
    methods (Static)
        function [A, Bhat] = do_dmdc(X, U)
            % Uses Zhe's code
            %   Note: have to decide on a truncation rank for both
            %   the dynamics and the control signal
            X1 = X(:,1:end-1);
            X2 = X(:,2:end);
            if U == 0
                U = zeros(1, size(X1,2));
            end
            r = optimal_truncation(X2);
            rtilde = 0; % Added adaptive discovery of truncation

            [~, ~, Bhat, ~, Uhat, ~, ~, ~, ~, ~, Atilde] = ...
                func_DMDc(X1, X2, U, r, rtilde);
            
            % Put in back in full (neuron) space
            U = Uhat(:,1:r);
            A = U*Atilde*U';
        end
        
        function err = calc_dmd_error(X, A, BU)
            % Calculates the error using a previously calculated A matrix
            %   The U here will generally be an externally identified
            %   controller
            X1 = X(:,1:end-1);
            X2 = X(:,2:end);
            
            err = X2 - A*X1 - BU;
            if ~isempty(BU)
                err = err(:,1:size(A,2));
            end
        end
            
    end
end

