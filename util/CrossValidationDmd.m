classdef CrossValidationDmd < SettingsImportableFromStruct
    % CrossValidationDmd
    %   Uses random column slices to cross-validate a DMD or DMDc model
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
    % Created: 04-Sep-2018
    %========================================
    
    
    properties (SetAccess={?SettingsImportableFromStruct}, Hidden=true)
        % Basic properties
        verbose
        seed
        num_test_columns
        num_folds_to_test
        % For dmdc
        control_signal
        % For dmd_func
        dmd_args
    end
    
    properties
        % Imported
        dat
        sz
        dmd_func
        
        % Calculated in constructor
        fold_indices
        fold_indices_plus_1
        num_folds
        used_sz
        
        % Calculated from data
        test_errors
        train_errors
    end
    
    methods
        function self = CrossValidationDmd(dat, dmd_func, settings)
            
            %% Set defaults and import settings
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
            %==========================================================================
            
            %% Import data
            if isnumeric(dat)
                self.dat = dat;
            else
                error('Must pass data matrix')
            end
            assert(isa(dmd_func,'function_handle'),...
                'Must pass a function handle.')
            self.dmd_func = dmd_func;
            self.preprocess();
            %==========================================================================

            %% Split data into disjoint folds
            self.calc_fold_indices();
            if self.num_folds_to_test == 0
                self.num_folds_to_test = self.num_folds;
            end
            %==========================================================================

            %% Perform dmd on each fold
            self.calc_all_test_errors();
            %==========================================================================

        end
        
        function set_defaults(self)
            defaults = struct(...
                'verbose', true,...
                'seed', 1,...
                'num_test_columns', 4,...
                'num_folds_to_test', 0,...
                'control_signal', [],...
                'dmd_args', {{}});
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
        end
        
        function preprocess(self)
            self.sz = size(self.dat);
        end
        
        function calc_fold_indices(self)
            % Calculates folds by randomly assigning columns, making sure
            % that the columns come from equally distributed blocks
            
            % First calculate blocks, evenly spaced with equal sizes
            %   Each COLUMN refers to a block;
            sz_blocks = self.num_test_columns;
            num_blocks = floor(self.sz(2)/sz_blocks);
            these_ind = 1:(sz_blocks*num_blocks);
            if sz_blocks*num_blocks == self.sz(2)
                these_ind = these_ind(1:end-sz_blocks);
            end
            block_ind = reshape(these_ind, [num_blocks, sz_blocks]);
            self.num_folds = num_blocks;
            self.used_sz = [self.sz(1), these_ind(end)];
            
            % Next randomize to get the real indices
            %   Each ROW refers to a fold
            rng(self.seed);
            self.fold_indices = zeros(size(block_ind));
            for i = 1:sz_blocks
                self.fold_indices(:,i) = block_ind(randperm(num_blocks),i);
            end
        end
        
        function calc_all_test_errors(self)
            % Uses calculated fold indices to produce a dmd model and a
            % test and training error
            test_err_vec = zeros(self.num_folds_to_test,1);
            train_err_vec = zeros(self.num_folds_to_test,1);
            
            for i = 1:self.num_folds_to_test
                if self.verbose && (mod(i,10)==0 || i==1)
                    fprintf('Calculating fold %d/%d...\n',i,self.num_folds)
                end
                [ind_test, ind_train] = ...
                    get_fold_indices(self, i);
                [train_err_vec(i), test_err_vec(i)] = ...
                    self.calc_one_dmd_test(ind_test, ind_train);
            end
            
            self.test_errors = test_err_vec;
            self.train_errors = train_err_vec;
        end
        
        function [train_err, test_err] = ...
                calc_one_dmd_test(self, ind_test, ind_train)
            % Calculates a single test run
            X1 = self.dat(:,ind_train);
            X2 = self.dat(:,ind_train+1);
            X1_test = self.dat(:,ind_test);
            X2_test = self.dat(:,ind_test+1);
            
            arg = self.dmd_args;
            if isempty(self.control_signal)
                [train_err, A] = self.dmd_func(X1, X2, arg);
                test_err = norm(X2_test - A*X1_test)/numel(X1_test);
            else
                [train_err, A, B] = self.dmd_func(X1, X2,...
                    self.control_signal(:,ind_train), arg);
                u_test = self.control_signal(:,ind_test);
                test_err = norm(X2_test - (A*X1_test+B*u_test))/...
                    numel(X1_test);
            end
        end
        
        function [ind_test, ind_train] = ...
                get_fold_indices(self, i)
            % Calculates the indices for fold 'i' split into test and
            % training sets
            ind_test = self.fold_indices(i,:);
%             ind2_test = ind1_test + 1;
            
            % The complement of the test indices are the training indices
            ind_train = 1:self.used_sz(2);
            ind_train(ind_test) = [];
%             ind2_train = ind1_train + 1;
        end
        
    end
    
    methods % Plotting
        function fig = plot_box(self)
            % Simple box plot of both test and training data
            fig = figure;
            boxplot(self.test_errors);
            test_ylim = ylim;
            hold on
            boxplot(self.train_errors);
            all_limits = [test_ylim ylim];
            ylim([min(all_limits), max(all_limits)]);
            % Now for reference
            text(1.1, mean(self.test_errors), '<- mean for testing')
            text(1.1, mean(self.train_errors), '<- mean for training')
        end
    end
end

