classdef compare_connectome < SettingsImportableFromStruct
    % Class for comparing experimental and dynamic connectomes
    
    properties (SetAccess=private, Hidden=true)
        % The adjacency matrices
        dyn_adj
        labeled_dyn_adj
        ctr_adj
        labeled_ctr_adj
        exp_adj_chem
        labeled_exp_adj_chem
        exp_adj_gap
        labeled_exp_adj_gap
        
        labeled_exp_adj_full
        
        connectome_struct
        all_exp_names
        sorted_names
    end
    
    properties (SetAccess={?SettingsImportableFromStruct}, Hidden=true)
        id_array
        
        gap_to_chem_weight
        
        to_subtract_diagonal
        to_normalize_max
    end
    
    methods
        function self = compare_connectome(file_or_dat, settings)
            
            %% Set defaults and import settings
            if ~exist('settings','var')
                settings = struct();
            end
            self.set_defaults();
            self.import_settings_to_self(settings);
            %==========================================================================
            
            %% Import data
            if ischar(file_or_dat)
                self.dyn_adj = importdata(file_or_dat);
            elseif isnumeric(file_or_dat)
                self.dyn_adj = file_or_dat;
            elseif isa(file_or_dat, 'AdaptiveDmdc')
                % Extract the connectomic dynamics
                %   TODO: also import and deal with the control matrix
                x = 1:file_or_dat.x_len;
                self.dyn_adj = file_or_dat.A_original(x,x);
                self.id_array = file_or_dat.get_names(x, [], false, false);
                self.ctr_adj = file_or_dat.A_original(x,x(end):end);
            elseif isa(file_or_dat, 'CElegansModel')
                % Extract the connectomic dynamics
                %   TODO: also import and deal with the control matrix
                obj = file_or_dat.AdaptiveDmdc_obj;
                x = 1:obj.x_len;
                self.dyn_adj = obj.A_original(x,x);
                self.id_array = obj.get_names(x, [], false, false);
                self.ctr_adj = obj.A_original(x,x(end):end);
            else
                error('Must pass data matrix or filename')
            end
            connectome = importdata('ConnOrdered_040903.mat');
            self.exp_adj_chem = connectome.A_init_t_ordered;
            self.all_exp_names = connectome.Neuron_ordered;
            self.exp_adj_gap = connectome.Ag_t_ordered;
            self.connectome_struct = connectome;
            
            self.preprocess();
            %==========================================================================
            
        end
    end
    
    methods % Plotting
        function plot_imagesc(self, use_ctr_signal)
            if ~exist('use_ctr_signal','var')
                use_ctr_signal = false;
            end
            if ~use_ctr_signal
                this_dat = self.labeled_dyn_adj;
                title_str = 'Dynamic connectome';
            else
                this_dat = self.labeled_ctr_adj;
                title_str = 'Dynamic connectome (control signals)';
            end
            plot_2imagesc_colorbar(...
                this_dat, self.labeled_exp_adj_full, '1 2',...
                title_str, 'Experimental connectome')
            yticks(1:length(self.sorted_names))
            yticklabels(self.sorted_names)
            xticks(1:length(self.sorted_names))
            xticklabels(self.sorted_names)
            xtickangle(90)
        end
    end
    
    methods % Comparison metrics
        function err = hamming_by_row(self, row_num)
            % Calculates the Hamming distance between the experimental and
            % dynamic adjacency matrices
            % Default: calculate distance between entire matrices
        end
    end
    
    methods % Data processing
        
        function set_defaults(self)
            defaults = struct(...
                'id_array', {{}},...
                'gap_to_chem_weight', 1.0,...
                'to_subtract_diagonal', false,...
                'to_normalize_max', true);
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end
        end
        
        function preprocess(self)
            % Reduces the adjacency matrix to the subset of identified
            % neurons and creates an analagous experimental connectome of
            % the same neurons
            
            % Dynamic adjacency matrix
            [self.labeled_dyn_adj, self.sorted_names] = ...
                self.sort_names_and_adj(...
                self.id_array, self.dyn_adj);
            if ~isempty(self.ctr_adj)
                warning('Assuming the control matrix has the same neuron labels')
                if size(self.ctr_adj,2) > size(self.dyn_adj,2)
                    warning('Cutting off extra control signals')
                    this_ctr_adj = self.ctr_adj(:,1:size(self.dyn_adj,2));
                else
                    this_ctr_adj = self.ctr_adj;
                end
                [self.labeled_ctr_adj, ~] = ...
                    self.sort_names_and_adj(...
                    self.id_array, this_ctr_adj);
            end
            % Experimental adjacency matrix
            %   First, get the relevant subset
            this_all_exp_names = self.all_exp_names;
            neurons_to_keep = ismember(this_all_exp_names, self.id_array);
            this_all_exp_names(~neurons_to_keep) = {''};
            %   Second, do the chemical matrix and the gap junction matrix
            self.labeled_exp_adj_chem = self.sort_names_and_adj(...
                this_all_exp_names, self.exp_adj_chem);
            self.labeled_exp_adj_gap = self.sort_names_and_adj(...
                this_all_exp_names, self.exp_adj_gap);
            
            self.labeled_exp_adj_full = self.labeled_exp_adj_gap + ...
                (self.labeled_exp_adj_chem * self.gap_to_chem_weight);
            
            % Normalizations for visualization
            %   Note that diagonal entries don't really exist in the
            %   experimental connectomes, and their interpretation is
            %   somewhat different
            if self.to_subtract_diagonal
                self.labeled_dyn_adj = self.labeled_dyn_adj - ...
                    diag(diag(self.labeled_dyn_adj));
            end
            if self.to_normalize_max
                self.labeled_dyn_adj = self.labeled_dyn_adj * ...
                    max(max(self.labeled_exp_adj_full)) / ...
                    max(max(self.labeled_dyn_adj));
            end
        end
        
    end
    
    methods (Static)
        function [sorted_dat, sorted_names] = ...
                sort_names_and_adj(name_array, dat)
            % Sorts out the unlabeled neurons
            actually_labeled_ind = find(~cellfun(@isempty, name_array));
            num_labels = length(actually_labeled_ind);
            % Sort the dynamic ids and only keep those rows and columns
            [sorted_names, I_dyn] = sort(name_array);
            sorted_dat = dat(I_dyn,I_dyn);
            % Can only sort ascending, i.e. names sorted to the end
            n = length(dat);
            actually_labeled_ind = (n-num_labels):n;
            sorted_dat = sorted_dat(actually_labeled_ind, actually_labeled_ind);
            
            sorted_names = sorted_names(~cellfun(@isempty,sorted_names));
        end
    end
end

