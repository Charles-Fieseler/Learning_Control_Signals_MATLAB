classdef AbstractDmd < SettingsImportableFromStruct
    % Abstract DMD class; do not call directly!
    %
    % INPUTS
    %   INPUT1 -
    %   INPUT2 -
    %
    % OUTPUTS -
    %   OUTPUT1 -
    %   OUTPUT2 -
    %
    %
    % Dependencies
    %   Other m-files required: (updated on 29-Nov-2017)
    %             MATLAB (version 9.2)
    %             v2struct.m
    %
    %   See also: OTHER_FUNCTION_NAME
    %
    %
    %
    % Author: Charles Fieseler
    % University of Washington, Dept. of Physics
    % Email address: charles.fieseler@gmail.com
    % Website: coming soon
    % Created: 29-Nov-2017
    %========================================
    
    properties (Hidden=true)
        dat
        verbose
        model_order
        augment_data
        sz
        t0_each_bin %Restart the dmd modes at t=0 at each bin
        tspan
        
        approx_all
        
        %For the DMDplotter object
        plotter_set
        %Imported
        filename
        original_sz
        %User processing settings
        dt
        to_subtract_mean
        dmd_percent
        %Plotter dictionary, if option is used
        PlotterDmd_all
    end
    
    properties (Hidden=true, Transient=true)
        raw
    end
    
    
    methods
        
        function self = AbstractDmd(subclass_set_names, settings)
            %% Initialize with defaults
            defaults = struct(... %The default values
            	'verbose',true,...
                'dt',1,...
            	'dmd_percent',0.95,...
            	'model_order',-1,...
            	'to_subtract_mean',false,...
            	'plotter_set',struct(),...
            	'augment_data',0,...
            	't0_each_bin',true);
            
            if ~exist('settings','var')
                settings = struct;
            end
            if ~exist('subclass_set_names','var')
                subclass_set_names = {''};
            end
            
            namesS = fieldnames(settings);
            namesD = fieldnames(defaults);
            
            for j = 1:length(namesS)
                n = namesS{j};
                
                if ismember(n, namesD)
                    defaults.(n) = settings.(n);
                elseif ~ismember(n, subclass_set_names)
                    fprintf('Warning: "%s" setting not used\n',n)
                end
            end
            
            for key = fieldnames(defaults).'
                k = key{1};
                self.(k) = defaults.(k);
            end %Unpacks the struct into variables
            
            %Make sure the settings are the same for the plotter objects
            self.plotter_set.dt = self.dt;
            %             self.plotter_set.to_subtract_mean = self.to_subtract_mean;
            %             self.plotter_set.dmd_percent = self.dmd_percent;
            %             self.plotter_set.model_order = self.model_order;
            
            % Initialize the DMD object containers
            self.PlotterDmd_all = containers.Map();
            self.approx_all = containers.Map();
            %==========================================================================
        end
        
        
    end
    
    methods (Access=public)
        %Functions for accessing containers
        function key = vec2key(~, vec)
            %Returns the key value corresponding to the vector of format:
            %   vec(1:2) = (layer, time bin index)
            key = num2str(vec);
        end
        
        function vec = key2vec(~, key)
            %Returns the vector corresponding to the key value of format:
            %   vec(1:2) = (layer, time bin index)
            vec = str2num(key); %#ok<ST2NM>
        end
        
        %Data processing
        function preprocess(self)
            if self.verbose
                disp('Preprocessing...')
            end
            self.sz = size(self.raw);
            
            %If augmenting, stack data offset by 1 column on top of itself;
            %note that this decreases the overall number of columns (time
            %slices)
            aug = self.augment_data;
            self.original_sz = self.sz;
            if aug>0
                newSz = [self.sz(1)*aug, self.sz(2)-aug];
                newDat = zeros(newSz);
                for j=1:aug
                    thisOldCols = j:(newSz(2)+j-1);
                    thisNewRows = (1:self.sz(1))+self.sz(1)*(j-1);
                    newDat(thisNewRows,:) = self.raw(:,thisOldCols);
                end
                self.sz = newSz;
                self.raw = newDat;
            end
            
            self.dat = self.raw;
            if self.to_subtract_mean
                for jM=1:self.sz(1)
                    self.dat(jM,:) = self.raw(jM,:) - mean(self.raw(jM,:));
                end
            end
        end
        
    end
    
end

