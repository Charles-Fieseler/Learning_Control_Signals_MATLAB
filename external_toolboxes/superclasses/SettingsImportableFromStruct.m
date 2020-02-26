classdef SettingsImportableFromStruct < handle
    
    properties
        settings
    end
    
    methods
        %Constructor
        function self = SettingsImportableFromStruct()
            %             self.settings = settings;
        end
        
        function self = import_settings_to_self(self, new_settings)
            %import settings from a struct, overwriting a struct of defaults
            %   Used as a toolbox function in pretty much all of my functions
            %
            %   new_settings: should be passed in with fields to update, if any
            %
            %   self: set the default settings for that object
            
            if ~exist('new_settings','var')
                new_settings = struct;
            end
            if ~exist('self','var')
                error('Must pass a class or struct')
            end
            
            self.settings = new_settings;
            newNames = fieldnames(new_settings);
            
            for j = 1:length(newNames)
                n = newNames{j};
                
                if isprop(self, n)
                    self.check_types(self, new_settings, n);
                    
                    %Copy to the full object
                    self.(n) = new_settings.(n);
                else
                    warning('Setting "%s" not a property of the object.\n', n)
                end
            end %for
        end %func
        
    end
    
    methods
        function [all_settings, self] = ...
                import_settings(self, new_settings, all_settings )
            %import settings from a struct, overwriting a struct of defaults
            %   Used as a toolbox function in pretty much all of my functions
            %
            %   all_settings: should be passed in with defualt values for ALL fields of
            %                   interest
            %   new_settings: should be passed in with fields to update, if any
            %
            %   self: (Optional) additionally set the
            %               default settings for that object, if they are properties
            
            if ~exist('new_settings','var')
                new_settings = struct;
            end
            if ~exist('self','var')
                %So that isempty(self)==true
                self = struct([]);
            end
            
            namesS = fieldnames(new_settings);
            namesD = fieldnames(all_settings);
            
            for j = 1:length(namesS)
                n = namesS{j};
                
                if max(strcmp(n,namesD)) > 0 %Check to see if the given setting is used
                    self.check_types(new_settings, all_settings, n);
                    
                    all_settings.(n) = new_settings.(n);
                else
                    warning('Setting "%s" not used.\n', n)
                end
                %Copy to the full object whether updated or default
                if ~isempty(self) && isprop(self, n)
                    self.(n) = all_settings.(n);
                end
            end %for
        end %func
        
        function check_types(~, obj_A, obj_B, field_name)
            %Do basic type checking for the field of the objects, if passed
            % This does NOT check for the same sizes in arrays
            if exist('field_name','var')
                typeA = class(obj_A.(field_name));
                typeB = class(obj_B.(field_name));
            else
                typeA = class(obj_A);
                typeB = class(obj_B);
            end
            assert(isequal(typeA, typeB),...
                'Expected type %s for variable %s; found type %s',...
                typeB, field_name, typeA)
        end
        
    end %methods
    
end

