function [obj_array] = initialize_multiple_models(...
    filename_array, object_initializer, base_settings, ...
    settings_loop_fieldname, settings_loop_vals)
% Creates an array of objects, using the (data) files in filename_array and
% different settings matrices, if settings_loop_fieldname and 
% settings_loop_vals are set
if ~exist('object_initializer', 'var')
    object_initializer = @CElegansModel;
end
if ~exist('base_settings', 'var')
    base_settings = struct();
end
if ~exist('settings_loop_fieldname', 'var')
    settings_loop_fieldname = {};
end

n = length(filename_array);
m = length(settings_loop_fieldname);
obj_array = cell(n, m);
for iFile = 1:n
    fprintf('Analyzing file number %d\n', iFile)
    for iSettings = 1:max(m,1)
        if isstruct(base_settings)
            settings = base_settings;
        elseif iscell(base_settings)
            settings = base_settings{iFile};
        end
        assert(isstruct(settings),...
            'Must pass scalar or vector array of structs in base_settings')
        if ~isempty(settings_loop_fieldname)
            settings.settings_loop_fieldname = ...
                settings_loop_vals{iSettings};
        end
        obj_array{iFile, iSettings} = object_initializer(...
            filename_array{iFile}, settings);
    end
end
    
end