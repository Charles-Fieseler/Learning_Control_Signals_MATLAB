function [z_struct] = convertKato2Zimmer(filename_states, filename_dat)
% Converts data of the original Kato Science paper format to the single
% struct that my functions expect.
%
% Input
%   filename_states - string of the filename to the struct of state labels
%   filename_dat - string of the filename to the data struct

if iscell(filename_states)
    filename_dat = filename_states{2};
    filename_states = filename_states{1};
end
assert(ischar(filename_states) && ischar(filename_dat),...
    'Must pass strings for both filenames')

%---------------------------------------------
% Import data 
%---------------------------------------------
states = importdata(filename_states);
dat = importdata(filename_dat);

%---------------------------------------------
% Find which run this is
%---------------------------------------------
num_runs = length(states.dataset);
run_names = cell(num_runs,1);
for i = 1:num_runs
    % Is a full filename
    run_names{i} = states.dataset(i).datasetName;
end

this_run_name = strsplit(dat.FlNm,'_');
this_run_name = this_run_name(1);

ind = contains(run_names, this_run_name);

%---------------------------------------------
% Get the proper data and export
%---------------------------------------------
sz = size(dat.NeuronIds);
ID = cell(sz);
ID2 = cell(sz);
ID3 = cell(sz);
for i = 1:sz(2)
    ID{i} = [];
    ID2{i} = [];
    ID3{i} = [];
    if ~isempty(dat.NeuronIds{i})
        if length(dat.NeuronIds{i}) == 1
            ID{i} = dat.NeuronIds{i}{1};
            % A couple are mislabeled
            if strcmp(ID{i},'SMBDL')
                ID{i} = 'SMDDL';
            elseif strcmp(ID{i}, 'SMBDR')
                ID{i} = 'SMDDR';
            elseif contains(ID{i}, '-')
                ID{i} = [];
            end
        elseif length(dat.NeuronIds{i}) == 2
            ID{i} = dat.NeuronIds{i}{1};
            ID2{i} = dat.NeuronIds{i}{2};
        elseif length(dat.NeuronIds{i}) == 3
            ID{i} = dat.NeuronIds{i}{1};
            ID2{i} = dat.NeuronIds{i}{2};
            ID3{i} = dat.NeuronIds{i}{3};
        end
    end
end


z_struct = struct(...
    'traces', dat.deltaFOverF_bc,...
    'tracesDif', dat.deltaFOverF_deriv,...
    'timeVectorSeconds', dat.tv,...
    'ID', {ID},...
    'ID2', {ID2},...
    'ID3', {ID3},...
    'SevenStatesKey', {states.key},...
    'SevenStates', states.dataset(ind).stateTimeSeries,...
    'trialname', this_run_name);
if isfield(dat,'stimulus')
    z_struct.stimulus = dat.stimulus;
else
    z_struct.stimulus = struct();
end

end

