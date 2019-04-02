function [final_dat] = calculate_correlations_and_differences(...
    all_models_table, all_settings_loop_names)
% Calculates correlations between models and their data for varying control
% signals, splitting the output data by interpretable neuron clusters
sz = size(all_models_table);
all_dat_table = cell(sz);
baseline_dat_table = cell(sz(1),1);
dat_func = @calc_correlation_matrix;
baseline_func = @calc_linear_correlation;
for i = 1:sz(1)
    fprintf('Analyzing filename: %d\n', i);
    for i2 = 1:sz(2)
        all_dat_table{i,i2} = dat_func(all_models_table{i,i2});
    end
    baseline_dat_table{i} = baseline_func(all_models_table{i,1});
end
all_dat_table = cell2table([baseline_dat_table all_dat_table]);
all_dat_table.Properties.VariableNames = ...
    ['Linear_correlation' all_settings_loop_names];

% Build comparable clusters
%   Note: Each row of the model table has the same neurons and dataset
interpretable_clusters = { ...
    {'AVA', 'RIM', 'AIB', 'RME', 'VA', 'DA',},...
    {'AVB', 'RIB', 'RME'},...
    {'SMDD'},...
    {'SMDV', 'RIV'} ,...
    {'ASK', 'OLQ', 'URY', 'AVF', 'RIS', 'IL'}};
interpretable_clusters_names = {...
    'Reversal', 'Forward', 'Dorsal_turn', 'Ventral_turn', 'Other'};
num_clusters = length(interpretable_clusters);
all_ind_table = cell(sz(1),num_clusters);
for i = 1:sz(1)
    this_model = all_models_table{i,1};
    for i2 = 1:num_clusters
        all_ind_table{i, i2} = ...
            this_model.name2ind(interpretable_clusters{i2});
    end
end
all_ind_table = cell2table(all_ind_table);
all_ind_table.Properties.VariableNames = interpretable_clusters_names;
% Now calculate the data differences by cluster
all_differences = {...
    'Linear_correlation',...
    'No_control',...
    'Rev_and_turns',...
    {'Reversal', 'Linear_correlation'},...
    {'Rev_and_turns', 'Reversal'},...
    {'All', 'Rev_and_turns'} };
all_differences_names = {...
    'Linear', 'No_Control', 'Full',...
    'Reversal', 'Turn', 'Forward'};
num_differences = length(all_differences);
clust_dat_cell = cell(num_differences, 1);
sz = size(all_ind_table);
for iDiff = 1:num_differences
    this_dat = cell(sz);
    if iscell(all_differences{iDiff})
        n1 = all_differences{iDiff}{1};
        n2 = all_differences{iDiff}{2};
    else
        n1 = all_differences{iDiff};
        n2 = [];
    end
    for iFile = 1:sz(1)
        dat1 = all_dat_table{iFile, n1}{1};
        if isempty(n2)
            dat2 = zeros(size(dat1));
        else
            dat2 = all_dat_table{iFile, n2}{1};
        end
        for iClust = 1:sz(2)
            ind = all_ind_table{iFile, iClust}{1};
            this_dat{iFile, iClust} = dat1(ind) - dat2(ind);
        end
    end
    this_dat = cell2table(this_dat);
    this_dat.Properties.VariableNames = interpretable_clusters_names;
    clust_dat_cell{iDiff} = this_dat;
end

% Then combine across individuals
final_dat_cell = cell(num_differences, num_clusters);
for i = 1:num_differences
    for i2 = 1:num_clusters
        final_dat_cell{i, i2} = real(vertcat(clust_dat_cell{i}{:,i2}{:}));
    end
end
final_dat = cell2table(final_dat_cell);
final_dat.Properties.VariableNames = interpretable_clusters_names;
final_dat.Properties.RowNames = all_differences_names;

end

