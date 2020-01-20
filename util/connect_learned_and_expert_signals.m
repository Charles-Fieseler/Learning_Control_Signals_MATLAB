function [correlation_table_raw] = ...
    connect_learned_and_expert_signals(all_U2, all_models, f_dist)
% Uses a cell array of learned signals (across datasets, all_U2) and the
% corresponding models (all_models) to build a table of correspondance between the
% learned and expert-identified signals
% Input:
%   f_dist (correlation) - the distance function to use
% corr_threshold = [0.1, 0.3]; % Signals need an offset; want a threshold after as well

if ~exist('f_dist', 'var') || isempty(f_dist)
    f_dist = @(x1, x2) 1 - squareform(pdist([x1; x2], 'correlation'));
    f_dist_str = 'correlation';
else
    f_dist_str = 'custom';
end

corr_threshold = [0, 0]; % Signals need an offset; want a threshold after as well
max_offset = 20;
num_files = length(all_models);
correlation_table_raw = table();
for i = 1:num_files
    tm = all_models{i};
    learned_ctr = all_U2{i};
%     learned_n = size(learned_ctr, 1);
    exp_ctr = tm.control_signal(:,1:end-1);
    exp_n = size(exp_ctr, 1);
%     this_dat = [exp_ctr; learned_ctr];
%     all_corr = f_dist(this_dat);
    all_corr = f_dist(exp_ctr, learned_ctr);
    if strcmp(f_dist_str, 'correlation')
        all_corr = all_corr - diag(diag(all_corr));
    end
    
%     key_ind = contains(tm.state_labels_key, tm.global_signal_subset);
%     key = tm.state_labels_key(key_ind);
    % Build correlation for all control signals
    key = tm.state_labels_key;
    model_index = i;
    % TODO: refactor using finddelay()
    for i2 = 1:length(key)
        [max_corr, max_ind] = max(all_corr(i2,:));
        % Note: make sure you're not correlating learned with learned
        if (max_corr > corr_threshold(1)) && (max_ind>exp_n)
            % See if an offset gives a better correlation
            best_offset = 0;
            for i3 = 1:max_offset
                % Positive offset
%                 tmp = f_dist([this_dat(i2,1:end-i3);...
%                     this_dat(max_ind,i3+1:end)]);
                tmp = f_dist(exp_ctr(i2,1:end-i3),...
                    learned_ctr(max_ind-exp_n, i3+1:end));
                this_corr = tmp(1,2); % off-diagonal term
                if this_corr > max_corr
                    max_corr = this_corr;
                    best_offset = i3;
                end
                % Negative offset
%                 tmp = f_dist([this_dat(i2,i3+1:end);...
%                     this_dat(max_ind,1:end-i3)]);
                tmp = f_dist(exp_ctr(i2,i3+1:end),...
                    learned_ctr(max_ind-exp_n,1:end-i3));
                this_corr = tmp(1,2); % off-diagonal term
                if this_corr > max_corr
                    max_corr = this_corr;
                    best_offset = -i3;
                end
            end
            
            if max_corr > corr_threshold(2)
                experimental_signal_index = i2;
                experimental_signal_name = key(i2);
                learned_signal_index = max_ind - exp_n;
                maximum_correlation = max_corr;
                correlation_table_raw = [correlation_table_raw;
                    table(model_index, maximum_correlation, best_offset,...
                    experimental_signal_index, experimental_signal_name,...
                    learned_signal_index)]; %#ok<AGROW>
            end
        end
    end
    
%     figure;imagesc(all_corr);colorbar
%     ind = tm.state_labels_ind(1:end-1);
%     key = tm.state_labels_key;
%     fig = figure;
%     subplot(2,1,1)
%     plot_colored(this_dat(1,:), ind, key, 'plot', [], fig);
%     subplot(2,1,2)
%     plot_colored(this_dat(22,:), ind, key, 'plot', [], fig);
end
end

