function [f1_mat] = calc_f1_score(dat, recon)
% Uses the external 'calc_false_detection' function to get false
% positives/negatives/etc. and then calculates the composite 'F1'
% score
sz_dat = size(dat,1);
f1_mat = zeros(sz_dat + size(recon,1));
window_true_spike = 5;

for i = 1:sz_dat
    for i2 = 1:size(recon,1)
        [num_fp, num_fn, ~, ~, ~,...
            true_pos, ~, ~, ~] = ...
            calc_false_detection(dat(i,:), recon(i2,:),...
            [], [], window_true_spike);
        t = 2*true_pos;
        f1_mat(i, sz_dat+i2) = t / (t + num_fp + num_fn);
    end
end

% Put in same format as squareform(pdist(...,'correlation'))


end

