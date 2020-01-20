function [accuracy_mat] = calc_accuracy(dat, recon)
% Uses the external 'calc_false_detection' function to get false
% positives/negatives/etc. and then calculates the composite 'accuracy'
% score
sz_dat = size(dat,1);
accuracy_mat = zeros(sz_dat + size(recon,1));
for i = 1:sz_dat
    for i2 = 1:size(recon,1)
        [num_fp, num_fn, ~, ~, ~,...
            true_pos, ~, ~, true_neg] = ...
            calc_false_detection(dat(i,:), recon(i2,:));
        t = true_pos + true_neg;
        accuracy_mat(i, sz_dat+i2) = t / (t + num_fp + num_fn);
    end
end

% Put in same format as squareform(pdist(...,'correlation'))


end

