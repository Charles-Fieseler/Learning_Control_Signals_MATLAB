function [new_dat] = time_delay_embed(dat, aug)
% Time-delay embeds the dat
sz = size(dat);
if aug>1
    new_sz = [sz(1)*aug, sz(2)-aug];
    new_dat = zeros(new_sz);
    for j = 1:aug
        old_cols = j:(new_sz(2)+j-1);
        new_rows = (1:sz(1)) + ...
            sz(1)*(aug-j);
        new_dat(new_rows,:) = dat(:,old_cols);
    end
else
    new_dat = dat;
end
end

