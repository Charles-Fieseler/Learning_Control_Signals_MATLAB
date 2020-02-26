function [all_starts, all_ends] = ...
    calc_contiguous_blocks(binary_ind, length_for_detection, gap_length_okay)
% Given a binary vector with blocks of 1's and 0's, returns all the
% starting and ending indices for the 1-blocks
% Input
%   binary_ind - the data
%   length_for_detection (1) - the length a series of 1's must be to count
%   gap_length_okay (0) - the minimum length of a gap needed to break a
%       contiguous block
%
% Output:
%   all_starts - the indices
%   all_ends - the indices, where binary_ind(all_ends(1)) == 1 is true
if ~exist('length_for_detection', 'var')
    length_for_detection = 1;
end
if ~exist('gap_length_okay', 'var')
    gap_length_okay = 0;
end

assert(islogical(binary_ind), 'Must pass a logical vector')
if size(binary_ind,1) > 1
    binary_ind = binary_ind.';
end
assert(size(binary_ind,1) == 1,...
    'Must pass a logical VECTOR')

all_starts = diff(binary_ind);
all_ends = find(all_starts.*(all_starts<0));
all_starts = find(all_starts.*(all_starts>0)) + 1;
% Fix ends
if binary_ind(1)
    all_starts = [1 all_starts];
end
if length(all_starts) > length(all_ends)
    all_ends = [all_ends length(binary_ind)];
end
% Remove small blocks and connect small gaps
if gap_length_okay > 0
    all_gap_lengths = all_starts(2:end) - all_ends(1:end-1);
    long_enough = (all_gap_lengths >= gap_length_okay);
    start_ind = true(size(all_starts));
    end_ind = true(size(all_starts));
    for i = 1:length(long_enough)
        if ~long_enough(i)
            start_ind(i+1) = false;
            end_ind(i) = false;
        end
    end
    all_ends = all_ends(end_ind);
    all_starts = all_starts(start_ind);
end

all_block_lengths = all_ends - all_starts + 1;
ind = (all_block_lengths >= length_for_detection);
all_ends = all_ends(ind);
all_starts = all_starts(ind);

end

