function fig = plot_gray_lines(dat, alpha)
% Plots a matrix of dat as gray lines, with columns as time and rows as
% channels
if ~exist('alpha', 'var')
    alpha = 1.3 - tanh(size(dat,1));
end

fig = figure;
hold on
for i = 1:size(dat,1)
    plot(dat(i,:), 'Color', [1 1 1]-alpha, 'LineWidth', 2)
end
end

