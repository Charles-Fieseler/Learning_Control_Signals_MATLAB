function [u,s,v,proj3d] = plotSVD( dat, which_plots )
%quick SVD plotting routine: Takes the SVD, plots the first couple modes
%and the trajectory in those modes
% The input parameters are:
%   dat = the data of use. Note that we plot the SVD modes along the rows

%% Defaults: plot only singular value decay
if ~exist('which_plots','var')
    which_plots = struct();
end
defaults = struct(...
    'average',false,...
    'sigma',true,...
    'PCA3d',false,...
    'PCA_opt','o',...
    'sigma_modes',[1 2],...
    'to_subtract_mean',true);
for key = fieldnames(defaults).'
    k = key{1};
    if ~isfield(which_plots, k)
        which_plots.(k) = defaults.(k);
    end
end
%==========================================================================

%% Subtract off the averages
sz = size(dat);
%datmean = zeros(sz(2),1);
if which_plots.to_subtract_mean
    datmean = mean(dat,2);
    dat = dat - datmean;
%     for j = sz(1):-1:1
%         datmean(j) = mean(dat(j,:));
%         dat(j,:) = dat(j,:) - datmean(j); %Subtract off the average voltage for each site
%     end
end


%% Plot the averages
if which_plots.average
    figure;
    plot(datmean,'o');
    title('Mean values');
    xlabel('Data entry number');
    ylabel('Value; unsure of the units');
    drawnow;
end


%% Take the SVD and plot it
% Just the SVD
[u,s,v] = svd(dat);%Note that this SVD starts from the initial time and might include transients
s = diag(s);

if which_plots.sigma
    %Plot the first 10 singular values
    figure;
    subplot(2,1,1);
    plot(s(1:10),'o');
    title('First 10 singular values');

    %Plot the first 2 modes (2d)
    subplot(2,1,2); hold on;
    num_modes = length(which_plots.sigma_modes);
    leg = cell(num_modes,1);
    for i=1:num_modes
        this_mode = which_plots.sigma_modes(i);
        plot(u(:,this_mode));
        leg{i} = sprintf('Mode %d',this_mode);
    end
    title(sprintf('%d svd modes',num_modes));
    legend(leg);
    xlabel('Data entry number');
    ylabel('Value');
    drawnow;
end

modes3d = u(:,1:3);
proj3d = (modes3d.')*dat;
if which_plots.PCA3d
    % Plot the dynamics in 3d (the first 3 modes)
    figure;
    
    plot3(proj3d(1,:),proj3d(2,:),proj3d(3,:), which_plots.PCA_opt); 
    hold on;
    text(proj3d(1,1),proj3d(2,1),proj3d(3,1),'Start');
    text(proj3d(1,end),proj3d(2,end),proj3d(3,end),'End');
    title('Dynamics in the space of the first three modes')
    xlabel('mode 1'); ylabel('mode 2'); zlabel('mode 3');
    drawnow;
end

end

