function [all_filenames, num_type_1] = get_Zimmer_filenames()

n = 15;
all_filenames = cell(n, 1);
base_folder = 'C:\Users\charl\Documents\MATLAB\Collaborations\Zimmer_data\';
% foldername1 = '../../Zimmer_data/WildType_adult/';
foldername1 = [base_folder 'WildType_adult\'];
% filename1_end = 'simplewt%d\wbdataset.mat';
filename1_end = 'wbdataset.mat';
num_type_1 = 5;
foldername2 = [base_folder 'npr1_1_PreLet\'];
filename2_end = 'wbdataset.mat';

for i = 1:n
    if i <= num_type_1
        subfolder = dir(foldername1);
        all_filenames{i} = [foldername1, ...
            subfolder(i+2).name, '\', filename1_end];
%         all_filenames{i} = sprintf([foldername1, filename1_end], i);
    else
        subfolder = dir(foldername2);
        all_filenames{i} = [foldername2, ...
            subfolder(i-num_type_1+2).name, '\', filename2_end];
    end
end

end