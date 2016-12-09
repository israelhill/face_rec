clear; clc; close all;
%% read in dataset
% Mac Path
allSubFolders = genpath('images/CroppedYale_fixed');
% Windows Path
% allSubFolders = genpath('images\CroppedYale_fixed');
remain = allSubFolders;
listOfFolderNames = {};
while true
    [singleSubFolder, remain] = strtok(remain, ':');
    if isempty(singleSubFolder)
        break;
    end
    listOfFolderNames = [listOfFolderNames singleSubFolder];
end
numberOfFolders = length(listOfFolderNames);
for k = 1 : numberOfFolders
    thisFolder = listOfFolderNames{k};
    filePattern = sprintf('%s/*.pgm', thisFolder);
    baseFileNames = dir(filePattern);
    numberOfImageFiles = length(baseFileNames);
    if numberOfImageFiles >= 1
        for f = 1 : numberOfImageFiles
            fullFileName = fullfile(thisFolder, baseFileNames(f).name);
            cropped{k,f} = imread(fullFileName);
        end
    end
end

cropped = cropped(~cellfun(@isempty,cropped));
for i = 1:length(cropped)
    cropped_set(:,:,i) = cropped{i};
end

%% averaging
avg_cropped = mean(cropped_set,3);
figure
imagesc(avg_cropped); colormap gray;

%% reshape
for i = 1:length(cropped_set)
    cropped_vector(:,i) = reshape(cropped_set(:,:,i),[1 192*168]);
end
avg_cropped_vector = reshape(avg_cropped,[192*168 1]);
avg_cropped_vector = repmat(avg_cropped_vector,1,2414);
figure
imagesc(cropped_vector); colormap gray;
figure
imagesc(avg_cropped_vector); colormap gray;

%% subtract the mean
A = double(cropped_vector)-avg_cropped_vector;

%% reshape back to faces?
for i = 1:length(cropped_set)
    eigen(:,:,i) = reshape(A(:,i),[192 168]);
end