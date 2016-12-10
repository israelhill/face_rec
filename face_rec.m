clear; clc; close all;
%% read in dataset
% Mac Path
% allSubFolders = genpath('images/CroppedYale_fixed');
% Windows Path
allSubFolders = genpath('images\CroppedYale_fixed');
remain = allSubFolders;
listOfFolderNames = {};
while true
    % Mac Path
    % [singleSubFolder, remain] = strtok(remain, ':');
    % Windows Path
    [singleSubFolder, remain] = strtok(remain, ';');
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
% for i = 1:length(cropped_set)
%     eigen(:,:,i) = reshape(A(:,i),[192 168]);
% end

%% covariance
A = single(A);
% C = cov(A');
% 
% %% eigen
% [V,D] = eig(C);
% eigval = diag(D);
% eigvalsorted = eigval(end:-1:1);
% V = fliplr(V);

%% SVD
[U,S,V] = svd(A,'econ');

%% showing eigenfaces?
for i = 0:15
    subplot(4,4,i+1)
    imagesc(reshape(U(:,i+1),192,168)); colormap gray
end