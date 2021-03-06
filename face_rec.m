clear; clc; close all;
%% read in dataset
% Check OS
if ismac
    allSubFolders = genpath('images/CroppedYale_fixed');
else 
    allSubFolders = genpath('images\CroppedYale_fixed');
end

remain = allSubFolders;
listOfFolderNames = {};
while true
    % Check OS
    if ismac
        [singleSubFolder, remain] = strtok(remain, ':');
    else
        [singleSubFolder, remain] = strtok(remain, ';');
    end
    
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
for j = 1:length(cropped)
    cropped_set(:,:,j) = cropped{j};
end

%% split into testing and training sets
[testing_set,testing_idx] = datasample(cropped_set,size(cropped_set,3)/2,3,'Replace',false);
training_set = cropped_set;
training_set(:,:,testing_idx) = [];

%% averaging
avg_cropped = mean(training_set,3);
figure
imagesc(avg_cropped); colormap gray;

%% reshape
for j = 1:length(training_set)
    cropped_vector(:,j) = reshape(training_set(:,:,j),[1 192*168]);
end
avg_cropped_vector = reshape(avg_cropped,[192*168 1]);
avg_cropped_vector = repmat(avg_cropped_vector,1,size(training_set,3));
figure
imagesc(cropped_vector); colormap gray;
figure
imagesc(avg_cropped_vector); colormap gray;

%% subtract the mean
A = double(cropped_vector)-avg_cropped_vector;
A = single(A);

%% SVD
[U,S,V] = svd(A,'econ');

%% showing eigenfaces
for j = 0:27
    subplot(4,7,j+1)
    imagesc(reshape(U(:,j+1),192,168)); colormap gray
end

%% weights
close all
% find weights of training set
figure
hold on
for i = 1:size(training_set,3)   
    for j = 1:13
        w_training(j) = U(:,j)'*A(:,i);
    end
    omega_training(:,i) = w_training;
    plot(w_training)
end
hold off


%% face recognition
% find weight of testing set
for j = 1:length(testing_set)
    cropped_vector_testing(:,j) = reshape(testing_set(:,:,j),[1 192*168]);
end
A_training = double(cropped_vector_testing)-avg_cropped_vector;
A_training = single(A_training);

figure
hold on
for i = 1:size(testing_set,3)   
    for j = 1:13
        w_testing(j) = U(:,j)'*A_training(:,i);
    end
    omega_testing(:,i) = w_testing;
    plot(w_testing)
end
hold off

%% compute Euclidian difference, make prediction
figure
count = 1;
for k = 1:5
    k = k + 5;
    for j = 1:length(omega_training)
        euclidian_arr(j) = norm(omega_training(:,k)-omega_testing(:,j));
    end
    [prediction, p_index] = min(euclidian_arr)

    subplot(5,2,count);
    imagesc(training_set(:,:,k)); colormap gray; title('Face to recognize');
    subplot(5,2,count + 1);
    imagesc(testing_set(:,:,p_index)); colormap gray; title('Prediction');
    count = count + 2;
end
 
% % display original face
% figure
% imagesc(training_set(:,:,5)); colormap gray;
% figure
% imagesc(training_set(:,:,50)); colormap gray;
% 
% % reconstructs face based off weight
% recon5 = zeros(32256,1);
% recon50 = zeros(32256,1);
% for j = 1:13
%     recon5 = recon5+omega(j,5)*U(:,j);
%     recon50 = recon50+omega(j,50)*U(:,j);
% end
% recon5 = reshape(recon5,192,168);
% recon50 = reshape(recon50,192,168);
% figure
% imagesc(recon5); colormap gray;
% figure
% imagesc(recon50); colormap gray;