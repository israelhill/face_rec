clc; clear all; close all;

%% file input
% reading in image files
if ismac
    folderpath = 'images/CroppedYale_fixed/yaleB%02d/*.pgm';
else
    folderpath = 'images\\CroppedYale_fixed\\yaleB%02d\\*.pgm';
end
for i = 1:39
   srcfiles = dir(sprintf(folderpath,i));
   for j = 1:length(srcfiles)
       if ismac
           file = strcat(sprintf('images/CroppedYale_fixed/yaleB%02d/',i),srcfiles(j).name);
       else
           file = strcat(sprintf('images\\CroppedYale_fixed\\yaleB%02d\\',i),srcfiles(j).name);
       end
       cropped_set{i,j} = imread(file);
   end
   face_divides(i) = length(srcfiles);
end

% removing empty files
index = 1;
for i = 1:39
    for j = 1:64
        if ~isempty(cropped_set{i,j})
            cropped_set_fixed(:,:,index) = cropped_set{i,j};
            index = index+1;
        end
    end
end

% putting faces into buckets
face_divides(face_divides == 0) = [];
face_divides = cumsum(face_divides);
face_divides = [0 face_divides];
for i = 1:39
    if i ~= 14
        face_names{i} = strcat('Face',num2str(i));
    end
end
face_names = face_names(~cellfun(@isempty,face_names));
face_buckets = discretize([1:2414],face_divides,'categorical',face_names,'IncludedEdge','right');

%% split into testing and training sets
[training_set,training_idx] = datasample(cropped_set_fixed,size(cropped_set_fixed,3)/2,3,'Replace',false);
testing_set = cropped_set_fixed;
testing_set(:,:,training_idx) = [];

%% averaging
avg_cropped = mean(training_set,3);
figure
imagesc(avg_cropped); colormap gray; title('Average Face');

%% reshape
for j = 1:length(training_set)
    cropped_vector(:,j) = reshape(training_set(:,:,j),[1 192*168]);
end
avg_cropped_vector = reshape(avg_cropped,[192*168 1]);
avg_cropped_vector = repmat(avg_cropped_vector,1,size(training_set,3));
figure
imagesc(cropped_vector); colormap gray; title('Rearranged Faces');
figure
imagesc(avg_cropped_vector); colormap gray; title('Rearranged Average Faces');

%% subtract the mean
A = single(double(cropped_vector)-avg_cropped_vector);

%% SVD
[U,S,V] = svd(A,'econ');

%% showing eigenfaces
for j = 0:27
    subplot(4,7,j+1)
    imagesc(reshape(U(:,j+1),192,168)); colormap gray
end

%% weights
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
close all; clc;
test_img_set = randi(length(testing_set),5);
for i = 1:5
    test_img = test_img_set(i);
    for j = 1:length(omega_testing)
        euclidian_arr(j) = norm(omega_testing(:,test_img)-omega_training(:,j));
    end
    [prediction, p_index] = min(euclidian_arr);
    
    % this returns the cropped_set_fixed index
    csf_idx = training_idx(p_index);
    face_num = char(face_buckets(csf_idx));
    figure
    subplot(1,2,1)
    imshow(testing_set(:,:,test_img)); title('Input face');
    subplot(1,2,2)
    imshow(cropped_set_fixed(:,:,csf_idx)); title(strcat('Predicted: ',face_num));
    
    % finds possible matches from training set
    possible_matches = find(face_buckets == face_buckets(csf_idx));
    for i = 1:length(possible_matches)
        possible_idx = possible_matches(i);
        face_diff(:,:,i) = cropped_set_fixed(:,:,possible_idx)-testing_set(:,:,test_img);
        matched_face(i) = sum(sum(face_diff(:,:,i)));
    end
    matched_face = matched_face == 0;
    if sum(matched_face) == 0
        disp('No matching image found in predicted face')
    else
        matched_face_idx = find(matched_face == 1);
        disp(['Matching face found! Image ' num2str(matched_face_idx)])
    end
end
% randomly split into half for training and testing
% get index of training set
% divide index into faces
% correspond new training set index with complete index
% say which face it think it corresponds to
% do difference on all the images from that face to confirm