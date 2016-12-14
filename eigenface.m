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
imwrite(mat2gray(avg_cropped),'avg_face.png');

%% reshape
for j = 1:length(training_set)
    cropped_vector(:,j) = reshape(training_set(:,:,j),[1 192*168]);
end
avg_cropped_vector = reshape(avg_cropped,[192*168 1]);
avg_cropped_vector = repmat(avg_cropped_vector,1,size(training_set,3));
figure
imagesc(cropped_vector); colormap gray; %title('Rearranged Faces');
xlabel('Testing Face'); ylabel('Pixels');
print(gcf,'rearr_test_face','-dpng','-r300');
%imwrite(mat2gray(cropped_vector),'rearrface.png');
figure
imagesc(avg_cropped_vector); colormap gray; title('Rearranged Average Faces');
%imwrite(mat2gray(avg_cropped_vector),'rearr_avg_face.png');

%% subtract the mean
A = single(double(cropped_vector)-avg_cropped_vector);

%% SVD
[U,S,V] = svd(A,'econ');

%% showing eigenfaces
figure
for j = 1:6
    subplot(3,2,j)
    imagesc(reshape(U(:,j),192,168)); axis off; colormap gray
    imwrite(mat2gray(reshape(U(:,j),192,168)),strcat('Eigen',num2str(j),'.png'));
end

%% weights
% find weights of training set
figure
hold on
for i = 1:size(training_set,3)   
    for j = 4:500
        w_training(j) = U(:,j)'*A(:,i);
    end
    omega_training(:,i) = w_training;
    plot(w_training)
end
hold off
xlabel('Eigenface'); ylabel('Weight');
print(gcf,'training_weight','-dpng','-r300');

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
    for j = 4:500
        w_testing(j) = U(:,j)'*A_training(:,i);
    end
    omega_testing(:,i) = w_testing;
    plot(w_testing)
end
hold off

%% compute Euclidian difference, make prediction
%test_img_set = randi(length(testing_set),5);
num_correct = 0;
percent_correct = 0.0;
figure
hold on
for img = 1:length(testing_set)
    test_img = testing_set(:,:,img);
    for j = 1:length(omega_testing)
        euclidian_arr(j) = norm(omega_testing(:,img)-omega_training(:,j));
    end
    [prediction, p_index] = min(euclidian_arr);
    
    % this returns the cropped_set_fixed index
    csf_idx = training_idx(p_index);
    face_num = char(face_buckets(csf_idx));
    possible_matches = find(face_buckets == face_buckets(csf_idx));
    found_match = false;
    for i = 1:length(possible_matches)
        possible_idx = possible_matches(i);
        if (testing_set(:,:,img) == cropped_set_fixed(:,:,possible_idx))
            disp(['Matching face found! Training image ' num2str(img) ' matches ' num2str(possible_idx) '. Predicted: ' num2str(p_index)])
            found_match = true;
            num_correct = num_correct + 1;
            scatter(img,min(euclidian_arr),'.','g')
        end
    end
    if ~found_match
        disp(['No matching image found in predicted face. Closest match: image ' num2str(p_index)])
        scatter(img,min(euclidian_arr),'.','r')
    end
    
end
hold off
xlabel('Test Face'); ylabel('Minimum Euclidean Distance');
print(gcf,'distance','-dpng','-r300');

percent_correct = num_correct / length(testing_set);
disp(['Percent Correct ' num2str(percent_correct)])