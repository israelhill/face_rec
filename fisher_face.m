%%
path = 'images/CroppedYale_fixed';
folder = list_files(path);
% initialize the empty return values
X =[];
y =[];
width = 0;
height = 0;
% start counting with class index 1
classIdx = 1;
% for each file ...
for i = 1: length (folder)
    i
    subject = folder{i};
    % get files in this subdir
    images = list_files([path, filesep, subject]) ;
    
    % ignore a file or empty folder
    if(isempty(images))
        continue;
    end
    
    % for each image
    for j = 1: length(images)
        % ... get the absolute path
        filename = [path, filesep, subject, filesep, images{j}];
        ds_store = [path, filesep, subject, filesep, '.DS_Store'];
        if(strcmp(filename, ds_store) == 1)
            continue;
        end
        % ... read the image
        T = single(imread(filename));
        T = imresize(T, 0.5);
        % ... get the image information
        [height, width] = size(T);
        % ... reshape into a row vector and append to data matrix
        X = [X; reshape(T,1,width*height)];
        % ... append the corresponding class to the class vector
        y = [y, classIdx];
    end
    % ... increase the class index
    classIdx = classIdx + 1;
end % ... for - each folder.

% number of samples
n = size(X,2);
% get a random index
randomIdx = uint32(rand()*n);
% split data
% into training set
Xtrain = X(:, [1:(randomIdx-1), (randomIdx+1):n]); 
ytrain = y([1:(randomIdx-1), (randomIdx+1):n]);
% into test set
Xtest = X(:,randomIdx);
ytest = y(randomIdx);

%%
[rows, cols] = size(Xtrain);
classes = unique(y);
num_classes = length(classes);

% PCA
c = rows - num_classes; 
data_mean = mean(Xtrain);
Xm = Xtrain - repmat(data_mean, rows, 1); 
if(rows > cols)
    k = Xm'*Xm;
%     [W_pca, s1, v1] = svd(k, 'econ');
    [W_pca, d] = eig(k);
    [d, index] = sort(diag(d), 'descend');
    W_pca = W_pca(:, index);
    W_pca = W_pca(:,1:c);
else
    k = Xm*Xm';
%     [W_pca, s1, v1] = svd(k, 'econ');
    [W_pca, d] = eig(k);
    W_pca = Xm'*W_pca;
    for i = 1 : rows
        W_pca(:,i) = W_pca(:,i)/norm(W_pca(:,i));
    end
    [d, index] = sort(diag(d), 'descend');
    W_pca = W_pca(:, index);
    W_pca = W_pca(:,1:c);
end
fld_projection = (Xtrain - repmat(data_mean, size(Xtrain, 1), 1))*W_pca;


%%
% LDA
[rows, cols] = size(fld_projection);
scatter_within = single(zeros(cols, cols));
scatter_between = single(zeros(cols, cols));
fld_mean = mean(fld_projection);

for i = 1 : num_classes
    current_class = fld_projection(find(y == classes(i)),:);
    num_rows = size(current_class, 1);
    class_mean = mean(current_class);
    current_class = current_class - repmat(class_mean, num_rows, 1);
    if i == 1
        scatter_within = current_class' * current_class;
        scatter_between = num_rows*(class_mean - fld_mean)'...
            *(class_mean - fld_mean);
    else
        scatter_within = scatter_within + current_class' * current_class;
        scatter_between = scatter_between + num_rows ...
            * (class_mean - fld_mean)'*(class_mean - fld_mean);
    end
end
% solve for the eigenvectors
% [W_fld,S,V] = svd(inv(scatter_within)*scatter_between);
[W_fld, d] = eig(scatter_between, scatter_within);
[d, index] = sort(diag(d), 'descend');
W_fld = W_fld(:, index);
W_fld = W_fld(:,1:num_classes-1);

W = W_pca*W_fld;
%%

figure; hold on; 
title(sprintf('Fisherfaces')); 
for i=1:min(16, num_classes-1)
    subplot(4,4,i);
   
    x = W(:,i);
    minX = min(x(:));
    maxX = max(x(:));
    x = x - minX;
    x = x ./ (maxX - minX);
    x = x .* (255);
    x = uint8(x);
    
    fisherface = reshape(x, height, width);
    imagesc(fisherface);
    colormap(jet(256));
    title(sprintf('Fisherface #%i', i));
end

steps = 1:min(16,num_classes-1);
Q = Xtrain(1,:); % first image to reconstruct
figure; hold on;
title(sprintf('Fisherfaces Reconstruction'));
for i=1:min(16, length(steps))
    subplot(4,4,i);
    numEv = steps(i);
    replication = repmat(data_mean, size(Q, 1), 1);
    projection = (Xtrain(1,:) - replication)*W(:,numEv);    
    reconstructed_img = projection * W(:,numEv)' + repmat(data_mean, size(projection, 1), 1);
    
    x = reconstructed_img;
    minX = min(x(:));
    maxX = max(x(:));
    x = x - minX;
    x = x ./ (maxX - minX);
    x = x .* (255);
    x = uint8(x);
    
    reconstructed_final = reshape(x, height, width);
    imagesc(reconstructed_final);
    title(sprintf('Fisherface #%i', numEv));
end










