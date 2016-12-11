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
        T = imresize(T, 0.1);
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
%%
% LDA
[rows, cols] = size(X);
classes = unique(y);
num_classes = length(classes);

scatter_within = single(zeros(cols, cols));
scatter_between = single(zeros(cols, cols));
data_mean = mean(X);

for i = 1 : num_classes
    current_class = X(find(y == classes(i)),:);
    num_rows = size(current_class, 1);
    class_mean = mean(current_class);
    current_class = current_class - repmat(class_mean, num_rows, 1);
    if i == 1
        scatter_within = current_class' * current_class;
        scatter_between = num_rows*(class_mean - data_mean)'...
            *(class_mean - data_mean);
    else
        scatter_within = scatter_within + current_class' * current_class;
        scatter_between = scatter_between + num_rows ...
            * (class_mean - data_mean)'*(class_mean - data_mean);
    end
end
% solve for the eigenvectors
[U,S,V] = svd(inv(scatter_within)*scatter_between);
U = U(:,1:num_classes-1);















