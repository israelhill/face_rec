clc; clear all;
folderpath = 'images\\CroppedYale_fixed\\yaleB%02d\\*.pgm';
for i = 1:39
   srcfiles = dir(sprintf(folderpath,i));
   for j = 1:length(srcfiles)
       file = strcat(sprintf('images\\CroppedYale_fixed\\yaleB%02d\\',i),srcfiles(j).name);
       cropped_set{i,j} = imread(file);
   end
   face_divides(i) = length(srcfiles);
end

index = 1;
for i = 1:39
    for j = 1:64
        if ~isempty(cropped_set{i,j})
            fixed(:,:,index) = cropped_set{i,j};
            index = index+1;
        end
    end
end

% randomly split into half for training and testing
% get index of training set
% divide index into faces
% correspond new training set index with complete index
% say which face it think it corresponds to
% do difference on all the images from that face to confirm