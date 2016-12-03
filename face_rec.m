clear; clc; close all;

% read in dataset
croppedB01_set = imageSet('C:\Users\David\Downloads\CroppedYale\yaleB01');
for i = 1:croppedB01_set.Count
    croppedB01{i} = read(croppedB01_set,i);
    figure
    imagesc(croppedB01{i}); colormap gray;
end