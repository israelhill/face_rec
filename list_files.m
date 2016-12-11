function L = list_files(path)
    L = dir(path);
    L = L(3 : length(L));
    L = struct2cell(L);
    L = L (1, :);
end