function [img_names, img_classes, bbox, attributes] = read_att_data(fname)
% [img_names, img_classes, bbox, attributes] = read_att_data(fname)
% This function reads the attribute data for the given file name
%  INPUT: <fname> the file name of the dataset to load
%           apascal_train.txt, apascal_test.txt, ayahoo_test.txt
%  Output:  <img_names> cell list  of the files corresponding to each object
%           <img_classes> cell list of the class name for each object
%           <bbox> n x 4 matrix of bounding boxes - [xmin ymin xmax ymax] 
%              Each row corresponds to an object
%           <attributes> n x 64 matrix of attributes (one row per object)
%              Attribute names can be found in attribute_names.txt
%
N_ATTS = 64;

fd = fopen(fname);
res = textscan(fd, ['%s %s' repmat(' %f',1,N_ATTS + 4)],'CollectOutput',1);
fclose(fd);

img_names = res{1}(:,1);
img_classes = res{1}(:,2);
bbox = res{2}(:, 1:4);
attributes = res{2}(:,5:end);
