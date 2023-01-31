function display_images(fname, imdir)
% Example function for using read_att_data.m, and the dataset
% Input: <fname> name of dataset to use
%        <imdir> directory where images are stored

[img_names img_classes bboxes attributes] = read_att_data(fname);

for i = 1:length(img_names)
   im = imread(fullfile(imdir, img_names{i}));

   bbox = bboxes(i,:);
   subplot(1,2,1)
   imagesc(im);
   subplot(1,2,2)
   imagesc(im(bbox(2):bbox(4), bbox(1):bbox(3), :))
   title(img_classes{i})
   pause
end
