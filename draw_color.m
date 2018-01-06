image_name = p1; % change image name according to your needs
pixel_numbers = size(before_hair, 1);
color = [0 0 0]; % change color of the hair according to your needs

rows = size(image_name, 1);
cols = size(image_name, 2);
resized = imresize(image_name, [50 50]);
r = zeros(pixel_numbers, 3);
r(:,1) = before_hair(:,2);
r(:,2) = before_hair(:,1);
r(:,3) = 1;
resized=insertShape(resized,'FilledCircle',r,'Color',color);
resize_back = imresize(resized, [rows cols]);
imshow(resize_back)

top_x = round(min(after_hair(:,1)));
bottom_x = round(max(after_hair(:,1)));
left_y = round(min(after_hair(:,2)));
right_y = round(max(after_hair(:,2)));

result = [];
deb = [];
mask = zeros(size(image_name,1),size(image_name,2));
for i=top_x:bottom_x
    for j=left_y:right_y
        pixel = [resize_back(i,j,1),resize_back(i,j,2),resize_back(i,j,3)];
        deb = [deb; pixel];
        if resize_back(i,j,2) <= 70 && resize_back(i,j,3) <= 70 && resize_back(i,j,1) <= 70
            result = [result; i j];
            mask(i,j) = 1;
        end
    end
end

for i=1:size(result,1)
    image_name(result(i,1),result(i,2),:) = color;
end
imshow(image_name)
save mask.mat mask