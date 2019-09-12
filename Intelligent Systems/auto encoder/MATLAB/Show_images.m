function [] = Show_images(data)

image_data = zeros(28, 28); % for using imshow we need 28*28 matirx

for i = 1:28
    
    image_data(i,:) = data(1, (i-1)*28+1 : i*28);
    
end

imshow(image_data);

end