function [] = Show_images_for_10_class(data, weights)

recover_data = Neural_Network(data, weights);

%K = [1, 2, 3, 4, 5, 8, 9, 19, 22, 62];
    %7  2  1  0  4  9  5   3   6   8

K = [4, 3, 2, 19, 5, 9, 22, 1, 62, 8];
    
[~, lenght] = size(K);

for i = 1:lenght
    
    figure
    
    suptitle(['Images for Number ', num2str(i-1)]);
    
    s(1) = subplot(1, 2, 1);
    Show_images(data(K(i), :));
    
    s(2) = subplot(1, 2, 2);
    Show_images(recover_data(K(i), :));
    
    title(s(1), 'Original Image');
    title(s(2), 'Recover Image');

end
end