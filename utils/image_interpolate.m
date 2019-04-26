%% init parameters
flo_pth = '../../dataset/MPI-Sintel-complete/training/flow/alley_1/';
img_pth = '../../dataset/MPI-Sintel-complete/training/final/alley_1/';

%%
for img_epoch = 1: 49
    img_nm = [img_pth, gen_nm(img_epoch, 0)];
    img_p = imread(img_nm);
%     figure(1), imshow(img_p), title('img\_pre');

%     img_nm = [img_pth, gen_nm(img_epoch + 1, 0)];
%     img_a = imread(img_nm);
%     figure(2), imshow(img_a), title('img\_after');

    [h, w, ~] = size(img_p);
    img_o = uint8(zeros(h, w, 3));
    flo_nm = [flo_pth, gen_nm(img_epoch, 2)];
    flo_i = readFlowFile(flo_nm);
    for i = 1:h
        for j = 1:w
            x = flo_i(i, j, 1);
            y = flo_i(i, j, 2);
            [outx, outy] = gen_cor(i, j, x, y, h, w);
            img_o(outx, outy, :) = img_p(i, j, :);
        end
    end

%     figure(3), imshow(img_o), title('img\_gen');

    % inpaint
    img_mask = gen_mask(img_o);
    img_o = inpaintCoherent(img_o, img_mask);
    figure(4), imshow(img_o), title('img\_inpaint');
    
    disp(['epoch ', num2str(img_epoch), ' accomplished']);
    img_nm = [img_pth, gen_nm(img_epoch, 1)];
%     imwrite(img_o, img_nm);
end

% % compare output and target
% img_comp = img_a - img_o;
% figure(4), imshow(img_comp), title('img\_comparison');

%% Utils
function imgnm = gen_nm(num, key)
    % key == 0, input image name
    % key == 1, output interpolation image name
    % key == 2, .flo file name
    if key == 0
        s = num2str(num);
        len = length(s);
        for i = 1:4-len
            s = ['0', s];
        end
        imgnm = ['frame_', s, '.png'];
    elseif key == 1
        s = num2str(num);
        len = length(s);
        for i = 1:4-len-1
            s = ['0', s];
        end
        imgnm = ['frame_1', s, '.png'];
    elseif key == 2
        s = num2str(num);
        len = length(s);
        for i = 1:4-len
            s = ['0', s];
        end
        imgnm = ['frame_', s, '.flo'];
    else
        disp('KEY ERROR !!!')
    end
end

function [x, y] = gen_cor(imgx, imgy, dx, dy, h, w)
    x = round(imgx + dy / 2);
    y = round(imgy + dx / 2);
    if x < 1
        x = 1;
    end
    if x > h
        x = h;
    end
    if y < 1
        y = 1;
    end
    if y > w
        y = w;
    end
end

function mask = gen_mask(img)
    [h, w, ~] = size(img);
    mask = zeros(h, w);
    for i = 1:h
        for j = 1:w
            mask(i,j) = sum(img(i,j,:));
        end
    end
    mask = logical(mask);
    mask = ~ mask;
end