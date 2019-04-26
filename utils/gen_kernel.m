%% init parameters
flo_pth = '../../dataset/MPI-Sintel-complete/training/flow/alley_1/';
img_pth = '../../dataset/MPI-Sintel-complete/training/final/alley_1/';

img_epoch = 48;
img_nm = [img_pth, gen_nm(img_epoch, 0)];
disp(['image name: ', img_nm]);
img_i = imread(img_nm);

flo_nm = [flo_pth, gen_nm(img_epoch - 1, 2)];
flo_pre = readFlowFile(flo_nm);
flo_nm = [flo_pth, gen_nm(img_epoch, 2)];
flo_i = readFlowFile(flo_nm);

[h, w, ~] = size(flo_i);
pre = zeros(h, w, 2);
aft = zeros(h, w, 2);

% calculate the two relationship map
for i = 1:h
    for j = 1:w
        dx = flo_pre(i, j, 1);
        dy = flo_pre(i, j, 2);
        [pre(i, j, 1), pre(i, j, 2)] = gen_cor_flo(i, j, dx, dy, h, w);
        
        dx = flo_i(i, j, 1);
        dy = flo_i(i, j, 2);
        [x, y] = gen_cor_flo(i, j, dx, dy, h, w);
        aft(x, y, 1) = x;
        aft(x, y, 2) = y;
    end
end

img_o = uint8(zeros(h, w, 3));
% 
for i = 1:h
    for j = 1:w
        [r, g, b] = cal_pix(i, j, pre, img_i, aft);
        img_o(i, j, 1) = r;
        img_o(i, j, 2) = g;
        img_o(i, j, 3) = b;
    end
end

figure(1), imshow(img_i), title('img\_in');
figure(2), imshow(img_o), title('img\_gen');

% img_mask = gen_mask(img_o);
% figure(16), imshow(img_mask), title('mask');
% img_o = inpaintCoherent(img_o, img_mask);
% figure(3), imshow(img_o), title('img\_inpaint');

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

function [x, y] = gen_cor_flo(imgx, imgy, dx, dy, h, w)
    x = round(imgx + dy);
    y = round(imgy + dx);
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

function [r, g, b] = cal_pix(i, j, pre, img_i, aft)
    total = 1;
    before = 0;
    after = 0;
    [h, w, ~] = size(img_i);
    
    x = pre(i, j, 1);
    y = pre(i, j, 2);
    if x>1 && x<h && y>1 && y<w
        before = double(img_i(x, y, :));
        total = total + 1;
    end
    
    x = aft(i, j, 1);
    y = aft(i, j, 2);
    if x>1 && x<h && y>1 && y<w
        after = double(img_i(x, y, :));
        total = total + 1;
    end
    
    now = double(img_i(i, j, :));
    intensity = uint8((before + now + after) / total);
    
    r = uint8(intensity(1, 1, 1));
    g = uint8(intensity(1, 1, 2));
    b = uint8(intensity(1, 1, 3));
    
%     if total < 3
%         r = 0;
%         g = 0;
%         b = 0;
%     end
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

