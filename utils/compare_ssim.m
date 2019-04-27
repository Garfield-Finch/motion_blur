img_pth = '../../dataset/MPI-Sintel-complete/training/final/alley_1/';

img_nm = 'frame_0007.png';
img_nm_avg1 = 'frame_1006.png';
img_nm_avg2 = 'frame_1007.png';

img_ori = imread([img_pth, img_nm]);

% figure(1), imshow(img_ori), title('img\_ori');

img_avg = double(imread([img_pth, img_nm]));
img_avg = img_avg + double(imread([img_pth, img_nm_avg1]));
img_avg = img_avg + double(imread([img_pth, img_nm_avg2]));
img_avg = img_avg / 3;
img_avg = uint8(img_avg);

figure(2), imshow(img_avg), title('img\_avg');

img_avg_ld = imread([img_pth, 'frame_0007_avg.png']);
figure(16), imshow(img_avg_ld), title('load avg img');

img_dif = double(img_avg) - double(img_avg_ld);
figure(161), imshow(img_dif, []), title('difference img');
sum(sum(img_dif))

img_nm = 'frame_0007_ker.png';
img_ker = imread([img_pth, img_nm]);

figure(3), imshow(img_ker), title('img\_ker');

[ssimval,ssimmap] = ssim(img_avg, img_ker);

figure(4), imshow(ssimmap), title('ssimmap');

ssimval
