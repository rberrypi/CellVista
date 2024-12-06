clc; clear all;


fancy=imread('demosaiced.tif');
fancy=fancy/(8*(2^16));
subplot(1,2,1);
imshow(fancy);axis image;

subplot(1,2,2);
img=imread('test_0.tif');
RGB = double(demosaic(img,'RGGB'))/(2^16);
imshow(RGB);