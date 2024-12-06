clc; clear all;

range=[0,1];
subplot(3,1,1);
img_A_cpu=imread('C:\fslim2\x64\Debug\CPU_Version.tif');
imagesc(img_A_cpu,range);axis image;title('CPU');
subplot(3,1,2);
img_A_gpu=single(imread('C:\fslim2\x64\Debug\polarizer_frame_3_16.tif'));
imagesc(img_A_gpu,range);axis image;title('GPU');
subplot(3,1,3);
imagesc(img_A_cpu-img_A_gpu,[0,1]);axis image;

% img_one=imread('test_0.tif');
% img_A=imread('C:\fslim2\x64\Debug\polarizer_frame_0.tif');
% img_B=imread('C:\fslim2\x64\Debug\polarizer_frame_1.tif');
% img_C=imread('C:\fslim2\x64\Debug\polarizer_frame_2.tif');
% img_D=imread('C:\fslim2\x64\Debug\polarizer_frame_3.tif');
% imagesc(img_A);axis image;
%
% subplot(2,2,1);
% imagesc(img_A);axis image;
% subplot(2,2,2);
% imagesc(img_B);axis image;
% subplot(2,2,3);
% imagesc(img_C);axis image;
% subplot(2,2,4);
% imagesc(img_D);axis image;