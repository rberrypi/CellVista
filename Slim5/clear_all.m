clc; clear all;
% A=imread('Gold_1_label_test.tif');
% B=imread('Test_1_label_test.tif');
% A=imread('Gold_call_0_after_flatten.tif');
% B=imread('Test_call_0_after_flatten.tif');
A=imread('File_1_Test_step_3_flatten_tree.tif');
B=imread('File_3_Test_step_3_flatten_tree.tif');
crop = @(x) x((end-64):end,1:32);
%crop = @(x) x(1:32,1:32);
subplot(3,1,1);
imagesc(crop(A));axis image;
subplot(3,1,2);
imagesc(crop(B));axis image;
subplot(3,1,3);
imagesc(crop(A)-crop(B));axis image;
%0,15 is wrong
colormap(jet);