%Edges maybe
clc; clear all;
X=imread('coins.png');
X=single(X>150);
subplot(2,2,1);
imagesc(X);axis image;
subplot(2,2,2);
imagesc(conv2(X,[1,1,1;1,1,1;1,1,1]));axis image;
subplot(2,2,3);
imagesc(conv2(X,[0,1,0;1,-4,1;0,1,0])>0);axis image;
subplot(2,2,4);
img_img = single(conv2(X,[1,1,1;1,1,1;1,1,1])>0)-single(conv2(X,[0,1,0;1,-4,1;0,1,0])>0);
imagesc(img_img);axis image;
%imagesc(single(conv2(X,[1,1,1;1,1,1;1,1,1])>0)-single(conv2(X,[0,1,0;1,-4,1;0,1,0])>0));axis image;