clc; clear all;
%
two_tangels = zeros(2048,2048,'uint32');
[m,n]=size(two_tangels);
[XX,YY] = meshgrid(1:n,1:m);
ZZ=sign(mod(XX,15)).*sign(mod(YY,14));
fixed=int32(ZZ);
se=ones(3);
fixed=imerode(fixed,se);
writetif(fixed,'input_test_0.tif');
%
fixed(512:600,900:1239)=1;
writetif(fixed,'input_test_1.tif');
%
img=imread('circbw.tif');
img=int32(imresize(img,[512,512]));
writetif(img,'input_test_2.tif');
%
img=imread('trees.tif');
img=int32(imresize(img,[512,512])>100);
writetif(img,'input_test_3.tif');
%
img=imread('coins.png');
img=uint32(imresize(img>80,[1024,1024]));
writetif(img,'input_test_4.tif');

