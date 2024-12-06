clc; clear all;
% 2448 x 2048
[xx,yy]=meshgrid([1:512],[1:612]);
%
% img_to_read='C:\fslim2\x64\Debug\polarizer_frame_0.tif';
% x_lower = xx(1:2:end,1:2:end);
% y_lower = yy(1:2:end,1:2:end);
% z1 = griddata(x_lower,y_lower,x_lower,xx,yy,'cubic');
% z2=imread(img_to_read);
% imagesc(z1);axis image;
%
XX=uint16(yy)+10;
XX(2:2:end,2:2:end)=XX(2:2:end,2:2:end)+15;
XX(2:2:end,1:2:end)=XX(2:2:end,1:2:end)+20;
XX(1:2:end,2:2:end)=15;
writetif(XX,'test_0.tif');
writetif(XX,'test_1.tif');
writetif(XX,'test_2.tif');
writetif(XX,'test_3.tif');