clc; clear all;
two_tangels = zeros(2064,2048,'uint32');
[m,n]=size(two_tangels);
[XX,YY] = meshgrid(1:n,1:m);
ZZ=sign(mod(XX,15)).*sign(mod(YY,14));
fixed=int32(ZZ);
se=ones(3);
fixed=imerode(fixed,se);
writetif(fixed,'ZZ.tif');
imagesc(fixed);axis image;
% for r=1:m
%     for c=1:n
%         if mod(c
%     end
% end
% fixed=int32(two_tangels);
% writetif(fixed,'prepared.tif');
% took = 0;