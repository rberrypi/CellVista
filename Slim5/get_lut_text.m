clc; clear all;
lut=flip(squeeze(imread('C:\Users\qli\Desktop\luts\hsatl2.tif')));
lut=lut(4:end,:);%hack
lut=imresize(lut,[256,3]);
lut=flip(lut);
fileID = fopen('test.h','w');
fprintf(fileID,'{');
for i=1:length(lut)
    fprintf(fileID,'%d, %d, %d,', lut(i,1),lut(i,2),lut(i,3));
end
fprintf(fileID,'}');
fclose(fileID);
