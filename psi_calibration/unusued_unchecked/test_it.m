basedir='D:\Mikhail\CellEmergence\Part12\final_scan2';
fullpath = @(ch,t) fullfile(basedir,sprintf('f0_t%d_i0_ch%d_c8_r7_z0_m0.tif',ch,t));
for t=[0:50]
subplot(2,1,1);
imagesc(imread(fullpath(t,0)));axis image;
subplot(2,1,2);
imagesc(imread(fullpath(t,1)));axis image;
drawnow;
pause(0.1);
end