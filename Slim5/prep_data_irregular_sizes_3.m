clc; clear all;
size_pears = [
    2048,2048;
    1776,1776;
    1776,1760;
    2064,2048;
    2048,2048;
    2048,2064;
    1920,1080;
    1392,1040;
    1500,256;
    768,768;
    528,512;
    240,256;
    144,128;
    2048,2048;
    1920,1080;
    1392,1040;
    512,512;
    128,128;
    ];

for idx = 1
    n=size_pears(idx,1);
    m=size_pears(idx,2);
    two_tangels = zeros(m,n,'uint32');
    [m,n]=size(two_tangels);
    [XX,YY] = meshgrid(1:n,1:m);
    ZZ=sign(mod(XX,15)).*sign(mod(YY,14));
    fixed=int32(ZZ);
    se=ones(3);
    fixed=imerode(fixed,se);
    fname = sprintf('input_different_size_test_%d.tif',idx-1);
    disp(fname);
    writetif(fixed,fname);
    fname_map = sprintf('input_different_size_test_%d_phase_map.tif',idx-1);
    [XX,YY] = meshgrid(1:size(fixed,2),1:size(fixed,1));
    fixed_phase = single(fixed).*sqrt((XX.^2)+(YY.^2));
    writetif(fixed_phase,fname_map);
end