function segmentation_phantom
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
    fixed(1:100,1:100)=0;
    se = ones(3,3);
    fixed=imerode(fixed,se);
    fixed=imerode(fixed,se);
    fixed=imerode(fixed,se);
    current_folder = pwd;
    prefix='\test_images\segmentation_phantom\';
    D=uint16(1000*(1+0.9*fixed));
    A=uint16(1000*(1+0.1*fixed));
    B=uint16(1000*(1-0.3*fixed));
    C=uint16(1000*(1-0.9*fixed));
    %generates a 0.7 radian shift image...
    %in Code 0->B, 1->C, 2->D, 3->A
    writetif(A,sprintf('%s%stest_%d.tif',current_folder,prefix,3));
    writetif(B,sprintf('%s%stest_%d.tif',current_folder,prefix,0));
    writetif(C,sprintf('%s%stest_%d.tif',current_folder,prefix,1));
    writetif(D,sprintf('%s%stest_%d.tif',current_folder,prefix,2));
    %
    phi=slim(single(A),single(B),single(C),single(D));
    figure;imagesc(phi);axis image;
end
end

function phi = slim(A,B,C,D)
A=single(A);
B=single(B);
C=single(C);
D=single(D);
At=2.5;
Gs=D-B;
Gc=A-C;
figure;imagesc(atan2(Gs,Gc));axis image;
L=((Gc.^2 + Gs.^2).^(1/2));
t1=A+C;
t2=sqrt((4*A.*C)-(Gs).^2);
top=t1-t2;bottom=t1+t2;
beta1=sqrt(top./bottom);
L1=real(sum(beta1(:)))/sum(L(:));%Or select a region
top=L1.*At.*Gs;
bottom=1+(L1.*At).*(Gc);
phi=atan2(top,bottom);
end