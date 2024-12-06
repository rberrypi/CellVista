A=single(imread('C:\tests\four_frames_async\f0_t0_i0_ch0_c0_r0_z0_m3.tif'));
B=single(imread('C:\tests\four_frames_async\f0_t0_i0_ch0_c0_r0_z0_m0.tif'));
C=single(imread('C:\tests\four_frames_async\f0_t0_i0_ch0_c0_r0_z0_m1.tif'));
D=single(imread('C:\tests\four_frames_async\f0_t0_i0_ch0_c0_r0_z0_m2.tif'));
At=2.5;
%
Gs=D-B;
Gc=A-C;
L=((Gc.^2 + Gs.^2).^(1/2));
t1=A+C;
t2=sqrt((4*A.*C)-(Gs).^2);
top=t1-t2;bottom=t1+t2;
beta1=sqrt(top./bottom);
L1=real(sum(beta1(:)))/sum(L(:));%Or select a region
top=L1.*At.*Gs;
bottom=1+(L1.*At).*(Gc);
phi=atan2(top,bottom);
%
imagesc(phi,[-0.7,1.4]);axis image;colormap(gray);
