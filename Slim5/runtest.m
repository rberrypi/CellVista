function runtest
At=2.45;
cbar=[-0.3,0.7];
rd = @(str) single(imread(str));
A=rd('C:\Users\Misha\Desktop\fslim2\Slim5\testcell\test3.tif');
B=rd('C:\Users\Misha\Desktop\fslim2\Slim5\testcell\test0.tif');
C=rd('C:\Users\Misha\Desktop\fslim2\Slim5\testcell\test1.tif');
D=rd('C:\Users\Misha\Desktop\fslim2\Slim5\testcell\test2.tif');
imagesc(SLIM_PHASE(A,B,C,D,At),cbar);colormap(jet);axis image;
end
function phi=SLIM_PHASE(A,B,C,D,At)
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
end