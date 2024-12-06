%%Slices
clc;
p=1;
A=single(imread(strcat('Ae.tif'))); % 270.bmp
B=single(imread(strcat('Be.tif'))); %   0.bmp
C=single(imread(strcat('Ce.tif'))); %  90.bmp
D=single(imread(strcat('De.tif'))); % 180.bmp  
[H,W] = size(A);
%% del_phi
del_phi=atan2(D-B,A-C);
del_phiD=imread('del_phi.tif');
p1=del_phi(p,p);
p2=del_phiD(p,p);
diffv=sum(abs(del_phi(:))-abs(del_phiD(:)));
fprintf(1,'Del_Phi %f %f %f\n',diffv,p1,p2);
%% L
L=((A-C)+(D-B))./(sin(del_phi)+cos(del_phi))/4; %E0*E1 (sqrt(Io.IR))
LD=imread('L.tif');
p1=L(p,p);
p2=LD(p,p);
L(~isfinite(L))=0;
LD(~isfinite(LD))=0;
diffv=(mean(abs(L(:))-abs(LD(:))));
fprintf(1,'L %f %f %f\n',diffv,p1,p2);
%% Beta
g1=(A+C)/2; %E0^2+E1^2=s (Io+IR)
g2=L.*L; %E0^2*E1^2=p (Io.IR)
x1=g1/2-sqrt(g1.*g1-4*g2)/2;
x2=g1/2+sqrt(g1.*g1-4*g2)/2; %solutions (x1= E0^2 and x2= E1^2)
beta1=sqrt(x1./x2);
beta1D=imread('beta.tif');
beta1(~isfinite(beta1))=0;
beta1D(~isfinite(beta1D))=0;
p1=beta1(p,p);
p2=beta1D(p,p);
diffv=sum(abs(beta1(:))-abs(beta1D(:)));
fprintf(1,'Beta %f %f %f\n',diffv,p1,p2);
%% 
%average over whole frame
Bav=sum(beta1(:));
fprintf(1,'Bav %f\n',Bav);
Lav=sum(L(:));
fprintf(1,'Lav %f\n',Lav);
L1=real(Bav/Lav) ; %<(E0/E1)>/<(E0.E1)>=<1/E1^2>)
fprintf(1,'B/L %f\n',L1);
%%.0019
LL=L1*L; %<1/E1^2>.E0.E1=E0/E1
beta=LL*2.45; %real beta, 2.45 for 40X and 4 for 10X for fSLIM
phi=atan2(beta.*sin(del_phi),1+beta.*cos(del_phi));
phiD=imread('phi.tif');
p1=phi(p,p);
p2=phiD(p,p);
diffv=sum(abs(phi(:))-abs(phiD(:)));
fprintf(1,'Phi %f %f %f\n',diffv,p1,p2);
%%
%figure,
%crange=[-.3,0.7];
subplot(1,2,1);
imagesc(phi,crange);
subplot(1,2,2);
imagesc(phiD,crange);