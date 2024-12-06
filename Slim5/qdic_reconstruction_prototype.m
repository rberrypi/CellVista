function qdic_reconstruction_prototype
dbstop if error;
basedir='D:\Mikhail\BabyCows\re_experiment';
functor = 'C:\Users\Datahoarder\Desktop\qdic_red_munged2_munged2_functor.mat';
outdir=strcat(basedir, '_output_matlabput');
mkdir(outdir);
%f67_t16_i59_ch1_c0_r0_z0_m3.tif
fname=@(f,t,i,ch,c,r,z,m) sprintf('%s//f%d_t%d_i%d_ch%d_c%d_r%d_z%d_m%s.tif',basedir,f,t,i,ch,c,r,z,m); % FOV, TIME, Channel, Frame Number, PAT
fout=@(f,t,i,ch,c,r,z,m) sprintf('%s//f%d_t%d_i%d_ch%d_c%d_r%d_z%d_m%s.tif',outdir,f,t,i,ch,c,r,z,m); % FOV, TIME, Channel, Frame Number, PAT
colormap(gray);
ff=0;tt=0:0;
chch=3;cc=0;rr=0;zz=60;ii=0;
band=[0,150];
[m,n]=size(imread(fname(ff(1),tt(1),ii(1),chch(1),cc(1),rr(1),zz(1),'0')));
dim=2^nextpow2(max([m,n]));
qdic=load(functor,'qdic');qdic=qdic.qdic;
%delete(gcp('nocreate'));
%parpool('local',3); %Else uses too much ram
for f=ff
    for t=tt
        if (exist(feval(fout,f,t,0,0,0,0,0,'int'),'file')==2)
            disp('Skipping');
            continue
        end
        for i=ii
            %
            for ch=chch
                for c=cc
                    for r=rr
                        for z=zz
                            fnameA=feval(fname,f,t,i,ch,c,r,z,'0');
                            fnameB=feval(fname,f,t,i,ch,c,r,z,'1');
                            fnameC=feval(fname,f,t,i,ch,c,r,z,'2');
                            fnameD=feval(fname,f,t,i,ch,c,r,z,'3');
                            if (~((exist(fnameA, 'file') == 2) && (exist(fnameB, 'file') == 2) && ...
                                    (exist(fnameC, 'file') == 2) && (exist(fnameD, 'file') == 2)))
                                continue;
                            end
                            tic;
                            A=single(imread(fnameA));
                            B=single(imread(fnameB));
                            C=single(imread(fnameC));
                            D=single(imread(fnameD));
                            [qdic,~,~,~]=QDIC(A,B,C,D);
                            imagesc(qdic);axis image;colormap(jet);
                            qdic = fourier_filter(qdic,filter,2);
                            writetif(qdic,feval(fout,f,t,i,ch,c,r,z,'qdic'));
                            toc;
                        end
                    end
                end
            end
        end
        %     writetif(dic_s,feval(fout,f,t,0,0,0,0,0,'dic'));
        %     writetif(qdic_s,feval(fout,f,t,0,0,0,0,0,'qdic'));
        %     writetif(int_s,feval(fout,f,t,0,0,0,0,0,'int'));
    end
    %
    %
    
end
end


function img = fourier_filter(input,filt,kind)
[m,n]=size(input);
dim=2^nextpow2(max([m,n]));
pad=padarray(input,[(dim-m)/2 (dim-n)/2],'symmetric','both');
img_filt=(ifft2(fft2(pad).*filt));
if (kind==1)
    img_filt=imag(img_filt);
end
if (kind==2)
    img_filt=real(img_filt);
end
img=imcrop(img_filt,[1+(dim-n)/2,1+(dim-m)/2,n-1,m-1]);
end

function [qdic_filtered] = QDIC_filtered(A,B,C,D,filter)
%Mikhail 6/23
%D=1+cos(t-1pi/2)
%A=1+cos(t+0pi/2)%SLIM convention
%B=1+cos(t+2pi/2)
%C=1+cos(t+3pi/2)
a2=(B-D)/2;
a1=(A-C)/2;
qdic=atan2(a2,a1);
qdic_filtered=fourier_filter(qdic,filter,2);
end

function crange=getCrange(img)
f=figure(2);
imagesc(img);
crange=caxis;
r=0.3*(crange(2)-crange(1));
m=0.5*(crange(2)+crange(1));
crange=[-r,r]+m;
close(f);
end

function img=scaleToRange(img,crangein,crangeout,ceiling)
if nargin<4
    ceiling=true;
end
min=crangein(1);max=crangein(2);
a=crangeout(1);b=crangeout(2);
if (ceiling)
    img(img>max)=max;
    img(img<min)=min;
end
img=(a+(b-a)*(img-min)/(max-min));
end

function matlabSaveImages(stack,name,s)
writerObj = VideoWriter(name);
writerObj.FrameRate = 10;
open(writerObj);
%maxV=max(stack(:));
%minV=min(stack(:));
crange=getCrange(stack(:,:,ceil(end/2)));
stack=scaleToRange(stack,crange,[0,1]);
stack(stack<0)=0;
stack(stack>1)=1;
[~,~,p]=size(stack);
for i=1:p
    ptr=stack(:,:,i);
    if (s~=1)
        ptr=imresize(ptr,s);
    end
    writeVideo(writerObj,squeeze(ptr));
end
close(writerObj);
end

function safeMakeDir(fullpath)
if ~exist(fullpath, 'dir')
    mkdir(fullpath);
end
end