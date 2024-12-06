function FullCurve_color
%6/30/2017
dbstop if error;
%basedir='E:\Color_DIC\h&e_two';
basedir='C:\Users\QLI\Desktop\h&e_tma2';
%basedir='E:\Color_DIC\rbcs_red';
name = @(x) sprintf('%s\\f0_t0_i0_ch0_c0_r0_z0_m%d.tif',basedir,x);
color_comp=[600,525,460];
files=300:510;
[phase,~,amp,rect]=PhaseExtraction(basedir,name,files);
for c = 1:3
    [shifts(c,:),~,~]=PhaseSearching(phase(c,:),amp(c,:));
end
show_gallery(phase,amp,rect,name,shifts,files,color_comp);
end

function slice=getSlice(img,rect)
rect=ceil(rect);
crop =@(img) (img(rect(2):(rect(2)+rect(4)),rect(1):(rect(1)+rect(3))));
slice=crop(img);
end

function show_gallery(phase,amp,rect,name,shifts,files,color_comp)
labels={'A','B','C','D'};
colors={'r','g','b'};
shift_idx=shifts+min(files)-1;
    function label_a_plot(proxy)
        for c=1:3
            set(h(c),'Color',colors{c},'LineStyle','-')
            for shift=1:4
                %x_pos=shifts(c,shift)+min(files)-1;
                label=strcat('\leftarrow ',labels{shift});
                text(shift_idx(c,shift),proxy(c,shifts(c,shift)),label,'color',colors{c});
            end
        end
    end

subplot(2,3,1);
h=plot(files,amp(1,:),files,amp(2,:),files,amp(3,:));
xlim([min(files),max(files)]);
label_a_plot(amp);
title('amp');

subplot(2,3,4);
h=plot(files,phase(1,:),files,phase(2,:),files,phase(3,:));
xlim([min(files),max(files)]);
label_a_plot(phase);
title('phase');

comp=[];
subplot(2,3,2);
c=1;
A=single(imread(name(shift_idx(c,1))));
B=single(imread(name(shift_idx(c,2))));
C=single(imread(name(shift_idx(c,3))));
D=single(imread(name(shift_idx(c,4))));
qdic_red = QDICTWO(A(:,:,c),B(:,:,c),C(:,:,c),D(:,:,c));
qdic_red_crop = getSlice(qdic_red,rect);
red_title = sprintf('Red: %0.2f', mean(qdic_red_crop(:)));
imagesc(qdic_red);axis image;colormap(gray);
title(red_title);

subplot(2,3,3);
c=2;
A=single(imread(name(shift_idx(c,1))));
B=single(imread(name(shift_idx(c,2))));
C=single(imread(name(shift_idx(c,3))));
D=single(imread(name(shift_idx(c,4))));
qdic_green = QDICTWO(A(:,:,c),B(:,:,c),C(:,:,c),D(:,:,c));
green_title = sprintf('Green: %0.2f', mean(qdic_green(:)));
imagesc(qdic_green);axis image;colormap(gray);
title(green_title);

subplot(2,3,5);
c=3;
A=single(imread(name(shift_idx(c,1))));
B=single(imread(name(shift_idx(c,2))));
C=single(imread(name(shift_idx(c,3))));
D=single(imread(name(shift_idx(c,4))));
qdic_blue = QDICTWO(A(:,:,c),B(:,:,c),C(:,:,c),D(:,:,c));
blue_title = sprintf('Blue: %0.2f', mean(qdic_blue(:)));
imagesc(qdic_blue);axis image;colormap(gray);
title(blue_title);

subplot(2,3,6);
scale=0.5;
merged = cat(3,qdic_red,qdic_green,qdic_blue);
merged_fixed_16 = fix_qdic(merged,rect,color_comp);
merged_fixed_16=uint16(scale_range(merged_fixed_16,[-scale,scale],[0,65535]));
imagesc(merged_fixed_16);axis image;
writetif_color(merged_fixed_16,'merged.tif');
title('color_merger');
end

function qidc_out = fix_qdic(stack,rect,color_comp)
qidc_out=stack;
band=[0,60];
[m,n,~]=size(stack);
dim=2^nextpow2(max([m,n]));
filter = bandpass_mex_no_gpu(band,band,[dim,dim]);
filter(1,1)=0;
qidc_out(:,:,1) = fourier_filter(stack(:,:,1),filter,2);
qidc_out(:,:,2) = fourier_filter(stack(:,:,2),filter,2);
qidc_out(:,:,3) = fourier_filter(stack(:,:,3),filter,2);
%
qidc_out(:,:,1)=qidc_out(:,:,1)./(color_comp(1)/max(color_comp));
qidc_out(:,:,2)=qidc_out(:,:,2)./(color_comp(2)/max(color_comp));
qidc_out(:,:,3)=qidc_out(:,:,3)./(color_comp(3)/max(color_comp));
% stack_red_ref=getSlice(stack(:,:,1),rect);
% stack_green_ref=getSlice(stack(:,:,2),rect);
% stack_blue_ref=getSlice(stack(:,:,3),rect);
% red_munged = stack(:,:,1);
% green_munged = imhistmatch(stack(:,:,2),red_munged,256);
% blue_munged = imhistmatch(stack(:,:,3),red_munged,256);
% qidc_out = cat(3,red_munged,green_munged,blue_munged);
end

function qdic = QDICTWO(A,B,C,D)
a2=(B-D);%actually -a2
a1=(A-C);
qdic=atan2(a2,a1);
end

function A_out = scale_range(A,in_range,out_range)
%
%       (b-a)(x - min)
% f(x) = --------------  + a
%           max - min
%
A_out=A;
A_out(A_out>in_range(2))=in_range(2);
A_out(A_out<in_range(1))=in_range(1);
scale = (out_range(2)-out_range(1))/(in_range(2)-in_range(1));
A_out=scale*(A_out-in_range(1)+out_range(1));
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

function [phase,gamma,amp,rr]=PhaseExtraction(basedir,name,files)
h=figure;
img=imread(name(median(files)));
img=img(:,:,1);
imagesc(img);axis image;
rr=getrect;
rr=ceil(rr);
close(h);
amp=[];
for i=files
    img=imread(name(i),'PixelRegion',{[rr(2),rr(2)+rr(4)],[rr(1),rr(1)+rr(3)]});
    img=mean(mean(img,1),2);
    amp(:,end+1)=img;
end
raw=amp;
Ymin=min(raw,[],2);
Ymax=max(raw,[],2);
raw=bsxfun(@minus,raw,((Ymin+Ymax)/2));
raw=bsxfun(@times,raw,1./((Ymin+Ymax)/2));
phase=raw;
for c=1:size(raw,1)
    gamma=hilbert(raw(c,:));
    phase(c,:)=angle(gamma);
    phase(c,:)=unwrap(phase(c,:));
end
end

function [shifts,err,avgerror]=PhaseSearching(phase,amp)
[~,lct] = findpeaks(amp,'MinPeakProminence',4);
%lct=lct(end);
%[~,lct]=max(amp);
firstidx=@(XXX,start,v) find(XXX>(XXX(start)+v),1,'first');%first above
    function err=error3(XXX,A,B,C,Z)
        clamp = @(small,value,large) max([min([value,large]),small]);
        pZ=XXX(clamp(1,Z,length(XXX)));
        pA=XXX(clamp(1,A,length(XXX)));
        pB=XXX(clamp(1,B,length(XXX)));
        pC=XXX(clamp(1,C,length(XXX)));
        dists=[(pA-pZ-pi/2),(pB-pA-pi/2),(pC-pB-pi/2),...
            (pB-pZ-pi),(pC-pA-pi),(pC-pZ-3*pi/2)];
        err=sum(dists.^2);
    end
    function disp=avgStep(XXX,A,B,C,Z)
        pZ=XXX(Z);pA=XXX(A);
        pB=XXX(B);pC=XXX(C);
        dists=[pA-pZ,pB-pA,pC-pB];
        disp=mean(dists);
    end
err=inf;
shifts=[];
for startsearch=lct
    p0g=(firstidx(phase,startsearch,-1*pi/2)-1);
    p1g=(startsearch);
    p2g=(firstidx(phase,startsearch,1*pi/2)-1);
    p3g=(firstidx(phase,startsearch,2*pi/2)-1);
    if (isempty(p1g)||isempty(p2g)||isempty(p3g)||isempty(p0g))
        continue
    end
    %space=[-1,0,1];
    %space=[0];
    space=[-2,-1,0,1,2];
    for p0=p0g+space
        if (p0<1)
            continue
        end
        for p1=p1g+space
            for p2=p2g+space
                for p3=p3g+space
                    ernew=error3(phase,p1,p2,p3,p0);
                    if (ernew <err)
                        shifts=[p1,p2,p3,p0];
                        err=ernew;
                    end
                end
            end
        end
    end
    disp(err);
end
avgerror=avgStep(phase,shifts(1),shifts(2),shifts(3),shifts(4));
end

function [shifts,err]=MinStepErrorPhaseSearching(phase,amp,start,stop)
searchrange=phase(end)-phase(1);
if(searchrange<1.6*pi)
    plotCurves(amp,phase,[],start,stop);
    error('Range only contains %0.2f RAD, go buy a new SLM',searchrange);
end
for m1=1:length(phase)
    %disp(m1);
    for m2=m1:length(phase)
        for m3=m2:length(phase)
            for m4=m3:length(phase)
                p0=phase(m1);p1=phase(m2);
                p2=phase(m3);p3=phase(m4);
                dists=[(p1-p0-pi/2),(p2-p1-pi/2),(p3-p2-pi/2),...
                    (p2-p0-pi),(p3-p1-pi),(p3-p0-3*pi/2)];
                %err1=sqrt((p1-p0-pi/2)^2+(p2-p0-pi)^2+(p3-p0-3*pi/2)^2);
                err1=sum(dists.^2);
                if m1==1;
                    err=err1;
                    qq=[m1 m2 m3 m4];
                end
                if err1<err
                    err=err1;
                    qq=[m1 m2 m3 m4];
                end
            end
        end
    end
end
shifts=start+qq;
end

function [shifts,err]=simplePhaseSearching(XXX,start,stop)
p1=start;
firstidx=@(XXX,start,v) find(XXX>(XXX(start)+v),1,'first');
p2=firstidx(XXX,start,1*pi/2)-1;
p3=firstidx(XXX,start,2*pi/2)-1;
p4=firstidx(XXX,start,3*pi/2)-1;%Assuming up interpolation
shifts=[p1,p2,p3,p4];
err=(XXX(p1)-XXX(p2))^2 + (XXX(p3)-XXX(p2))^2 + (XXX(p4)-XXX(p3))^2;
end