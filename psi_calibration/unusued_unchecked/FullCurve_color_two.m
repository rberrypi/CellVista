function FullCurve_color_two
%7/8/2017
%
%
%
%

% bot_forumla_char =
% 
%     '0.29645948088398717948351712741332*b1 - 0.075774994722692570455616904481094*b2 - 0.44564105009959415881594286847626*b3 - 0.052573409768122139615868383165834*b4 + 0.27752997370642168940391102870986*b5'
% 
% Elapsed time is 3.600706 seconds.
% 
% bot_forumla_char =
% 
%     '1.0325639440966268894398257851122*b3 - 0.45034216174288832313312818124005*b2 - 0.51400277809814488623410854696398*b1 + 0.46615590565119201805522509587106*b4 - 0.53437490990678569812781415277928*b5'
% 
% Elapsed time is 3.567842 seconds.
% 
% bot_forumla_char =
% 
%     '0.39267413081472049788804097755246*b1 + 0.4547748781804781078408438211725*b2 - 0.44921778219379810281641387414474*b3 - 0.57465130129386853185870955965126*b4 + 0.17642007449246802894623863507104*b5'

dbstop if error;
color_comp=[600,525,460];
rect=[38   295    77   413];
crange_qdic=[-0.5,0.5];
crange_gamma=[-0.5,0.5];
basedir='E:\Color_DIC\h&e_tma2';
name = @(x) sprintf('%s\\f0_t0_i0_ch0_c0_r0_z0_m%d.tif',basedir,x);
files=300:510;
[phase,~,amp,rect]=PhaseExtraction(basedir,name,files,rect);
[ref_shifts,~,~]=PhaseSearching(phase(1,:),amp(1,:));
%ref_shifts(end+1)=length(phase);
shiftoids=phase(:,ref_shifts);
for c = 1:3
    tic;
    [qdic{c},gamma_r{c}]=derive_psi_algorithm(shiftoids(c,:));
    toc;
end
%show_gallery(phase,amp,rect,name,ref_shifts,files,color_comp,qdic,crange_qdic);
show_gallery(phase,amp,rect,name,ref_shifts,files,color_comp,gamma_r,crange_gamma);
end

function slice=getSlice(img,rect)
rect=ceil(rect);
crop =@(img) (img(rect(2):(rect(2)+rect(4)),rect(1):(rect(1)+rect(3))));
slice=crop(img);
end

function show_gallery(phase,amp,rect,name,shifts,files,color_comp,functor,crange)
labels={'A','B','C','D','E'};
colors={'r','g','b'};
shift_idx=shifts+min(files)-1;
    function label_a_plot(proxy)
        for c=1:3
            set(h(c),'Color',colors{c},'LineStyle','-')
            for shift=1:length(shifts)
                label=strcat('\leftarrow ',labels{shift});
                text(shift_idx(shift),proxy(c,shifts(shift)),label,'color',colors{c});
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
A=single(imread(name(shift_idx(1))));
B=single(imread(name(shift_idx(2))));
C=single(imread(name(shift_idx(3))));
D=single(imread(name(shift_idx(4))));
E=single(imread(name(shift_idx(5))));
% 
% addpath('C:\fslim2\Slim5\test_images\PSI');
% %test_0.tif
% A_sixteen=imread(name(shift_idx(1)));
% B_sixteen=imread(name(shift_idx(2)));
% C_sixteen=imread(name(shift_idx(3)));
% D_sixteen=imread(name(shift_idx(4)));
% outdir='C:\fslim2\Slim5\test_images\PSI\color_psi_dic_with_taps';
% writetif(shift_idx,sprintf('%s\\shift_idx.tif',outdir));
% color_to_stack2(A_sixteen(:,:,1),A_sixteen(:,:,2),A_sixteen(:,:,3),sprintf('%s\\test_0.tif',outdir));
% color_to_stack2(B_sixteen(:,:,1),B_sixteen(:,:,2),B_sixteen(:,:,3),sprintf('%s\\test_1.tif',outdir));
% color_to_stack2(C_sixteen(:,:,1),C_sixteen(:,:,2),C_sixteen(:,:,3),sprintf('%s\\test_2.tif',outdir));
% color_to_stack2(D_sixteen(:,:,1),D_sixteen(:,:,2),D_sixteen(:,:,3),sprintf('%s\\test_3.tif',outdir));

%save(filename,variables);

subplot(2,3,2);
c=1;

red_processorer = functor{c};
qdic_red = red_processorer(A(:,:,c),B(:,:,c),C(:,:,c),D(:,:,c),E(:,:,c));
qdic_red_crop = getSlice(qdic_red,rect);
red_title = sprintf('Red: %0.2f', mean(qdic_red_crop(:)));
imagesc(qdic_red);axis image;colormap(gray);
title(red_title);

subplot(2,3,3);
c=2;
green_processorer = functor{c};
qdic_green = green_processorer(A(:,:,c),B(:,:,c),C(:,:,c),D(:,:,c),E(:,:,c));
green_title = sprintf('Green: %0.2f', mean(qdic_green(:)));
imagesc(qdic_green);axis image;colormap(gray);
title(green_title);

subplot(2,3,5);
c=3;
blue_processorer = functor{c};
qdic_blue = blue_processorer(A(:,:,c),B(:,:,c),C(:,:,c),D(:,:,c),E(:,:,c));
blue_title = sprintf('Blue: %0.2f', mean(qdic_blue(:)));
imagesc(qdic_blue);axis image;colormap(gray);
title(blue_title);

subplot(2,3,6);
merged = cat(3,qdic_red,qdic_green,qdic_blue);
merged_fixed_16 = fix_qdic(merged,rect,color_comp);
merged_fixed_16=uint16(scale_range(merged_fixed_16,crange,[0,65535]));
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
%
writetif(var(qidc_out,[],3),'test_variance.tif');
here=0;
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

function [phase,gamma,amp,rr]=PhaseExtraction(basedir,name,files,rect)
h=figure;
img=imread(name(median(files)));
img=img(:,:,1);
imagesc(img);axis image;
if (isempty(rect))
    rr=getrect;
    rr=ceil(rr);
else
    rr=rect;
end
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

function [qdic,gamma]=derive_psi_algorithm(shift_list)
%Idea from James C. Wyant's lecture notes, and converted to MATLAB
syms a0 a1 a2;
shifts = sym('a',[1 length(shift_list)]);
measured = sym('b',[1 length(shifts)]);
for i=1:length(shifts)
    shifts(i)=shift_list(i);
end
assume(measured,'positive');
irradiance = @(n) a0+a1*cos(shifts(n))+a2*sin(shifts(n));
for i=1:length(shifts)
    temp(i)=(measured(i) - irradiance(i)).^2;
end
error=sum(temp);
d0=diff(error,a0);
d1=diff(error,a1);
d2=diff(error,a2);
eqns = [d0 == 0, d1 == 0, d2==0];
vars = [a0, a1, a2];
[const,bot,top,~,~] = solve(eqns, vars, 'ReturnConditions', true);
top=(-top);
%simplify steps mess with the computational complexity
qdic_algo=(atan2(top,bot));
qdic=matlabFunction(qdic_algo);
gamma_algo=(sqrt((top^2)+(bot^2))/const);
gamma=matlabFunction(gamma_algo);
%formula
top_formula = expand(vpa(top));
top_forumla_char=char(top_formula)
bot_formula = expand(vpa(bot));
bot_forumla_char=char(bot_formula)
here=0;
end