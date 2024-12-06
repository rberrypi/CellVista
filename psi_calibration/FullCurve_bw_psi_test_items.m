function FullCurve_bw_psi_test_items
%10/22/2017
%
dbstop if error;
color_comp=[600];
%rect=[267 274 244 252];
%rect=[197,253,18,16];
rect=[];
basedir='Z:\Mikhail\Metrology\2020_17_7_objective_attenuation_measurements\psi_piston_scan_40x_red';
name = @(x) fullfile(basedir,sprintf('f0_t0_i0_ch0_c0_r0_z0_m%d.tif',x));
%name = @(x) fullfile(basedir,sprintf('f0_t0_i0_ch0_c0_r0_z0_mCustom%d.tif',x));
files=0:510;
%constraint = 180:512;
constraint=2:510;
% constraint = 1:length(files);
[phase,~,amp,rect]=PhaseExtraction(name,files,rect);

f=figure;
subplot(2,1,1);
plot(amp);
ylabel('amplitude');
subplot(2,1,2);
plot(phase);
ylabel('phase');

%[ref_shifts,~,~]=PhaseSearching(phase(1,:),amp(1,:),constraint,1);
ref_shifts=[];
close(f);

shiftoids=phase(:,ref_shifts);
for c = 1
    tic;
    [qdic{c},gamma_r{c},taps{c}]=derive_psi_algorithm(shiftoids(c,:));
    toc;
end
qdic_red=show_gallery(phase,amp,rect,name,ref_shifts,files,qdic,gamma_r);
output = @(x) fullfile(basedir,x);
save(output('taps'),'taps');
save(output('qdic'),'qdic');
save(output('gamma_r'),'gamma_r');
ref_shifts=(ref_shifts-1)/2;
save(output('ref_shifts'),'ref_shifts');
writetif(qdic_red,fullfile(basedir,'phase_example.tif'));
end

function slice=getSlice(img,rect)
rect=ceil(rect);
crop =@(img) (img(rect(2):(rect(2)+rect(4)),rect(1):(rect(1)+rect(3))));
slice=crop(img);
end

function [qdic_red] = show_gallery(phase,amp,rect,name,shifts,files,qdic_functor,gamma_functor)
labels={'A','B','C','D','E'};
colors={'r'};
shift_idx=shifts+min(files)-1;
    function label_a_plot(proxy)
        for c=1:length(colors)
            set(h(c),'Color',colors{c},'LineStyle','-')
            for shift=1:length(shifts)
                label=strcat('\leftarrow ',labels{shift});
                text(shift_idx(shift),proxy(c,shifts(shift)),label,'color',colors{c});
            end
        end
    end

subplot(2,3,1);
h=plot(files,amp(1,:));
xlim([min(files),max(files)]);
label_a_plot(amp);
title('amp');

subplot(2,3,4);
h=plot(files,phase(1,:));
xlim([min(files),max(files)]);
label_a_plot(phase);
title('phase');

comp=[];
A=single(imread(name(shift_idx(1))));
B=single(imread(name(shift_idx(2))));
C=single(imread(name(shift_idx(3))));
D=single(imread(name(shift_idx(4))));

subplot(2,3,2);
c=1;
red_processorer = qdic_functor{c};
qdic_red = red_processorer(A(:,:,c),B(:,:,c),C(:,:,c),D(:,:,c));
qdic_red_crop = getSlice(qdic_red,rect);
red_title = sprintf('Phi: %0.2f', mean(qdic_red_crop(:)));
imagesc(qdic_red);axis image;colormap(gray);
title(red_title);

subplot(2,3,3);
c=1;
red_processorer = gamma_functor{c};
gamma_red = red_processorer(A(:,:,c),B(:,:,c),C(:,:,c),D(:,:,c));
gamma_red_crop = getSlice(gamma_red,rect);
red_title = sprintf('Gamma: %0.2f', mean(gamma_red_crop(:)));
imagesc(gamma_red);axis image;colormap(gray);
title(red_title);

subplot(2,3,5);
imagesc(cos(qdic_red).*gamma_red);axis image;colormap(gray);
title('Red Composite');

subplot(2,3,6);
imagesc((A+B+C+D)./((1+cos(qdic_red).*gamma_red)));axis image;colormap(gray);
title('Red Non-Interfermetric');


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

function [phase,gamma,amp,rr]=PhaseExtraction(name,files,rect)
if (isempty(rect))
    h=figure;
    img=imread(name(round(median(files))));
    img=img(:,:,1);
    imagesc(img);axis image;
    rr=getrect;
    rr=ceil(rr);
    close(h);
else
    rr=rect;
end
amp=[];
for i=files
    img=imread(name(i),'PixelRegion',{[rr(2),rr(2)+rr(4)],[rr(1),rr(1)+rr(3)]});
    img=mean2(img);
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

function [shifts,err,avgerror]=PhaseSearching(phase,amp,constraint,big_peak)
if (big_peak<0)
    error('no valid phase shifts');
elseif (big_peak==0)
    [~,lct] = findpeaks(amp,'MinPeakProminence',32);
else
    disp('Warning using negative amplitude hueristic, you might run out of phase shift!');
    [~,lct] = findpeaks(-amp,'MinPeakProminence',4);
end
lct=intersect(lct,constraint);
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
clip = @(v,min_v,max_v) min(max(v,min_v),max_v);
err=inf;
shifts=[];
for startsearch=lct
    p0g=(firstidx(phase,startsearch,-1*pi/2)-1);
    p1g=(startsearch);
    p2g=(firstidx(phase,startsearch,1*pi/2)-1);
    p3g=(firstidx(phase,startsearch,2*pi/2)-1);
    
    p0g=clip(p0g,constraint(1),constraint(end));
    p1g=clip(p1g,constraint(1),constraint(end));
    p2g=clip(p2g,constraint(1),constraint(end));
    p3g=clip(p3g,constraint(1),constraint(end));
    
    if (isempty(p1g)||isempty(p2g)||isempty(p3g)||isempty(p0g))
        continue
    end
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

if (isempty(shifts))
    [shifts,err,avgerror]=PhaseSearching(phase,amp,constraint,big_peak-1);
else
    avgerror=avgStep(phase,shifts(1),shifts(2),shifts(3),shifts(4));
end
end

function [qdic,gamma,taps]=derive_psi_algorithm(shift_list)
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
% top_formula = expand(vpa(top));
% top_forumla_char=char(top_formula)
% bot_formula = expand(vpa(bot));
% bot_forumla_char=char(bot_formula)
% here=0;
top_func=matlabFunction(top);
bot_func=matlabFunction(bot);
const_func=matlabFunction(const);
taps(1,1)=top_func(1,0,0,0);taps(1,2)=bot_func(1,0,0,0);taps(1,3)=const_func(1,0,0,0);
taps(2,1)=top_func(0,1,0,0);taps(2,2)=bot_func(0,1,0,0);taps(2,3)=const_func(0,1,0,0);
taps(3,1)=top_func(0,0,1,0);taps(3,2)=bot_func(0,0,1,0);taps(3,3)=const_func(0,0,1,0);
taps(4,1)=top_func(0,0,0,1);taps(4,2)=bot_func(0,0,0,1);taps(4,3)=const_func(0,0,0,1);
end