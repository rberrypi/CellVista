function FullCurve
%1/14/2016
dbstop if error;
basedir='C:\Users\qli\Desktop\test_14_10_2020_amplitude\';
name = @(t,x) sprintf('%s\\f0_t0_i%d_ch0_c0_r0_z0_mCustom%d.tif',basedir,t,x-1);
start=1;stop=254;
t=0;
%start=180;stop=255;
%start=1;stop=254;
[phase,~,amp,rect,fdirname]=PhaseExtraction(basedir,name,t,start,stop);
plotCurves(amp,phase);
[shifts,err,avgerror]=PhaseSearching(phase,amp,start,stop);
fprintf('%d %d %d %d : %d->%d : %f\n',shifts(1),shifts(2),shifts(3),shifts(4),start,stop,err);
testImages(basedir,name,t,shifts,rect,amp,phase,fdirname,start,stop,avgerror);
end

function testImages(basedir,name,t,shifts,rect,amp,phase,fdirname,start,stop,avgerror)
one=double(imread(name(t(1),shifts(1))));%Should be brightest
two=double(imread(name(t(1),shifts(2))));
three=double(imread(name(t(1),shifts(3))));%Should be darkest
four=double(imread(name(t(1),shifts(4))));
frames={one,two,three,four};
orig_int=getComp(frames{1},frames{2},frames{3},frames{4},rect);
comp=orig_int/orig_int(1);
order=[0,1,2,3];
%Make sure A is the brightest
[~,idx]=max(comp);
shifts=circshift(shifts,[1,-idx+1]);
order=circshift(order,[1,-idx+1]);
comp=circshift(comp,[1,-idx+1]);
frames=circshift(frames,[1,-idx+1]);
orig_int=circshift(orig_int,[1,-idx+1]);
labels=sprintf('A=%d,B=%d,C=%d,D=%d',order(1),order(2),order(3),order(4));%labels=sprintf('Comp: %f,%f,%f,%f\n',comp(1),comp(2),comp(3),comp(4));
A=frames{1};B=frames{2};
C=frames{3};D=frames{4};
%
writetif(A,sprintf('%s\\A.tif',basedir));writetif(B,sprintf('%s\\B.tif',basedir));
writetif(C,sprintf('%s\\C.tif',basedir));writetif(D,sprintf('%s\\D.tif',basedir));
%
[m,n]=size(amp);rect2=floor([n*0.25,n*0.75,m*0.25,m*0.75]);
c =@(img) (img(rect2(2):(rect2(2)+rect2(4)),rect2(1):(rect2(1)+rect2(3))));
[deltaPhi,sinval,cosval,inten]=QDIC(A,B,C,D,comp);
sinval=sinval./inten;
cosval=cosval./inten;
smallpart=getSlice(deltaPhi,rect);
varo=var(smallpart(:));
nocomp=atan2((B-D),A-C);
figure;
subplot(2,5,1);
offset=mean2(abs(getSlice(nocomp,rect)));
imagesc(nocomp);title(sprintf('\\Delta Phi (no comp)\n%f',offset));axis image;
writetif(nocomp,sprintf('%s\\nocomp.tif',basedir));
writetif(nocomp,sprintf('%s\\%s.tif',basedir,fdirname));
subplot(2,4,2);
offset=mean2(abs(getSlice(deltaPhi,rect)));
imagesc(deltaPhi);title(sprintf('\\Delta Phi\n%f',offset));axis image;
writetif(deltaPhi,sprintf('%s\\deltaphi.tif',basedir));
subplot(2,4,3);
imagesc(sinval);title('Sine');axis image;
writetif(sinval,sprintf('%s\\sine.tif',basedir));
subplot(2,4,4);
imagesc(cosval);title('Cosine');axis image;
writetif(cosval,sprintf('%s\\kosine.tif',basedir));
subplot(2,4,5);
imagesc(inten);title('Intensity');axis image;
writetif(inten,sprintf('%s\\intensity.tif',basedir));
subplot(2,4,6);
labels2=sprintf('%0.1f,%0.1f,%0.1f,%0.1f',(shifts(1)-1)/2,(shifts(2)-1)/2,(shifts(3)-1)/2,(shifts(4)-1)/2);
cammax=(2^16)-1;
normamp=orig_int/cammax;
bar(normamp);title(sprintf('Choose: %s\nChoose: %s\nBG Var %f',labels,labels2,varo));
subplot(2,4,7);
plot([start:stop],phase/pi);title('Phase!');xlabel('Frame #');ylabel('Pies');
xlim([start,length(phase)+start]);
subplot(2,4,8);
bar([start:stop],amp/cammax);title('Check for saturation amp');
xlim([start,length(amp)+start]);
text(shifts(1),normamp(1),'\leftarrow A','color','r');text(shifts(2),normamp(2),'\leftarrow B','color','r');
text(shifts(3),normamp(3),'\leftarrow C','color','r');text(shifts(4),normamp(4),'\leftarrow D','color','r');
%
colormap gray;
tightfig;
end

function comp=getComp(A,B,C,D,rect)
if (rect)
    rect=ceil(rect);
    CPM =@(img) mean2(img(rect(2):(rect(2)+rect(4)),rect(1):(rect(1)+rect(3))));
    comp=[CPM(A),CPM(B),CPM(C),CPM(D)];
else
    comp=[1,1,1,1];
end
end

function slice=getSlice(img,rect)
    rect=ceil(rect);
    cropped =@(img) (img(rect(2):(rect(2)+rect(4)),rect(1):(rect(1)+rect(3))));
    slice=cropped(img);
end

function plotCurves(amp,phase,shifts,start,stop)
figure;
subplot(2,1,1);
plot(phase/pi);title('Phase!');xlabel('Frame #');ylabel('Pies');
xlim([1,length(phase)]);
subplot(2,1,2);
cammax=(2^16)-1;
plot(amp/cammax);title('Check for saturation amp');
xlim([1,length(amp)]);
drawnow;
end

function [phase,gamma,amp,rr,fdirname]=PhaseExtraction(basedir,name,t,start,stop)
h=figure;
imagesc(imread(name(t(1),1)));axis image;
rr=getrect;
rr=ceil(rr);
close(h);
%pixelregion={[rr(1),rr(1)+rr(3)],[rr(2),rr(2)+rr(4)]};
%[r,c]=size(imread(name(1),'PixelRegion',pixelregion));
amp=zeros(1+start-stop,1);
for i=start:stop
    %disp(testname);
    slices=[];
    for tidx=t
        slices(:,:,end+1)=imread(name(tidx,i),'PixelRegion',{[rr(2),rr(2)+rr(4)],[rr(1),rr(1)+rr(3)]});
    end
    slices=mean(slices(:,:,2:end),3);
    amp(1+i-start)=mean2(slices);
end
raw=amp;
Ymin=min(raw);
Ymax=max(raw);
raw=raw-(Ymin+Ymax)/2;
raw=raw/(Ymax-(Ymin+Ymax)/2);%???
gamma=hilbert(raw);
phase=angle(gamma);
phase=unwrap(phase);
[~,fdirname,~]=fileparts(basedir);
%save(fdirname,'gamma','phase') ;
end

function [shifts,err,avgerror]=PhaseSearching(phase,amp,start,stop)
[~,lct] = findpeaks(amp,'MinPeakProminence',4);
%[~,lct]=max(amp);
firstidx=@(XXX,start,v) find(XXX>(XXX(start)+v),1,'first');%first above
function err=error3(XXX,A,B,C,Z)
                clamp = @(small,value,large) max([min([value,large]),small]);
                pZ=XXX(clamp(1,Z,length(phase)));
                pA=XXX(clamp(1,A,length(phase)));
                pB=XXX(clamp(1,B,length(phase)));
                pC=XXX(clamp(1,C,length(phase)));
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
    disp(shifts+start);
end
avgerror=avgStep(phase,shifts(1),shifts(2),shifts(3),shifts(4));
shifts=start+shifts;
end

function [shifts,err]=PhaseSearchingPump(phase,amp,start,stop)
[~,lct] = findpeaks(amp,'MinPeakProminence',4);
firstidx=@(XXX,start,v) find(XXX>(XXX(start)+v),1,'first');%first above
    function err=error(XXX,A,B,C,D)
        pA=XXX(A);pB=XXX(B);
        pC=XXX(C);pD=XXX(D);
        dists=[(pB-pA-pi/2),(pC-pB-pi/2),(pD-pC-pi/2),...
            (pC-pA-pi),(pD-pB-pi),(pD-pA-3*pi/2)];
        err=sum(dists.^2);
    end
err=inf;
shifts=[];
for startsearch=lct
    p1g=(startsearch);
    p2g=(firstidx(phase,startsearch,1*pi/2)-1);
    p3g=(firstidx(phase,startsearch,2*pi/2)-1);
    p4g=(firstidx(phase,startsearch,3*pi/2)-1);%Assuming up interpolation
    if (isempty(p1g)||isempty(p2g)||isempty(p3g)||isempty(p4g))
        continue
    end
    space=[-1,0,1];
    for p1=p1g+space
        for p2=p2g+space
            for p3=p3g+space
                for p4=p4g+space
                    ernew=error(phase,p1,p2,p3,p4);
                    if (ernew <err)
                        shifts=[p1,p2,p3,p4];
                        err=ernew;
                    end
                end
            end
        end
    end
end
shifts=start+shifts;
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