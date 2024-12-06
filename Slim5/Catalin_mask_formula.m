% This program is modified on March 19th, 2014 (to be used wit Pho Optics Box)
cd('C:\Users\Mikhail\Desktop\fslim\Slim5')
x=1:512;y=1:512;
[X,Y]=meshgrid(y,x);
Ir=115; %145
Or=150; %170;
x1=260;
y1=256; %232;
%--------------------
outer=200;
%1st mask
ring=209; %189; %0, #0  27
Z=(X-x1).^2+(Y-y1).^2;  % last
for m=1:512
    for n=1:512
        if Z(m,n)<(Ir/2-3)^2 %best case fitting ring: 112, 3 for 40X
            Z(m,n)=outer;
        elseif Z(m,n)>(Or/2+3)^2 %best case fitting ring: 130, 3 for 40X
            Z(m,n)=outer;
        else
            Z(m,n)=ring;
        end
    end
end
imwrite(uint8(Z),'pat0.bmp','bmp');
clear Z

% 2nd
ring= 210; %200; %pi/2, #1  39
Z=(X-x1).^2+(Y-y1).^2;  % last
for m=1:512
    for n=1:512
        if Z(m,n)<(Ir/2-3)^2 %best case fitting ring: 112, 3 for 40X
            Z(m,n)=outer;
        elseif Z(m,n)>(Or/2+3)^2 %best case fitting ring: 130, 3 for 40X
            Z(m,n)=outer;
        else
            Z(m,n)=ring;
        end
    end
end
imwrite(uint8(Z),'pat1.bmp','bmp');
clear Z

%3rd
ring= 220; %210; %pi, #2   49
Z=(X-x1).^2+(Y-y1).^2;  % last
for m=1:512
    for n=1:512
        if Z(m,n)<(Ir/2-3)^2 %best case fitting ring: 112, 3 for 40X
            Z(m,n)=outer;
        elseif Z(m,n)>(Or/2+3)^2 %best case fitting ring: 130, 3 for 40X
            Z(m,n)=outer;
        else
            Z(m,n)=ring;
        end
    end
end
imwrite(uint8(Z),'pat2.bmp','bmp');
clear Z

%4th
ring= 239; %220; %3pi/2, #3  59
Z=(X-x1).^2+(Y-y1).^2;  % last
for m=1:512
    for n=1:512
        if Z(m,n)<(Ir/2-3)^2 %best case fitting ring: 112, 3 for 40X
            Z(m,n)=outer;
        elseif Z(m,n)>(Or/2+3)^2 %best case fitting ring: 130, 3 for 40X
            Z(m,n)=outer;
        else
            Z(m,n)=ring;
        end
    end
end
imwrite(uint8(Z),'pat3.bmp','bmp');