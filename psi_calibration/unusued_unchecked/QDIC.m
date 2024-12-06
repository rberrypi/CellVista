function [qdic,sin,cos,int]=QDIC(A,B,C,D,comp)
%Mikhail 6/23
%D=1+cos(t-1pi/2)
%A=1+cos(t+0pi/2)%SLIM convention
%B=1+cos(t+2pi/2)
%C=1+cos(t+3pi/2)
    a2=(B-D)/2;%actually -a2
    a1=(A-C)/2;
    a0=(A+B+C+D)/4;
    int=((a2.^2+a1.^2).^(1/2))./a0;
    sin=a2./int;
    cos=a1./int;
    qdic=atan2(sin,cos);
    qdic=atan2(a2,a1);
end