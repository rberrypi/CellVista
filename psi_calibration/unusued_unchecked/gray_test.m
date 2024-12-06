function gray_test
basedir='E:\Color_DIC\rbcs_red';
name = @(x) sprintf('%s\\f0_t0_i0_ch0_c0_r0_z0_m%d.tif',basedir,x);
img=imread(name(1));
img_out = scale_range(img,[0,4095],[0,1]);
Afixed=auto_contrast(img_out);
imagesc(Afixed);axis image;
end

function A_out = scale_range(A,in_range,out_range)
%
%       (b-a)(x - min)
% f(x) = --------------  + a
%           max - min
%
A_out=single(A);
A_out(A_out>in_range(2))=in_range(2);
A_out(A_out<in_range(1))=in_range(1);
scale = (out_range(2)-out_range(1))/(in_range(2)-in_range(1));
A_out=scale*(A_out-in_range(1)+out_range(1));
end

function slice=getSlice(img,rect)
rect=ceil(rect);
crop =@(img) (img(rect(2):(rect(2)+rect(4)),rect(1):(rect(1)+rect(3))));
slice=crop(img);
end

function Afixed=auto_contrast(A)
imagesc(A(:,:,1));axis image;colormap(gray);
rect=getrect;
rect=ceil(rect);
R_mean=mean2(getSlice(A(:,:,1),rect));
G_mean=mean2(getSlice(A(:,:,2),rect));
B_mean=mean2(getSlice(A(:,:,3),rect));
Afixed=A;
Afixed(:,:,1)=(G_mean/R_mean)*Afixed(:,:,1);
Afixed(:,:,3)=(G_mean/B_mean)*Afixed(:,:,3);
end
