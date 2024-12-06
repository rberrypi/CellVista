clc;clear all;
names = {'in.tif','back.tif'};
sub={'in','back'};
for name_idx = 1:length(names)
    name = names{name_idx};
    img=imread(name);
    fix_b = @(x) uint16(x(1:4*floor((size(x,1))/4),1:4*floor((size(x,2))/4)));
    bw=@(x) x(:,:,1);
    make_text= @(x,idx,angle) bw(insertText(single(x),[0,size(x,2)/2],sprintf('%d-%d',idx,angle),'FontSize',64));
    %
    for pattern_idx=[0,1,2,3]
        r=0;c=0;
        img_new((1+r):2:(r+2*size(img,1)),(1+c):2:(c+2*size(img,2)))=make_text(img,pattern_idx,90);
        r=1;c=0;
        img_new((1+r):2:(r+2*size(img,1)),(1+c):2:(c+2*size(img,2)))=make_text(img,pattern_idx,135)/2;
        r=1;c=1;
        img_new((1+r):2:(r+2*size(img,1)),(1+c):2:(c+2*size(img,2)))=make_text(img,pattern_idx,0)/8;
        r=0;c=1;
        img_new((1+r):2:(r+2*size(img,1)),(1+c):2:(c+2*size(img,2)))=make_text(img,pattern_idx,45)/2;
        %
        writetif(uint16(img_new),sprintf('test_%d_%s.tif',pattern_idx,sub{name_idx}));
    end
    %
end