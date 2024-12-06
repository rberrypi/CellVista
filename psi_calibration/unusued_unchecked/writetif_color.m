function  writetif_color(data,filename,type)
%Todo handle different types instead of just casting to float
if (nargin <3)
    type='w';
end
if (isempty(data))
    warning(sprintf('%s is empty',filename));
    return;
end
%Big tiff (don't do this automatically because they aren't supported by any
%programs
elements = numel(data);
info=whos('data');
info_bytes = info.bytes;
big= (elements > (2*2^30)) | (info_bytes > (2*2^30));
if ((strcmp(type,'w'))&&(big))
    type='w8';
end
%
if (~isreal(data))
    writetif(imag(data),strrep(filename, '.tif', '_i.tif'),type);
    writetif(real(data),strrep(filename, '.tif', '_r.tif'),type);
    writetif(abs(data),strrep(filename, '.tif', '_mag.tif'),type);
else
    if (isfloat(data))
        bits=4*8;
        format=Tiff.SampleFormat.IEEEFP;
        data=single(data);
    end
    if(isa(data,'uint16'))
        bits=2*8;
        format=Tiff.SampleFormat.UInt;
    end
    if(isa(data,'uint8'))
        bits=1*8;
        format=Tiff.SampleFormat.UInt;
    end
    [r, c,p] = size(data);
    t = Tiff(filename, type);
    t.setTag('Photometric', Tiff.Photometric.MinIsBlack);
    t.setTag('ImageLength',r);
    t.setTag('ImageWidth',c);
    t.setTag('BitsPerSample',bits);
    t.setTag('SamplesPerPixel',p);
    t.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);%Like teh Campbell's soups
    t.setTag('SampleFormat', format);
    t.write(data);
    t.close();
end
end