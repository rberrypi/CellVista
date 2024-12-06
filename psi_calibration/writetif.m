function  writetif(data,filename,type)
%Todo handle different types instead of just casting to float
if (nargin <3)
    type='w';
end
if (~isreal(data))
%     writetif(abs(data),strrep(filename, '.tif', '_abs.tif'),type);
     writetif(imag(data),strrep(filename, '.tif', '_i.tif'),type);
     writetif(real(data),strrep(filename, '.tif', '_r.tif'),type);    
else
if (ndims(data)==3)% Cube
    %First one is weird
    [~,~,z]=size(data);
    writetif(data(:,:,1),filename,'w');
    for i=2:z
        writetif(data(:,:,i),filename,'a');%a for append
    end
else
    bits=32;
    [r, c] = size(data);
    t = Tiff(filename, type);
    t.setTag('Photometric', Tiff.Photometric.MinIsBlack);
    t.setTag('ImageLength',r);
    t.setTag('ImageWidth',c);
    t.setTag('BitsPerSample',bits);
    t.setTag('SamplesPerPixel',1);
    t.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);
    t.setTag('SampleFormat', Tiff.SampleFormat.IEEEFP);
    t.write(single(data));
    t.close();
end
end
end