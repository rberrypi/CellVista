function  writetif(data,filename,type)
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
[~,~,z]=size(data);
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
    if (ndims(data)==3)% Cube
        %First one is weird
        if (strcmp(type,'f'))
            zees=1:z;
            for i=zees 
                [pathstr,name,ext] = fileparts(filename);
                full_folder_str = [pathstr,filesep,name,filesep,sprintf('%d_out_of_%d%s',i-1,length(zees),ext)];
                if (i==1)
                    [pathstr,~,~] = fileparts(full_folder_str);
                    mkdir(pathstr);
                end
                writetif(data(:,:,i),full_folder_str);
            end
        else
            writetif(data(:,:,1),filename,type);
            for i=2:z
                writetif(data(:,:,i),filename,'a');%a for append
            end
        end
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
        [r, c] = size(data);
        t = Tiff(filename, type);
        t.setTag('Photometric', Tiff.Photometric.MinIsBlack);
        t.setTag('ImageLength',r);
        t.setTag('ImageWidth',c);
        t.setTag('BitsPerSample',bits);
        t.setTag('SamplesPerPixel',1);
        t.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);%Like teh Campbell's soups
        t.setTag('SampleFormat', format);
        t.write(data);
        t.close();
    end
end
end