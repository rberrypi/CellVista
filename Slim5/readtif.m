function [d]= readtif(filename)
foo=imfinfo(filename);
z=length(foo);
slice=foo(1);
if (strcmp(slice.SampleFormat,'Unsigned integer'))
    if (slice.MaxSampleValue >65535)
        dtype='uint32';
    elseif (slice.MaxSampleValue==65535)
        dtype='uint16';
    elseif (slice.MaxSampleValue==255)
        dtype='uint8';
    end
elseif (~isempty(strfind(slice.SampleFormat,'signed integer')))    
    if (slice.MaxSampleValue > 65535)
        dtype='int32';
    elseif (slice.MaxSampleValue==65535)
        dtype='int16';
    elseif (slice.MaxSampleValue==255)
        dtype='int8';
    end
elseif (strcmp(slice.SampleFormat,'IEEE floating point'))
    if (slice.BitsPerSample==32)
        dtype='single';
    elseif (slice.BitsPerSample==64)
        dtype='double';
    end    
end
w=slice.Width;
h=slice.Height;
d=zeros(h,w,z,dtype);
for i=1:z
    d(:,:,i)=imread(filename,i);
end
end