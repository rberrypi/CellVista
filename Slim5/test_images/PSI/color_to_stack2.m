function color_to_stack2(img_r,img_g,img_b,file_name_out)
% file='C:\fslim2\Slim5\test_images\PSI\color_psi_set_1\test_3_original.tif';
% file_name_out='C:\fslim2\Slim5\test_images\PSI\color_psi_set_1\test_3.tif';
% img_r=imread(file,1);
% img_g=imread(file,2);
% img_b=imread(file,3);
%
[r, c] = size(img_r);
%data=reshape(reshape([img_r;img_g;img_b],1,[]),r,[]);
data=cat(3,img_r,img_g,img_b);
t = Tiff(file_name_out, 'w');
t.setTag('Photometric', Tiff.Photometric.MinIsBlack);
t.setTag('ImageLength',r);
t.setTag('ImageWidth',c);
t.setTag('BitsPerSample',2*8);
t.setTag('SamplesPerPixel',3);
t.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);%Like teh Campbell's soups
t.setTag('SampleFormat', Tiff.SampleFormat.UInt);
t.write(data);
t.close();
end