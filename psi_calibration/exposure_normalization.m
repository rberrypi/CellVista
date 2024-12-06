clc; clear all;
%channel_settings_file ='C:\Users\qli\Desktop\part 10 testing ilumination sequences\acquisition test\rings and psi\channel_0.json';
channel_settings_file = 'C:\Users\qli\Desktop\part 10 testing ilumination sequences\capture_again\scan_settings.json';
detector_range = [70,(2^16)-1];
%
detector_target_mean=max(detector_range)*0.4;
hack_basedir= 'C:\Users\qli\Desktop\part 10 testing ilumination sequences\acquisition test\dot and psi\';
filename_functor =@(x) fullfile(hack_basedir,sprintf('f0_t0_i0_ch0_c0_r0_z0_mDarkfield%d.tif',x));
[output_folder,~]=fileparts(channel_settings_file);
output_file=strrep(channel_settings_file,'.json','_contrast_fixed.json');
addpath('jsonlab-master');
channel_settings_file_temp=tempname;
channel_settings_file_temp_fid = fopen(channel_settings_file_temp,'wt');
input_fid = fopen(channel_settings_file);
input_line = fgetl(input_fid);
while ischar(input_line)
    input_line=strrep(input_line,'NaN','0.0');
    %disp(input_line)
    fprintf(channel_settings_file_temp_fid,'%s',input_line);
    input_line = fgetl(input_fid);
end
fclose(input_fid);
fclose(channel_settings_file_temp_fid);
stub=loadjson(channel_settings_file_temp);
%%
names = fieldnames(stub.gui.light_paths{1}.frames);
for name_idx=1:length(names)
    img=imread(filename_functor(name_idx-1));
    img_mean = mean2(img);
    exposure_boost_factor = detector_target_mean/img_mean;%In reality we need to do some dark current componesation
    old_exposure = stub.gui.light_paths{1}.frames.(names{name_idx}).exposure_time.count;
    new_exposure = min([1500*1000,exposure_boost_factor*old_exposure]);
    stub.gui.light_paths{1}.frames.(names{name_idx}).exposure_time.count = round(new_exposure);
end
savejson('',stub,output_file);