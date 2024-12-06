check = @(x) sum(isnan(x(:)));
dpm_frame_in = imread('dpm_frame_in.tif');
disp(check(dpm_frame_in));
dpm_input_mult = imread('dpm_input_mult.tif');
disp(check(dpm_input_mult));
dpm_ft_plain = imread('dpm_ft_plain.tif');
disp(check(dpm_ft_plain));