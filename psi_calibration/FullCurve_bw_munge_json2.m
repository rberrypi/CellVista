clc; clear all;
addpath('jsonlab-master');
input='C:\Users\qli\Desktop\test_14_10_2020_phase_oldCV\config_file_40x_09-23-2020.json';
output_file='C:\Users\qli\Desktop\test_14_10_2020_phase_oldCV\config_new.json';
input_basedir = fileparts(input);
input_file = @(x) fullfile(input_basedir,x);
output_name=strrep(input,'.json','.json');
stub=loadjson(input);
taps=load(input_file('taps'),'taps');taps=taps.taps;taps=cell2mat(taps);
qdic=load(input_file('qdic'),'qdic');qdic=qdic.qdic;
gamma_r=load(input_file('gamma_r'),'gamma_r');gamma_r=gamma_r.gamma_r;
ref_shifts=load(input_file('ref_shifts'),'ref_shifts');ref_shifts=ref_shifts.ref_shifts;
save(strrep(input,'.json','_qdic_functor'),'qdic');
save(strrep(input,'.json','_gamma_functor'),'gamma_r');
%
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value0.slm_levels.slm_value = ref_shifts(1);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value0.weights.x0x30_.top=taps(1,1);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value0.weights.x0x30_.bot=taps(1,2);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value0.weights.x0x30_.constant=taps(1,3);
% %
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value1.slm_levels.slm_value = ref_shifts(2);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value1.weights.x0x30_.top=taps(2,1);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value1.weights.x0x30_.bot=taps(2,2);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value1.weights.x0x30_.constant=taps(2,3);
% %
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value2.slm_levels.slm_value= ref_shifts(3);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value2.weights.x0x30_.top=taps(3,1);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value2.weights.x0x30_.bot=taps(3,2);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value2.weights.x0x30_.constant=taps(3,3);
% %
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value3.slm_levels.slm_value = ref_shifts(4);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value3.weights.x0x30_.top=taps(4,1);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value3.weights.x0x30_.bot=taps(4,2);
stub.value0.fixed_hardware_settings.modulator_settings.x0x30_.modulator_configuration.four_frame_psi.value3.weights.x0x30_.constant=taps(4,3);
%

savejson('',stub,output_file);
% %