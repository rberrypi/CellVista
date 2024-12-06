clc; clear all;
% addpath('jsonlab-1.5');
addpath('jsonlab-master');
input='D:\neha\hiv\try_new\calib\configured_after_old.json';
input_basedir = fileparts(input);
input_file = @(x) fullfile(input_basedir,x);
output_name=strrep(input,'.json','.json');
stub=loadjson(input);
taps=load(input_file('taps'),'taps');taps=taps.taps;
qdic=load(input_file('qdic'),'qdic');qdic=qdic.qdic;
gamma_r=load(input_file('gamma_r'),'gamma_r');gamma_r=gamma_r.gamma_r;
ref_shifts=load(input_file('ref_shifts'),'ref_shifts');ref_shifts=ref_shifts.ref_shifts;
save(strrep(input,'.json','_qdic_functor'),'qdic');
save(strrep(input,'.json','_gamma_functor'),'gamma_r');
%%
stub.value0.modulations.x0x30_.patterns.x0x30_.phase_shift_pattern.slm_value = ref_shifts(1);
stub.value0.modulations.x0x30_.patterns.x0x30_.weights.value0.top=taps(1,1);
stub.value0.modulations.x0x30_.patterns.x0x30_.weights.value0.bot=taps(1,2);
stub.value0.modulations.x0x30_.patterns.x0x30_.weights.value0.constant=taps(1,3);
%%
stub.value0.modulations.x0x30_.patterns.x0x31_.phase_shift_pattern.slm_value = ref_shifts(2);
stub.value0.modulations.x0x30_.patterns.x0x31_.weights.value0.top=taps(2,1);
stub.value0.modulations.x0x30_.patterns.x0x31_.weights.value0.bot=taps(2,2);
stub.value0.modulations.x0x30_.patterns.x0x31_.weights.value0.constant=taps(2,3);
%%
stub.value0.modulations.x0x30_.patterns.x0x32_.phase_shift_pattern.slm_value = ref_shifts(3);
stub.value0.modulations.x0x30_.patterns.x0x32_.weights.value0.top=taps(3,1);
stub.value0.modulations.x0x30_.patterns.x0x32_.weights.value0.bot=taps(3,2);
stub.value0.modulations.x0x30_.patterns.x0x33_.weights.value0.constant=taps(3,3);
%%
stub.value0.modulations.x0x30_.patterns.x0x33_.phase_shift_pattern.slm_value = ref_shifts(4);
stub.value0.modulations.x0x30_.patterns.x0x33_.weights.value0.top=taps(4,1);
stub.value0.modulations.x0x30_.patterns.x0x33_.weights.value0.bot=taps(4,2);
stub.value0.modulations.x0x30_.patterns.x0x33_.weights.value0.constant=taps(4,3);
%%
savejson('',stub,output_name);
% %