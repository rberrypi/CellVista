clc; clear all;
input='D:\Mikhail\BovineEmbryos_2017_12_08\rbc.json';
output_name=strrep(input,'.json','.json');
output_name_functor=strrep(output_name,'.json','_functor.mat');
qdic=load(output_name_functor,'qdic');qdic=qdic.qdic;
