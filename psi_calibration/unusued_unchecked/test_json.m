clc; clear all;
addpath('E:\Color_DIC\processing\jsonlab-1.5');
input='the_struggle_is_real.json';
output=strrep(input,'.json','_munged.json');
stub=loadjson(input);
taps=load('taps','taps');
taps=taps.taps;
% %
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value0.weights.value0.top=taps{1}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value0.weights.value0.bot=taps{1}(1,2);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value0.weights.value1.top=taps{2}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value0.weights.value1.bot=taps{2}(1,2);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value0.weights.value2.top=taps{3}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value0.weights.value2.bot=taps{3}(1,2);
% %
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value1.weights.value0.top=taps{1}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value1.weights.value0.bot=taps{1}(1,2);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value1.weights.value1.top=taps{2}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value1.weights.value1.bot=taps{2}(1,2);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value1.weights.value2.top=taps{3}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value1.weights.value2.bot=taps{3}(1,2);
% %
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value2.weights.value0.top=taps{1}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value2.weights.value0.bot=taps{1}(1,2);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value2.weights.value1.top=taps{2}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value2.weights.value1.bot=taps{2}(1,2);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value2.weights.value2.top=taps{3}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value2.weights.value2.bot=taps{3}(1,2);
% %
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value3.weights.value0.top=taps{1}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value3.weights.value0.bot=taps{1}(1,2);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value3.weights.value1.top=taps{2}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value3.weights.value1.bot=taps{2}(1,2);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value3.weights.value2.top=taps{3}(1,1);
stub.value0.SLMPatternSettings.PhaseShift_PatternSettings.value3.weights.value2.bot=taps{3}(1,2);
savejson('',stub,output);
% %
%stub=loadjson('fuck_it.json');
