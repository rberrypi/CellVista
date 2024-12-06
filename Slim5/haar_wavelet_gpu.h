#pragma once
#ifndef HAAR_WAVELET_H
#define HAAR_WAVELET_H
#include "camera_frame.h"
#include "thrust_gpu_var.h"
class haar_wavelet_gpu : thrust_gpu_var
{
	thrust::device_vector<float> haar_result_;
	static void  haarlet(thrust::device_vector<float>& output_d_vec, const  float* input_d_vec, const frame_size& s, bool low_pass);
public:
	float compute_fusion_focus(const float* input, const frame_size& size);
	float compute_fusion_focus(const camera_frame<float>& input);
};

#endif