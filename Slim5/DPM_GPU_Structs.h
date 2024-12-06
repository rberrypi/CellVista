#pragma once
#ifndef DPM_GPU_STRUCTS_H
#define DPM_GPU_STRUCTS_H
#include <thrust/device_vector.h>
#include <cuComplex.h>
#include "cufft_shared.h"
#include "compute_engine_shared.h"
#include <boost/core/noncopyable.hpp>

#include "background_update_functors.h"

class dpm_gpu_structs : boost::noncopyable
{
	thrust::device_vector<float> dpm_out_temp_buffer;
	thrust::device_vector<cuComplex> dpm_in_d_;// , dpm_in_ft_d_;
	//thrust::device_vector<cuComplex> dpm_small_img_temp_d_;
	cufft_wrapper big_ft_, small_inverse_ft_, small_ft_filter;
	enum quad_field { none, q00, q01, q11, q10 };
	[[nodiscard]] dpm_settings dpm_demux_large(out_frame out, in_frame camera_frame, const frame_size& size, const dpm_settings& base_band, bool update_bg);
	[[nodiscard]] dpm_settings dpm_demux(out_frame out, quad_field quad_to_output, in_frame camera_frame, const frame_size& size, const dpm_settings& base_band, bool update_bg);
	void dpm_double_demux(thrust::device_vector<cuComplex>& out, in_frame camera_frame, const frame_size& size, const dpm_settings& base_band, bool update_bg);
	//
	thrust::device_vector<cuComplex> amp_demux, phase_demux;
	thrust::device_vector<cuComplex> Y11, Y12, Y22, Y21;
	void merge_quad_for_pol(out_frame out, dpm_gpu_structs::quad_field field, const thrust::device_vector<cuComplex>& a, float c1_in, const thrust::device_vector<cuComplex>& b, float c2_in, int input_width, int input_height);
	
protected:
	dpm_bg_update_functor dpm_update_;
	
public:
	dpm_gpu_structs(const dpm_gpu_structs& that) = delete;
	dpm_gpu_structs() = default;
	[[nodiscard]] 	frame_size compute_dpm_phase(out_frame out, in_frame camera_frame, phase_processing processing, const frame_size& size, const dpm_settings& dpm_settings, const dpm_bg_update_functor& functor, bool update_bg, int channel_idx);
	[[nodiscard]] 	frame_size compute_dpm_phase_quads(out_frame out, in_frame A, in_frame B, in_frame C, in_frame D, const frame_size& size, const dpm_settings& dpm_settings, const dpm_bg_update_functor& functor, bool update_bg, int channel_idx);
	[[nodiscard]] frame_size compute_dpm_psi_octo(out_frame out, in_frame A, in_frame B, in_frame C, in_frame D, const frame_size& size, const dpm_settings& dpm_settings, const dpm_bg_update_functor& functor, bool update_bg, int channel_idx);
	void pre_allocate_dpm_structs(const frame_size& output_size);
};
#endif