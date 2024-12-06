#pragma once
#ifndef FOURIER_FILTER_H
#define FOURIER_FILTER_H
#include "compute_engine_shared.h"
#include <thrust/device_vector.h>
#include <cuComplex.h>
#include "cufft_shared.h"
#include "compute_and_scope_state.h"

class fourier_filter
{
	//todo fix complex symmetry for performance reasons
	//todo fix all the spurious static variables, for good reasons (like crashes)
	cufft_wrapper plan_;
	int max_n_;
	const static int demux_bins = 4;
	std::vector<float> dic_filter_cpu_temp_h;
	static constexpr std::array<int, demux_bins> angles = { -1, 0, 45, 90 };
	std::array<thrust::device_vector<cuComplex>, demux_bins>  big_imgs_;
	std::array<thrust::device_vector<float>, demux_bins> filters_g_;
	std::array<thrust::host_vector<float>, demux_bins> filters_h_;
	thrust::device_vector<int> counter;
	band_pass_settings old_band_;
	pixel_dimensions old_dimensions_;
	qdic_scope_settings old_qdic_settings_;
	phase_retrieval old_mode_;
	frame_size old_size_;
	thrust::device_vector<cuComplex> temp_spectrum;
	void lazyRadialAverage(thrust::device_vector<float>& output_lines, const thrust::device_vector<cuComplex>& input, int width_full, int height_full, int bins);
public:
	void pre_allocate_fourier_filters(const frame_size& output_size);
	fourier_filter() noexcept : max_n_(0), old_band_(band_pass_settings()), old_dimensions_(pixel_dimensions()), old_qdic_settings_(qdic_scope_settings()), old_mode_(phase_retrieval::custom_patterns) {
	}

	 void bandfilter_cpu(thrust::host_vector<float>& filter, const band_pass_settings& band, const frame_size& frame);
	 void dic_filter_cpu(thrust::host_vector<float>& filter, float shear_angle, float coherence_length, float pixel_ratio, bool do_derivative, const frame_size& frame);
	void filter_gen_cpu(bool force_regeneration, phase_retrieval mode, const band_pass_settings& bandpass, const scope_compute_settings& qdic, const frame_size& frame);
	void do_filter(camera_frame_internal in_place, phase_retrieval mode, const scope_compute_settings& qdic, const band_pass_settings& band);
};
#endif