#pragma once
#ifndef OCV_HISTOGRAM_H
#define OCV_HISTOGRAM_H
#include <boost/core/noncopyable.hpp>
#include "histogram_info.h"
struct frame_size;
struct cuda_histogram_npp_impl;
class cuda_npp : boost::noncopyable
{
	cuda_histogram_npp_impl* impl_;
	const static auto bin_count = 256;
	const static auto level_count = bin_count + 1;
public:
	void calc_histogram(histogram_info& info, const unsigned char* img_d, int n, int samples_per_pixel, const display_settings::display_ranges& range, bool is_auto_contrast) const;
	// create and initialize
	cuda_npp();
	virtual ~cuda_npp();

	// move: just keep the default
	//OCV_GPU_Add(OCV_GPU_Add&& a) = default; //stupid MSVC2013
	cuda_npp(cuda_npp&& a) noexcept;

	// copy: initialize with a copy of impl
	cuda_npp(const cuda_npp& a);

	cuda_npp& operator=(cuda_npp a);
};
#endif