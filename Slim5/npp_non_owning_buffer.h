#pragma once
#include <program_config.h>
#include <cuda_error_check.h>
#include "device_factory.h"
#include <npp.h>
#include "npp_error_check.h"
#include "thrust_resize.h"
#include "approx_equals.h"
#include "write_debug_gpu.h"
#include <boost/noncopyable.hpp>
#include "clamp_and_scale.h"
#include "scale.h"
#include <algorithm>
#include "ml_timing.h"


#ifndef NPP_NON_OWNING_BUFFER_
#define NPP_NON_OWNING_BUFFER_

struct npp_dimensions
{
	int nStep;//bytes, also called pitch
	NppiSize Size;
	NppiRect ROI;
	npp_dimensions(const int nStep, const NppiSize& Size, const NppiRect& ROI) : nStep(nStep), Size(Size), ROI(ROI) {}
};

template<typename T>
struct npp_non_owning_buffer : npp_dimensions
{
	T* buffer;
	npp_non_owning_buffer(T* buffer, const int nSrcStep, const NppiSize& Size, const NppiRect& ROI) : npp_dimensions(nSrcStep, Size, ROI), buffer(buffer) {}
	npp_non_owning_buffer(T* buffer, const npp_dimensions& npp_dimensions) : npp_dimensions(npp_dimensions), buffer(buffer) {}
	npp_non_owning_buffer(T* buffer, const NppiSize& Size) : npp_dimensions(Size.width * sizeof(T), Size, { 0,0,Size.width,Size.height }), buffer(buffer) {}
	npp_non_owning_buffer(T* buffer, const frame_size& Size) : npp_non_owning_buffer(buffer, NppiSize{ Size.width,Size.height }) {}
	static npp_non_owning_buffer safe_from_buffer(thrust::device_vector<T>& buffer_to_use, const frame_size& dimensions)
	{
		const NppiSize dimensions_npp = { dimensions.width, dimensions.height };
		return safe_from_buffer(buffer_to_use, dimensions_npp);
	}
	static npp_non_owning_buffer safe_from_buffer(thrust::device_vector<T>& buffer_to_use, const npp_dimensions& dimensions)
	{
		auto right_size = safe_from_buffer(buffer_to_use, dimensions.Size);
		static_cast<npp_dimensions&>(right_size) = dimensions;
		return right_size;
	}
	static npp_non_owning_buffer safe_from_buffer(thrust::device_vector<T>& buffer_to_use, const NppiSize& size)
	{
		const auto number_of_elements = size.width * size.height;
		const auto input_ptr = thrust_safe_get_pointer(buffer_to_use, number_of_elements);
		return npp_non_owning_buffer(input_ptr, size);
	}
	void write_full(const std::string& name, bool do_write) const
	{
		CUDASAFECALL(cudaDeviceSynchronize());
		auto start = buffer;
		const auto pitch_numel = nStep / sizeof(T);
		write_debug_gpu_with_pitch(start, Size.width, Size.height, pitch_numel, 1, name.c_str(), do_write);
	}
	void write(const std::string& name, bool do_write) const
	{
		CUDASAFECALL(cudaDeviceSynchronize());
		auto start = buffer + Size.width * ROI.y + ROI.x;
		const unsigned long long pitch_numel = nStep / sizeof(T);
		write_debug_gpu_with_pitch(start, ROI.width, ROI.height, pitch_numel, 1, name.c_str(), do_write);
	}
	auto thrust_begin()
	{
		return thrust::device_pointer_cast(buffer);
	}
	auto thrust_end()
	{
		const auto numel = Size.width * Size.height;
		return thrust::device_pointer_cast(buffer + numel);
	}
};

#endif