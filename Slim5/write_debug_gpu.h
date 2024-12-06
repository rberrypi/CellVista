#pragma once
#ifndef WRITE_DEBUG_GPU_H
#define WRITE_DEBUG_GPU_H

//weird stuff is here because thrust often crashes the compiler so we need to isolate it as much as possible
namespace thrust
{
	template<typename T> class device_allocator;
	//template<typename T, typename Alloc> class device_vector;
	template<typename T, typename Alloc  > class device_vector;
}


#include "cuComplex.h"
enum class write_debug_complex_mode { real, imaginary, absolute, log_one_pee, phase };

template<typename T>
void write_debug_gpu(const T* img, int width, int height, int samples_per_pixel, const char* name, bool do_it_anyways = false);
template void write_debug_gpu(const unsigned char* img, int width, int height, int samples_per_pixel, const char* name, bool do_it_anyways);
template void write_debug_gpu(const unsigned short* img, int width, int height, int samples_per_pixel, const char* name, bool do_it_anyways);
template void write_debug_gpu(const int* img, int width, int height, int samples_per_pixel, const char* name, bool do_it_anyways);
template void write_debug_gpu(const float* img, int width, int height, int samples_per_pixel, const char* name, bool do_it_anyways);

template<typename T> void write_debug_gpu(const thrust::device_vector<T,thrust::device_allocator<T>>& img, int width, int height, int samples_per_pixel, const char* name, bool do_it_anyways = false)
{
	const T* ptr = thrust::raw_pointer_cast(img.data());
	write_debug_gpu(ptr, width, height, samples_per_pixel, name, do_it_anyways);
}
template<typename T>
 void write_debug_gpu_complex(const thrust::device_vector<T,thrust::device_allocator<T>>& img, const int width, const int height, const int samples_per_pixel, const char* name, const write_debug_complex_mode mode = write_debug_complex_mode::absolute, const bool do_it_anyways = false)
{
	const auto img_ptr = thrust::raw_pointer_cast(img.data());
	write_debug_gpu_complex(img_ptr, width, height, samples_per_pixel, name, mode, do_it_anyways);
}


void write_debug_gpu_complex(const cuComplex* img, int width, int height, int samples_per_pixel, const char* name, write_debug_complex_mode mode = write_debug_complex_mode::absolute, bool do_it_anyways = false);


template<typename T> void write_debug_gpu_with_pitch(const T* img, int width, int height, int pitch_numel, int samples_per_pixel, const char* name, bool do_it_anyways = false);
template void write_debug_gpu_with_pitch(const unsigned char* img, int width, int height, int pitch_numel, int samples_per_pixel, const char* name, bool do_it_anyways);
template void write_debug_gpu_with_pitch(const float* img, int width, int height, int pitch_numel, int samples_per_pixel, const char* name, bool do_it_anyways);

#endif