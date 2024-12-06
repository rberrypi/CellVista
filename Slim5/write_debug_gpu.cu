#include <thrust/host_vector.h>
#include "write_tif.h"
#include "cuda_error_check.h"
#include <complex>
#include "write_debug_gpu.h"
//#include "thrust_resize.h"
template<typename T>
void write_debug_gpu(const T* img, const int width, const int height, const int samples_per_pixel, const char* name, const bool do_it_anyways)
{
#if write_debug
	do_it_anyways = true;
#endif
	if (do_it_anyways)
	{
		CUDA_DEBUG_SYNC();
		auto numel = width * height * samples_per_pixel;
		static thread_local thrust::host_vector<T> data_h(numel);
		data_h.resize(numel);
		const T* ptr = data_h.data();
		CUDASAFECALL(cudaMemcpy(data_h.data(), img, sizeof(T) * numel, cudaMemcpyDeviceToHost));
		write_tif<T>(name, ptr, width, height, samples_per_pixel, nullptr);
		CUDA_DEBUG_SYNC();
	}
}

template<typename T>
void write_debug_gpu_with_pitch(const T* img, int width, int height, const int pitch_numel, int samples_per_pixel, const char* name, bool do_it_anyways)
{
	do_it_anyways = true; // TODO: Remove this fucking line
	if (do_it_anyways)
	{
		CUDA_DEBUG_SYNC();
		int numel = width * height * samples_per_pixel;
		static thread_local thrust::host_vector<T> data_h(numel);
		data_h.resize(numel);
		auto ptr = static_cast<void*>(data_h.data());
		size_t pitch = samples_per_pixel * width * sizeof(T);
		CUDASAFECALL(cudaMemcpy2D(ptr, pitch, img, samples_per_pixel * pitch_numel * sizeof(T), pitch, height, cudaMemcpyKind::cudaMemcpyDeviceToHost)); 
		CUDASAFECALL(cudaDeviceSynchronize());
		auto ptr_t = static_cast<T*>(ptr); // <- this shit now contains the fucking prediction image
		write_tif<T>(name, ptr_t, width, height, samples_per_pixel, nullptr);
	}
}

void write_debug_gpu_complex(const cuComplex* img, const int width, const int height, const int samples_per_pixel, const char* name, const write_debug_complex_mode mode, const bool do_it_anyways)
{
#if write_debug
	do_it_anyways = true;
#endif
	if (do_it_anyways)
	{
		//todo replace with thrust?
		CUDA_DEBUG_SYNC();
		const auto numel = width * height * samples_per_pixel;
		const auto bytes = numel * sizeof(cuComplex);
		const auto bytes_r = numel * sizeof(float);
		cuComplex* tempc;
		CUDASAFECALL(cudaHostAlloc(reinterpret_cast<void**>(&tempc), bytes, cudaHostAllocDefault));
		CUDASAFECALL(cudaMemcpy(tempc, img, bytes, cudaMemcpyDeviceToHost));
		float* temp;
		CUDASAFECALL(cudaHostAlloc(reinterpret_cast<void**>(&temp), bytes_r, cudaHostAllocDefault));
		for (auto i = 0; i < numel; i++)
		{
			const auto v = tempc[i];
			switch (mode)//todo replace with transform
			{
			case write_debug_complex_mode::real:temp[i] = v.x; break;
			case write_debug_complex_mode::imaginary:temp[i] = v.y; break;
			case write_debug_complex_mode::absolute:temp[i] = hypotf(v.x, v.y); break;
			case write_debug_complex_mode::log_one_pee: {
				const auto h = hypot(v.x, v.y);
				temp[i] = log1p(h * h);
				break;
			}
			case write_debug_complex_mode::phase: {
				temp[i] = atan2f(v.y, v.x);
				break;
			}
			}
		}
		write_tif(name, temp, width, height, samples_per_pixel, nullptr);
		CUDASAFECALL(cudaFreeHost(tempc));
		CUDASAFECALL(cudaFreeHost(temp));
	}
}
