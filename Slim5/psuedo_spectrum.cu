#include "pseudo_spectrum.h"
#include "thrust_resize.h"
#include "cufft_error_check.h"
#include "write_debug_gpu.h"

__global__ void _fillComplexAndShift(cuComplex* dst, const float* src, const int cols, const int rows)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((x < cols) && (y < rows))
	{
		const int odd_even = (y + x) & 1;
		auto a = 1 - 2 * (odd_even);
		auto idx = y * cols + x;
		dst[idx].x = src[idx] * a;
		dst[idx].y = 0;//because there is no imaginary component
	}
}

void fillComplexAndShift(thrust::device_vector<cuComplex>& dst_vector, const thrust::device_vector<float>& src_vector, const frame_size& size)
{
	const auto blocksize = 32;
	dim3 threads(blocksize, blocksize);
	const auto div = [](int W, int X) { return static_cast<int>(ceil(W / (1.0f * X))); };
	dim3 grid(div(size.width, threads.x), div(size.height, threads.y));
	//auto dst = thrust::raw_pointer_cast(dst_vector.data());
	// ReSharper disable once CppLocalVariableMayBeConst
	auto dst = thrust_safe_get_pointer(dst_vector, size.n());
	const auto src = thrust::raw_pointer_cast(src_vector.data());//um, you can't do this right?
	_fillComplexAndShift << <grid, threads >> > (dst, src, size.width, size.height);
}

struct log_1p
{
	__host__ __device__
		float operator()(const cuComplex& x) const {
		const auto h = hypot(x.x, x.y);
		return log1p(h * h);
	}
};

void logCopyBack(thrust::device_vector<float>& dst, const thrust::device_vector<cuComplex>& src)
{
	thrust::transform(src.begin(), src.end(), dst.begin(), log_1p());
}

void pseudo_spectrum::do_pseudo_ft(thrust::device_vector<float>& inplace, const frame_size& frame)
{
#if 0
	const auto do_debug = true;
#else
	const auto do_debug = false;
#endif 
	fillComplexAndShift(img_ft_, inplace, frame);
	write_debug_gpu_complex(img_ft_, frame.width, frame.height, 1, "complex_shifted.tif", write_debug_complex_mode::real, do_debug);
	plan.take_ft(img_ft_, img_ft_, frame.width, frame.height, true);
	logCopyBack(inplace, img_ft_);
}