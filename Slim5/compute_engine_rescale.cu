#include "compute_engine.h"
#include "cuda_error_check.h"
#include "write_debug_gpu.h"
#include "clamp_and_scale.h"
#include "qli_runtime_error.h"

template<typename T, typename V>
static inline __host__ __device__  V clamp_n_scale_internal(const T& converme, T from_a, T from_b, T to_a, T to_b)
{
	//todo optomize
	// this can be optomized, for example division is bad
	auto val = (to_b - to_a) * (converme - from_a) / (from_b - from_a) + to_a;
	val = (val > to_b) ? to_b : val;//clamp rules
	val = (val < to_a) ? to_a : val;//clamp rules
	return val;
}

template<typename T, typename V>
__global__ void clamp_n_scale_three_color(V* out, const T* img,
	T from_a_r, T from_b_r, T to_a_r, T to_b_r,
	T from_a_g, T from_b_g, T to_a_g, T to_b_g,
	T from_a_b, T from_b_b, T to_a_b, T to_b_b,
	size_t numel)
{
	const auto i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < numel)
	{
		out[3 * i + 0] = clamp_n_scale_internal<T, V>(img[3 * i + 0], from_a_r, from_b_r, to_a_r, to_b_r);
		out[3 * i + 1] = clamp_n_scale_internal<T, V>(img[3 * i + 1], from_a_g, from_b_g, to_a_g, to_b_g);
		out[3 * i + 2] = clamp_n_scale_internal<T, V>(img[3 * i + 2], from_a_b, from_b_b, to_a_b, to_b_b);
	}
}

void compute_engine::move_clamp_and_scale(unsigned char* out_d_8_bit_img, const  float* img_d, const frame_size& frame_size, const int samples_per_pixel, const display_settings::display_ranges& range)
{
	CUDA_DEBUG_SYNC();
	const auto is_valid_pointer = img_d != nullptr;
	//
	const auto buffer_elements = frame_size.n();
	if (samples_per_pixel == 1)
	{
		const auto img_ptr = thrust::device_pointer_cast(img_d);
		auto out_ptr = thrust::device_pointer_cast(out_d_8_bit_img);
		const auto min = range.front().min;
		const auto max = range.front().max;
		thrust::transform(img_ptr, img_ptr + buffer_elements, out_ptr, clamp_n_scale<float, unsigned char>(min, max, 0.0f, 255.0f));
	}
	else if (samples_per_pixel == 3)
	{
		//maybe use thrust with a 3 strided data type?
		static int gridSize, blockSize;
		auto old_size = 0;
		const auto size_changed = buffer_elements != old_size;
		if (size_changed)
		{
			int minGridSize;//unused?
			CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, clamp_n_scale_three_color<float, unsigned char>, 0, 0));//todo bug here on the type!!!
			gridSize = (buffer_elements + blockSize - 1) / blockSize;
			old_size = buffer_elements;
		}
		clamp_n_scale_three_color << < gridSize, blockSize >> > (out_d_8_bit_img, img_d,
			range[0].min, range[0].max, 0.0f, 255.0f,
			range[1].min, range[1].max, 0.0f, 255.0f,
			range[2].min, range[2].max, 0.0f, 255.0f,
			buffer_elements);
	}
	else
	{
		qli_runtime_error("You got a bug and suck");
	}
	CUDA_DEBUG_SYNC();
}