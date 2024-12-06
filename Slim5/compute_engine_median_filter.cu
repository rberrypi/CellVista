#include "thrust_resize.h"
#include "compute_engine.h"
#include "cuda_error_check.h"

//http://www.cplusplus.com/reference/algorithm/swap/
template <class T> __host__ __device__ void not_std_swap(T& a, T& b)//stupid name to avoid warnings from NVCC
{
	T c(std::move(a)); a = std::move(b); b = std::move(c);
}
template <class T, size_t N> __host__ __device__ void not_std_swap(T(&a)[N], T(&b)[N])
{
	for (size_t i = 0; i < N; ++i) swap(a[i], b[i]);
}

template <class T> __host__ __device__ void compare_exchange(T& top, T& bot)
{
	if (top > bot)// bot is always greater
	{
		//std::swap(top, bot);
		not_std_swap(top, bot);//because std swap complains about using host func. in a kernel
	}
}

template<typename T> __host__ __device__ void five_tap_median(T& a, T& b, T& c, T& d, T& e)
{
	compare_exchange(a, b);
	compare_exchange(d, e);
	compare_exchange(a, c);
	compare_exchange(b, c);
	compare_exchange(a, d);
	compare_exchange(c, d);
	compare_exchange(b, e);
	compare_exchange(b, c);
	compare_exchange(d, e);
}

template<typename T>
using five_tap_avg_filter = void(*)(T* __restrict output, const T* __restrict a, const T* __restrict b, const T* __restrict c, const T* __restrict d, const T* __restrict e, int n);


template<typename T>
__global__ void _getMedGPU(T* __restrict output, const T* __restrict a, const T* __restrict b, const T* __restrict c, const T* __restrict d, const T* __restrict e, int n)
{
	const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		T A = a[idx], B = b[idx], C = c[idx], D = d[idx], E = e[idx];
		five_tap_median(A, B, C, D, E);
		output[idx] = C;//this is stupid but I can't come up with a good scheme to do t fully in-place
	}
}

template<typename T>
__global__ void _getAVGGPU(T* __restrict output, const T* __restrict a, const T* __restrict b, const T* __restrict c, const T* __restrict d, const T* __restrict e, int n)
{
	const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		T A = a[idx], B = b[idx], C = c[idx], D = d[idx], E = e[idx];

		output[idx] = (A + B + C + D + E) / 5;
	}
}

//obiously should read in bigger batches, todo debug is this is a runtime issue
template<typename T>
__global__ void _getHybridFilter(T* __restrict output, const T* __restrict a, const T* __restrict b, const T* __restrict c, const T* __restrict d, const T* __restrict e, const T* __restrict f, T* __restrict g, T* __restrict h, T* __restrict i, T* __restrict j, int n)
{
	const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		T data[10] = { a[idx],b[idx],c[idx],d[idx],e[idx],f[idx],g[idx],h[idx],i[idx],j[idx] };
		{
			T swap;
			if (data[0] > data[5]) { swap = data[0]; data[0] = data[5]; data[5] = swap; }
			if (data[1] > data[6]) { swap = data[1]; data[1] = data[6]; data[6] = swap; }
			if (data[2] > data[7]) { swap = data[2]; data[2] = data[7]; data[7] = swap; }
			if (data[3] > data[8]) { swap = data[3]; data[3] = data[8]; data[8] = swap; }
			if (data[4] > data[9]) { swap = data[4]; data[4] = data[9]; data[9] = swap; }
			if (data[0] > data[3]) { swap = data[0]; data[0] = data[3]; data[3] = swap; }
			if (data[5] > data[8]) { swap = data[5]; data[5] = data[8]; data[8] = swap; }
			if (data[1] > data[4]) { swap = data[1]; data[1] = data[4]; data[4] = swap; }
			if (data[6] > data[9]) { swap = data[6]; data[6] = data[9]; data[9] = swap; }
			if (data[0] > data[2]) { swap = data[0]; data[0] = data[2]; data[2] = swap; }
			if (data[3] > data[6]) { swap = data[3]; data[3] = data[6]; data[6] = swap; }
			if (data[7] > data[9]) { swap = data[7]; data[7] = data[9]; data[9] = swap; }
			if (data[0] > data[1]) { swap = data[0]; data[0] = data[1]; data[1] = swap; }
			if (data[2] > data[4]) { swap = data[2]; data[2] = data[4]; data[4] = swap; }
			if (data[5] > data[7]) { swap = data[5]; data[5] = data[7]; data[7] = swap; }
			if (data[8] > data[9]) { swap = data[8]; data[8] = data[9]; data[9] = swap; }
			if (data[1] > data[2]) { swap = data[1]; data[1] = data[2]; data[2] = swap; }
			if (data[3] > data[5]) { swap = data[3]; data[3] = data[5]; data[5] = swap; }
			if (data[4] > data[6]) { swap = data[4]; data[4] = data[6]; data[6] = swap; }
			if (data[7] > data[8]) { swap = data[7]; data[7] = data[8]; data[8] = swap; }
			if (data[1] > data[3]) { swap = data[1]; data[1] = data[3]; data[3] = swap; }
			if (data[4] > data[7]) { swap = data[4]; data[4] = data[7]; data[7] = swap; }
			if (data[2] > data[5]) { swap = data[2]; data[2] = data[5]; data[5] = swap; }
			if (data[6] > data[8]) { swap = data[6]; data[6] = data[8]; data[8] = swap; }
			if (data[2] > data[3]) { swap = data[2]; data[2] = data[3]; data[3] = swap; }
			if (data[4] > data[5]) { swap = data[4]; data[4] = data[5]; data[5] = swap; }
			if (data[6] > data[7]) { swap = data[6]; data[6] = data[7]; data[7] = swap; }
			if (data[3] > data[4]) { swap = data[3]; data[3] = data[4]; data[4] = swap; }
			if (data[5] > data[6]) { swap = data[5]; data[5] = data[6]; data[6] = swap; }
		}
		//
		T sum = 0;//maybe reuse a variable
		for (auto avg_idx = 2; avg_idx < 8; ++avg_idx)
		{
			sum = sum + data[avg_idx];
		}
		sum = sum / 6;
		//
		output[idx] = sum;
	}
}

void compute_engine::get_hybrid_filter(camera_frame_internal_buffer& output, const std::deque<camera_frame_internal>& input_frames)
{
	auto numel = input_frames[0].buffer->size();
	if (numel == 0)
	{
		output.resize(numel);
	}
	else
	{
		auto res_ptr = thrust_safe_get_pointer(output, numel);
		auto a_ptr = thrust::raw_pointer_cast(input_frames[0].buffer->data());
		auto b_ptr = thrust::raw_pointer_cast(input_frames[1].buffer->data());
		auto c_ptr = thrust::raw_pointer_cast(input_frames[2].buffer->data());
		auto d_ptr = thrust::raw_pointer_cast(input_frames[3].buffer->data());
		auto e_ptr = thrust::raw_pointer_cast(input_frames[4].buffer->data());
		auto f_ptr = thrust::raw_pointer_cast(input_frames[5].buffer->data());
		auto g_ptr = thrust::raw_pointer_cast(input_frames[6].buffer->data());
		auto h_ptr = thrust::raw_pointer_cast(input_frames[7].buffer->data());
		auto i_ptr = thrust::raw_pointer_cast(input_frames[8].buffer->data());
		auto j_ptr = thrust::raw_pointer_cast(input_frames[9].buffer->data());
		static int gridSize, blockSize;
		auto func = _getHybridFilter<float>;
		{
			static int old_size = { 0 };
			auto size_changed = (numel != old_size);
			if (size_changed)
			{
				int minGridSize;//unused?
				CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, 0, 0));
				gridSize = (numel + blockSize - 1) / blockSize;
				old_size = numel;
			}
		}
		func << < gridSize, blockSize >> > (res_ptr, a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, f_ptr, g_ptr, h_ptr, i_ptr, j_ptr, numel);
	}
}



void compute_engine::get_five_tap_filter(camera_frame_internal_buffer& output, const std::deque<camera_frame_internal>& input_frames, denoise_mode filter)
{
	auto numel = input_frames[0].buffer->size();
	if (numel == 0)
	{
		output.resize(numel);
	}
	else
	{
		auto res_ptr = thrust_safe_get_pointer(output, numel);
		auto a_ptr = thrust::raw_pointer_cast(input_frames[0].buffer->data());
		auto b_ptr = thrust::raw_pointer_cast(input_frames[1].buffer->data());
		auto c_ptr = thrust::raw_pointer_cast(input_frames[2].buffer->data());
		auto d_ptr = thrust::raw_pointer_cast(input_frames[3].buffer->data());
		auto e_ptr = thrust::raw_pointer_cast(input_frames[4].buffer->data());
		//
		five_tap_avg_filter<float> function = [&]()
		{
			switch (filter)
			{
			case denoise_mode::average:
				return _getAVGGPU<float>;
			case  denoise_mode::median:
				return _getMedGPU<float>;
			default:
				qli_not_implemented();
			}
		}();
		//
		static int grid_size, blockSize;
		{
			static int old_size = { 0 };
			static five_tap_avg_filter<float> old_processing_function = nullptr;
			auto size_changed = (numel != old_size) || (old_processing_function != function);
			if (size_changed)
			{
				int min_grid_size;//unused?
				CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &blockSize, function, 0, 0));
				grid_size = (numel + blockSize - 1) / blockSize;
				old_size = numel;
				old_processing_function = function;
			}
		}
		function << < grid_size, blockSize >> > (res_ptr, a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, numel);
	}
}
