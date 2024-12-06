#include "polarization_filters.h"
#include "thrust_resize.h"
#include "write_debug_gpu.h"
//unoptomized

template< typename T> __global__
void _merge_quads(float* output, const T* a, const T* b, const T* c, const T* d, int width, int height)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((x < width) && (y < height))
	{
		//okay well now we can do something
		const auto out_idx = width * (y)+x;
		const auto left_half = width / 2;
		const auto bottom_half = height / 2;
		const auto src_idx = left_half * (y % bottom_half) + (x % left_half);
		const auto x_quadrant = x >= left_half;
		const auto y_quadrant = y >= bottom_half;
		if (!x_quadrant && !y_quadrant)
		{
			output[out_idx] = a[src_idx];
		}
		else if (!x_quadrant && y_quadrant)
		{
			output[out_idx] = c[src_idx];
		}
		else if (x_quadrant && y_quadrant)
		{
			output[out_idx] = d[src_idx];
		}
		else if (x_quadrant && !y_quadrant)
		{
			output[out_idx] = b[src_idx];
		}
	}
}

struct stokes
{
	float s0, s1, s2;
	__host__ __device__

		float dolp() const
	{
		return hypot(s1, s2) / s0;
	}
	__host__ __device__

		float orient() const
	{
		//Might be wrong
		return 0.5 * atan2(s2, s1);
	}
};

template<typename T> __host__ __device__
stokes get_stokes(const T& I0, const T& I45, const T& I90, const T& I135)
{
	//See 2.54 in "Chapter 2 Polarizaiton Imaging" , you can also include the ts and tp
	return{ static_cast<float>(I0) + static_cast<float>(I90), static_cast<float>(I0) - static_cast<float>(I90),static_cast<float>(I45) - static_cast<float>(I135) };
}

template< typename T> __global__
void _stoke_0(float* output, const T* a, const T* b, const T* c, const T* d, int numel)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	if (x < numel)
	{
		const auto aa = a[x], bb = b[x], cc = c[x], dd = d[x];
		const auto params = get_stokes(aa, bb, cc, dd);
		output[x] = params.s0;
	}
}

template< typename T> __global__
void _stoke_1(float* output, const T* a, const T* b, const T* c, const T* d, int numel)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	if (x < numel)
	{
		const auto aa = a[x], bb = b[x], cc = c[x], dd = d[x];
		const auto params = get_stokes(aa, bb, cc, dd);
		output[x] = params.s1;
	}
}

template< typename T> __global__
void _stoke_2(float* output, const T* a, const T* b, const T* c, const T* d, int numel)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	if (x < numel)
	{
		const auto aa = a[x], bb = b[x], cc = c[x], dd = d[x];
		const auto params = get_stokes(aa, bb, cc, dd);
		output[x] = params.s2;
	}
}

template< typename T> __global__
void _degree_of_linear_polarization(float* output, const T* a, const T* b, const T* c, const T* d, int numel)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	if (x < numel)
	{
		const auto aa = a[x], bb = b[x], cc = c[x], dd = d[x];
		const auto params = get_stokes(aa, bb, cc, dd);
		output[x] = params.dolp();
	}
}

template< typename T> __global__
void _angle_of_linear_polarization(float* output, const T* a, const T* b, const T* c, const T* d, int numel)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	if (x < numel)
	{
		const auto aa = a[x], bb = b[x], cc = c[x], dd = d[x];
		const auto params = get_stokes(aa, bb, cc, dd);
		output[x] = params.orient();
	}
}

typedef void(*pol_functions)(float* out, const unsigned short* a, const unsigned short* b, const unsigned short* c, const unsigned short* d, int numel);

void polarization_filters::polarization_merger(out_frame out, in_frame a, in_frame b, in_frame c, in_frame d, phase_processing processing)
{
	const static std::unordered_map<phase_processing, pol_functions> function_map = {
	{ phase_processing::stoke_0, _stoke_0<unsigned short> },
	{ phase_processing::stoke_1, _stoke_1<unsigned short> },
	{ phase_processing::stoke_2, _stoke_2<unsigned short> },
	{ phase_processing::degree_of_linear_polarization, _degree_of_linear_polarization<unsigned short> },
	{ phase_processing::angles_of_linear_polarization, _angle_of_linear_polarization<unsigned short> }
	};
	//composes the frames around each other, note they are always even?
	auto ad = (thrust::raw_pointer_cast(a.data()));//360
	auto bd = (thrust::raw_pointer_cast(b.data()));//90
	auto cd = (thrust::raw_pointer_cast(c.data()));//180
	auto dd = (thrust::raw_pointer_cast(d.data()));//270
	const auto numel = a.size();
	auto dst_ptr = thrust_safe_get_pointer(out, numel);
	auto processing_functor = function_map.at(processing);
	static auto old_size = (-1);
	static auto old_functor = processing_functor;
	const auto rebuild_work_units = (numel != old_size) || (old_functor == processing_functor);
	static int grid_size, block_size;
	if (rebuild_work_units)
	{
		int minGridSize;//unused?
		CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &block_size, processing_functor, 0, 0));//todo bug here on the type!!!
		grid_size = (numel + block_size - 1) / block_size;
	}
	processing_functor << <grid_size, block_size >> > (dst_ptr, ad, bd, cd, dd, numel);
	static int gridSize, blockSize;
}

frame_size polarization_filters::merge_quads(out_frame out, in_frame a, in_frame b, in_frame c, in_frame d, const frame_size& size)
{
	//composes the frames around each other, note they are always even?
	frame_size out_size(size.width * 2, size.height * 2);
	auto ad = (thrust::raw_pointer_cast(a.data()));//360
	auto bd = (thrust::raw_pointer_cast(b.data()));//90
	auto cd = (thrust::raw_pointer_cast(c.data()));//180
	auto dd = (thrust::raw_pointer_cast(d.data()));//270
	auto dst_ptr = thrust_safe_get_pointer(out, out_size.n());
	//
	dim3 bs2d(16, 16);//not sure if optimal
	dim3 gs2d;
	gs2d.x = static_cast<unsigned int>(ceil(out_size.width / (1.f * bs2d.x)));
	gs2d.y = static_cast<unsigned int>(ceil(out_size.height / (1.f * bs2d.y)));
	const auto debugg = false;
	_merge_quads << <gs2d, bs2d >> > (dst_ptr, ad, bd, cd, dd, out_size.width, out_size.height);
	write_debug_gpu(dst_ptr, out_size.width, out_size.height, 1, "quad_filter.tif", debugg);
	return out_size;
}
