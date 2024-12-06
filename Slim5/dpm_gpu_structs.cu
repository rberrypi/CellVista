#include <algorithm>
#include <cufft.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include "cufft_error_check.h"
#include "thrust_resize.h"
#include "cufft_shared.h"
#include "write_debug_gpu.h"
#include "dpm_gpu_structs.h"
#include "cuda_error_check.h"
#include "time_slice.h"

struct scale_magnitude
{
	//  tell  CUDA that the following code can be executed on the CPU and the GPU
	__host__ __device__  cuComplex  operator()(const cuComplex& x, const float& y) const
	{
		return{ x.x * y, x.y * y };
	}
};

__global__ void _GetBackImage(float* dst, const cuComplex* src, const int w_in, const int h_in, const int w_out, const int h_out)
{
	const int c = threadIdx.x + blockIdx.x * blockDim.x;
	const int r = threadIdx.y + blockIdx.y * blockDim.y;
	// Make sure we do not go out of bounds
	if (r < h_out && c < w_out)
	{
		const auto c_pad = (w_in - w_out) / 2;
		const auto r_pad = (h_in - h_out) / 2;
		const auto r_new = r_pad + r;
		const auto c_new = c_pad + c;
		//
		if ((r_new < h_in) && (c_new < w_in))//dont think this can happen..
		{
			const auto in_idx = r_new * w_in + c_new;
			const auto out_idx = r * w_out + c;
			//
			const auto val = src[in_idx];
			dst[out_idx] = val.x;//This scaling is done at the output / (w_in*w_out);
		}
	}
}

__global__ void _GetBackImage_Imag(float* dst, const cuComplex* src, const int w_in, const int h_in, const int w_out, const int h_out)
{
	const int c = threadIdx.x + blockIdx.x * blockDim.x;
	const int r = threadIdx.y + blockIdx.y * blockDim.y;
	// Make sure we do not go out of bounds
	if (r < h_out && c < w_out)
	{
		const auto c_pad = (w_in - w_out) / 2;
		const auto r_pad = (h_in - h_out) / 2;
		const auto r_new = r_pad + r;
		const auto c_new = c_pad + c;
		//
		if ((r_new < h_in) && (c_new < w_in))//dont think this can happen..
		{
			const auto in_idx = r_new * w_in + c_new;
			const auto out_idx = r * w_out + c;
			//
			auto val = src[in_idx];
			dst[out_idx] = val.y;//This scaling is done at the output / (w_in*w_out);
		}
	}
}

void GetBackImage(cufftReal* img, const bool real_part, thrust::device_vector<cuComplex>& img_big, const frame_size& frame_in, const frame_size& frame_out)
{
	auto  src_ptr = raw_pointer_cast(img_big.data());
	//
	dim3 bs2d(16, 16);//not sure if optimal
	dim3 gs2d;
	gs2d.x = static_cast<unsigned int>(ceil(frame_out.width / (1.f * bs2d.x)));
	gs2d.y = static_cast<unsigned int>(ceil(frame_out.height / (1.f * bs2d.y)));
	if (real_part)
	{
		_GetBackImage << <gs2d, bs2d >> > (img, src_ptr, frame_in.width, frame_in.height, frame_out.width, frame_out.height);
	}
	else
	{
		_GetBackImage_Imag << <gs2d, bs2d >> > (img, src_ptr, frame_in.width, frame_in.height, frame_out.width, frame_out.height);//An alternative is multiply by i or something
	}
}

//dirty hack
__device__ unsigned char val(const unsigned char x)
{
	return x;
}

__device__ float val(const cuComplex x)
{
	return log1p(x.x * x.x + x.y * x.y);
}

template<typename T>
__global__  void _FindCenterOfMass_H(float* sums, const T* in, const int W, const int spacing, const int H)
{
	auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	if (x < W)
	{
		float sum = 0.0;
		for (auto y = 0; y < H; y++)
		{
			auto idx = y * spacing + x;
			sum += val(in[idx]);
		}
		sums[x] = sum;
	}
}

template<typename T>
__global__  void _FindCenterOfMass_W(float* sums, const T* in, int W, int spacing, int H)
{
	const int y = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	if (y < H)
	{
		float sum = 0.0;
		for (auto x = 0; x < W; x++)
		{
			auto idx = y * spacing + x;
			sum += val(in[idx]);
		}
		sums[y] = sum;
	}
}

template<typename T>
dpm_settings FindCenterOfMass(const thrust::device_vector<T>& array, const dpm_settings& guess, const int spacing, const int H_original, const int W_original)
{
	// Keep it Simple™
	// Todo replace with more thrust
	const auto blocksize = 64;
	dim3 threads(blocksize, 1);
	auto H = guess.dpm_phase_width;//its a square! (are these switched)
	auto W = guess.dpm_phase_width;
	thrust::host_vector<float> sumH_h(W, 0), sumW_h(H, 0);
	thrust::device_vector<float>  sumH_d(W, 0), sumW_d(H, 0);
	auto inrange = [](int x, int max) {return x > 0 && x < max; };
	auto uo = guess.dpm_phase_left_column;
	auto vo = guess.dpm_phase_top_row;
	if (!inrange(vo, H_original) || !inrange(uo, W_original))
	{
		//maybe throw or clamp, or fix up?
		return guess;
	}
	const T* top_of_array_d = thrust::raw_pointer_cast(array.data());
	const T* in = &top_of_array_d[vo * spacing + uo];
	{
		//Sum top to bottom for column max
		dim3 gridH(ceil(W / (1.0 * threads.x)), 1);
		auto sumH_d_ptr = thrust::raw_pointer_cast(sumH_d.data());
		_FindCenterOfMass_H << <gridH, threads >> > (sumH_d_ptr, in, W, spacing, H);
		CUDA_DEBUG_SYNC();
		thrust::copy(sumH_d.begin(), sumH_d.end(), sumH_h.begin());
	}
	{
		//Sum Left to Right for row max
		dim3 gridH(ceil(H / (1.0 * threads.x)), 1);
		auto sumW_d_ptr = thrust::raw_pointer_cast(sumW_d.data());
		_FindCenterOfMass_W << <gridH, threads >> > (sumW_d_ptr, in, W, spacing, H);
		CUDA_DEBUG_SYNC();
		thrust::copy(sumW_d.begin(), sumW_d.end(), sumW_h.begin());
	}
	//
	auto maxidx = [](auto in, int N) {return std::distance(in, std::max_element(in, in + N)); };
	uo = maxidx(sumH_h.begin(), W);
	vo = maxidx(sumW_h.begin(), H);
	//center
	uo = guess.dpm_phase_left_column + uo;
	vo = guess.dpm_phase_top_row + vo;
	//Now we want the corner
	auto fix = [](int value, int min, int max) {return std::max(std::min(value, max), min); };
	uo = fix(uo - guess.dpm_phase_width / 2, 0, W_original - guess.dpm_phase_width / 2);
	vo = fix(vo - guess.dpm_phase_width / 2, 0, H_original - guess.dpm_phase_width / 2);
	auto mass = guess;
	mass.dpm_phase_left_column = uo;
	mass.dpm_phase_top_row = vo;
	return mass;
}

struct RangeScaling
{
	const float min_value_in, min_value_out;
	const float scaleR;
	RangeScaling(const float& Min_value_in, const float& Max_value_in, const float& Min_value_out, const float& Max_value_out) :
		min_value_in(Min_value_in), min_value_out(Min_value_out), scaleR((Max_value_out - min_value_out) / (Max_value_in - min_value_in))
	{
	}
	__host__ __device__
		float operator()(const cuComplex& a) const
	{
		auto h = min_value_out + (log1p(a.x * a.x + a.y * a.y) - min_value_in) * scaleR;
		//std::clamp
		return static_cast<unsigned char>(thrust::min(thrust::max(h, 0.0f), 255.0f));
	}
};

__global__ void _FillComplexAndShift(cuComplex* __restrict__ dst, const unsigned short* __restrict__ src, int width, int rows)
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((x < width) && (y < rows))
	{
		const auto a = 1 - 2 * ((y + x) & 1);
		const auto in = y * width + x;
		const auto out = y * width + x;
		const float value = src[in];
		dst[out].x = value * a;
		dst[out].y = 0;
	}
}

void FillComplexAndShift(thrust::device_vector<cuComplex>& dst_vector, const thrust::device_vector<unsigned short>& src_vector, const frame_size& in)
{
	const auto blocksize = 32;//dies due to occupancy problems past 32, todo maybe replace with thrust
	dim3 threads(blocksize, blocksize);
	const auto div = [](int W, int X) { return static_cast<int>(ceil(W / (1.0f * X))); };
	dim3 grid(div(in.width, threads.x), div(in.height, threads.y));
	auto dst = thrust_safe_get_pointer(dst_vector, in.n());
	const auto src = thrust::raw_pointer_cast(src_vector.data());
	_FillComplexAndShift << <grid, threads >> > (dst, src, in.width, in.height);
}

__global__ void _fillComplexAndShiftoutOfPlace(cuComplex* dst, cuComplex* src, const int cols, const int rows)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((x < cols) && (y < rows))
	{
		const int odd_even = (y + x) & 1;
		auto a = 1 - 2 * (odd_even);
		auto idx = y * cols + x;
		dst[idx].x = src[idx].x * a;
		dst[idx].y = src[idx].y * a;
	}
}

__global__ void _fillComplexAndShiftInPlaceNoQuads(cuComplex* dst, const int cols, const int rows)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((x < cols) && (y < rows))
	{
		const int odd_even = (y + x) & 1;
		auto a = 1 - 2 * (odd_even);
		auto idx = y * cols + x;
		dst[idx].x = dst[idx].x * a;
		dst[idx].y = dst[idx].y * a;
	}
}

template<bool south, bool west>
__global__ void _fillComplexAndShiftQuads(cuComplex* dst, const cuComplex* src, const int cols_in, const int rows_in)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((x < cols_in) && (y < rows_in))
	{
		const int odd_even = (y + x) & 1;
		auto a = 1 - 2 * (odd_even);
		auto idx_in = y * cols_in + x;
		const auto offset_y = south ? (rows_in) : 0;
		const auto offset_x = west ? (cols_in) : 0;
		auto idx_out = (y + offset_y) * (2 * cols_in) + (x + offset_x);
		dst[idx_out].x = src[idx_in].x * a;
		dst[idx_out].y = src[idx_in].y * a;
	}
}

dpm_settings dpm_gpu_structs::dpm_demux_large(out_frame out, in_frame camera_frame, const frame_size& size, const dpm_settings& base_band, bool update_bg)
{
	const auto debuggle = false;
	if (!(base_band.dpm_phase_is_complete() && base_band.fits_in_frame(size)))
	{
		std::cout << "Warning invalid dpm settings" << std::endl;
		auto output_width = base_band.dpm_phase_width;
		thrust_safe_resize(out, 2 * output_width * output_width);
		thrust::fill(out.begin(), out.end(), 0.0f);
		return base_band;
	}
	//
	write_debug_gpu(camera_frame, size.width, size.height, 1, "DPM_1_input.tif", debuggle);
	//demux
	FillComplexAndShift(dpm_in_d_, camera_frame, size);//int spacein, int spaceout, int cols, int rows
	write_debug_gpu_complex(dpm_in_d_, size.width, size.height, 1, "DPM_2_complex_shifted.tif", write_debug_complex_mode::absolute, debuggle);
	big_ft_.take_ft(dpm_in_d_, dpm_in_d_, size.width, size.height, true);
	write_debug_gpu_complex(dpm_in_d_, size.width, size.height, 1, "DPM_3_FT.tif", write_debug_complex_mode::log_one_pee, debuggle);
	//
	{
		auto dest_ft = reinterpret_cast<cuComplex*>(thrust_safe_get_pointer(dpm_out_temp_buffer, 2 * size.n()));
		thrust::fill(dpm_out_temp_buffer.begin(), dpm_out_temp_buffer.end(), 0.0f);
		const int center_and_offset_column = round((size.width - base_band.dpm_phase_width) / 2);
		const int center_and_offset_row = round((size.height - base_band.dpm_phase_width) / 2);
		auto dest_ft_start = dest_ft + center_and_offset_column + center_and_offset_row * size.width;
		const auto src = thrust::raw_pointer_cast(dpm_in_d_.data());
		const auto src_offset = src + size.width * base_band.dpm_phase_top_row + base_band.dpm_phase_left_column;
		cudaMemcpy2D(dest_ft_start, sizeof(cuComplex) * size.width, src_offset, sizeof(cuComplex) * size.width, base_band.dpm_phase_width * sizeof(cuComplex), base_band.dpm_phase_width, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
		write_debug_gpu_complex(dest_ft, size.width, size.height, 1, "DPM_4_FT_Cropped.tif", write_debug_complex_mode::log_one_pee, debuggle);
		big_ft_.take_ft(dest_ft, dest_ft, size.width, size.height, true);
		write_debug_gpu_complex(dest_ft, size.width, size.height, 1, "DPM_5_IFT.tif", write_debug_complex_mode::log_one_pee, debuggle);
	}
	{
		const auto dest_ft = reinterpret_cast<float2*>(thrust::raw_pointer_cast(dpm_out_temp_buffer.data()));
		const auto blocksize = 32;
		dim3 threads(blocksize, blocksize);
		const auto div = [](int W, int X) { return static_cast<int>(ceil(W / (1.0f * X))); };
		dim3 grid(div(size.width, threads.x), div(size.height, threads.y));
		auto final_output_ptr = reinterpret_cast<float2*>(thrust_safe_get_pointer(out, 2 * size.n()));
		_fillComplexAndShiftoutOfPlace << <grid, threads >> > (final_output_ptr, dest_ft, size.width, size.height);
		write_debug_gpu_complex(final_output_ptr, size.width, size.height, 1, "DPM_6_IFT.tif", write_debug_complex_mode::real, debuggle);
	}
	//Copy into another place
	return base_band;
}

dpm_settings dpm_gpu_structs::dpm_demux(out_frame out, quad_field quad_to_output, in_frame camera_frame, const frame_size& size, const dpm_settings& base_band, bool update_bg)
{
	const auto debuggle = false;
	const auto valid_dpm = base_band.dpm_phase_is_complete();
	const auto fits_in_frame = base_band.fits_in_frame(size);
	if (!(valid_dpm && fits_in_frame))
	{
		std::cout << "Warning invalid dpm settings" << std::endl;
		auto output_width = base_band.dpm_phase_width;
		if (quad_to_output != quad_field::none)
		{
			//teriblru hack
			output_width *= 2;
		}
		thrust_safe_resize(out, 2 * output_width * output_width);
		thrust::fill(out.begin(), out.end(), 0.0f);
		return base_band;
	}
	//
	write_debug_gpu(camera_frame, size.width, size.height, 1, "DPM_1_input.tif", debuggle);
	//demux
	FillComplexAndShift(dpm_in_d_, camera_frame, size);//int spacein, int spaceout, int cols, int rows
	write_debug_gpu_complex(dpm_in_d_, size.width, size.height, 1, "DPM_2_complex_shifted.tif", write_debug_complex_mode::absolute, debuggle);
	big_ft_.take_ft(dpm_in_d_, dpm_in_d_, size.width, size.height, true);
	write_debug_gpu_complex(dpm_in_d_, size.width, size.height, 1, "DPM_3_FT.tif", write_debug_complex_mode::absolute, debuggle);
	//
	auto get_new_baseband = [&]
	{
		return base_band.dpm_snap_bg && update_bg ? FindCenterOfMass(dpm_in_d_, base_band, size.width, size.height, size.width) : base_band;
	};
	//
	auto direct_write = quad_to_output == quad_field::none;
	auto new_base_band = get_new_baseband();
	const auto complex_is_bigger = 2;
	auto& final_ft_destination = direct_write ? out : dpm_out_temp_buffer;
	auto final_ft_destination_ptr = reinterpret_cast<float2*>(thrust_safe_get_pointer(final_ft_destination, new_base_band.dpm_phase_width * new_base_band.dpm_phase_width * complex_is_bigger));
	ft_rectangle src_rect = { size.width,size.height, new_base_band.dpm_phase_left_column,new_base_band.dpm_phase_top_row };
	ft_rectangle dst_rect = { new_base_band.dpm_phase_width,new_base_band.dpm_phase_width, 0,0 };
	auto img_ft = thrust::raw_pointer_cast(dpm_in_d_.data());
	small_inverse_ft_.take_ft(img_ft, final_ft_destination_ptr, ft_settings(new_base_band.dpm_phase_width, new_base_band.dpm_phase_width, src_rect, dst_rect), false);
	write_debug_gpu_complex(final_ft_destination_ptr, new_base_band.dpm_phase_width, new_base_band.dpm_phase_width, 1, "DPM_4_IFT.tif", write_debug_complex_mode::absolute, debuggle);
	//
	{
		const auto blocksize = 32;
		dim3 threads(blocksize, blocksize);
		const auto div = [](int W, int X) { return static_cast<int>(ceil(W / (1.0f * X))); };
		dim3 grid(div(new_base_band.dpm_phase_width, threads.x), div(new_base_band.dpm_phase_width, threads.y));
		if (direct_write)
		{
			_fillComplexAndShiftInPlaceNoQuads << <grid, threads >> > (final_ft_destination_ptr, new_base_band.dpm_phase_width, new_base_band.dpm_phase_width);
		}
		else
		{
			auto out_quad_ptr = reinterpret_cast<float2*>(thrust_safe_get_pointer(out, 4 * new_base_band.dpm_phase_width * new_base_band.dpm_phase_width * complex_is_bigger));
			switch (quad_to_output)
			{
			case q00:
				_fillComplexAndShiftQuads<false, false> << <grid, threads >> > (out_quad_ptr, final_ft_destination_ptr, new_base_band.dpm_phase_width, new_base_band.dpm_phase_width);
				break;
			case q01:
				_fillComplexAndShiftQuads<false, true> << <grid, threads >> > (out_quad_ptr, final_ft_destination_ptr, new_base_band.dpm_phase_width, new_base_band.dpm_phase_width);
				break;
			case q11:
				_fillComplexAndShiftQuads<true, true> << <grid, threads >> > (out_quad_ptr, final_ft_destination_ptr, new_base_band.dpm_phase_width, new_base_band.dpm_phase_width);
				break;
			case q10:
				_fillComplexAndShiftQuads<true, false> << <grid, threads >> > (out_quad_ptr, final_ft_destination_ptr, new_base_band.dpm_phase_width, new_base_band.dpm_phase_width);
				break;
			}
			CUDA_DEBUG_SYNC();
		}
	}
	//
	return new_base_band;
}

frame_size dpm_gpu_structs::compute_dpm_phase(out_frame out, in_frame camera_frame, const phase_processing processing, const frame_size& size, const dpm_settings& dpm_settings, const dpm_bg_update_functor& functor, bool update_bg, int channel_idx)
{
	auto output = [&]
	{
		if (processing == phase_processing::diffraction_phase)
		{
			auto new_dpm_settings = dpm_demux(out, quad_field::none, camera_frame, size, dpm_settings, update_bg);
			frame_size output(new_dpm_settings.dpm_phase_width, new_dpm_settings.dpm_phase_width);
			if (update_bg)
			{
				if (!functor)
				{
					qli_runtime_error("Welp, this should be set");
				}
				functor(new_dpm_settings, channel_idx);
			}
			return output;
		}
		if (processing == phase_processing::diffraction_phase_larger)
		{
			auto settings = dpm_demux_large(out, camera_frame, size, dpm_settings, update_bg);
			return size;
		}
		qli_runtime_error("Mode isn't supported, you suck");
	}();
	return output;
}

frame_size dpm_gpu_structs::compute_dpm_phase_quads(out_frame out, in_frame A, in_frame B, in_frame C, in_frame D, const frame_size& size, const dpm_settings& dpm_settings, const dpm_bg_update_functor& functor, bool update_bg, int channel_idx)
{
	auto debug = false;
	write_debug_gpu(A, size.width, size.height, 1, "A.tif", debug);
	write_debug_gpu(B, size.width, size.height, 1, "B.tif", debug);
	write_debug_gpu(C, size.width, size.height, 1, "C.tif", debug);
	write_debug_gpu(D, size.width, size.height, 1, "D.tif", debug);
	auto new_dpm_settings = dpm_demux(out, quad_field::q00, A, size, dpm_settings, update_bg);
	dpm_demux(out, quad_field::q01, B, size, new_dpm_settings, false);
	dpm_demux(out, quad_field::q10, C, size, new_dpm_settings, false);
	dpm_demux(out, quad_field::q11, D, size, new_dpm_settings, false);
	//
	frame_size output(new_dpm_settings.dpm_phase_width * 2, new_dpm_settings.dpm_phase_width * 2);
	if (update_bg)
	{
		if (!functor)
		{
			qli_invalid_arguments();
		}
		functor(new_dpm_settings, channel_idx);
	}
	return output;
}

struct mulitply_into
{
	//  tell  CUDA that the following code can be executed on the CPU and the GPU
	__host__ __device__  cuComplex  operator()(const cuComplex& x, const cuComplex& y) const
	{
		return cuCmulf(x, y);
	}
};

void cudaMemcpy2D_safer(thrust::device_vector<cuComplex>& dst_v, size_t dpitch, const thrust::device_vector<cuComplex>& src_v, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	auto dst = thrust::raw_pointer_cast(dst_v.data());
	auto src = thrust::raw_pointer_cast(src_v.data());
	cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

void dpm_gpu_structs::dpm_double_demux(thrust::device_vector<cufftComplex>& out, in_frame camera_frame, const frame_size& size, const dpm_settings& base_band, bool update_bg)
{
	const auto debuggle = false;
	auto output_width = std::max(base_band.dpm_phase_width, base_band.dpm_amp_width);
	if (!(base_band.dpm_phase_is_complete() && base_band.fits_in_frame(size)))
	{
		std::cout << "Warning invalid dpm settings" << std::endl;
		thrust_safe_resize(out, output_width * output_width);
		thrust::fill(out.begin(), out.end(), make_cuComplex(0, 0));
		return;
	}
	//Step 1 Take FT of input image
	{
		FillComplexAndShift(dpm_in_d_, camera_frame, size);//int spacein, int spaceout, int cols, int rows
		big_ft_.take_ft(dpm_in_d_, dpm_in_d_, size.width, size.height, true);
		write_debug_gpu_complex(dpm_in_d_, size.width, size.height, 1, "DPM_FT.tif", write_debug_complex_mode::log_one_pee, debuggle);
	}
	//Do Phase
	{
		auto phase_dest = thrust_safe_get_pointer(phase_demux, output_width * output_width);
		if (output_width != base_band.dpm_phase_width)
		{
			thrust::fill(phase_demux.begin(), phase_demux.end(), make_cuComplex(0, 0));
		}
		auto img_ft = thrust::raw_pointer_cast(dpm_in_d_.data());
		const auto src_offset = img_ft + size.width * base_band.dpm_phase_top_row + base_band.dpm_phase_left_column;
		auto shift_left = (output_width - base_band.dpm_phase_width) / 2;
		auto shift_top = (output_width - base_band.dpm_phase_width) / 2;
		const auto dst_offset = phase_dest + shift_left + shift_top * output_width;
		CUDA_DEBUG_SYNC();
		//( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind ) 
		CUDASAFECALL(cudaMemcpy2D(dst_offset, output_width * sizeof(cuComplex), src_offset, size.width * sizeof(cuComplex), base_band.dpm_phase_width * sizeof(cuComplex), base_band.dpm_phase_width, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
		CUDA_DEBUG_SYNC();
		write_debug_gpu_complex(phase_dest, output_width, output_width, 1, "DPM_PHASE_Part.tif", write_debug_complex_mode::log_one_pee, debuggle);
		small_ft_filter.take_ft(phase_dest, phase_dest, output_width, output_width, false);
		write_debug_gpu_complex(phase_dest, output_width, output_width, 1, "DPM_PHASE_FT_Part.tif", write_debug_complex_mode::phase, debuggle);
		CUDA_DEBUG_SYNC();
	}
	//Do Amp
	{
		auto amp_dest = thrust_safe_get_pointer(amp_demux, output_width * output_width);
		if (output_width != base_band.dpm_amp_width)
		{
			thrust::fill(amp_demux.begin(), amp_demux.end(), make_cuComplex(0, 0));
		}
		auto img_ft = thrust::raw_pointer_cast(dpm_in_d_.data());
		const auto src_offset = img_ft + size.width * base_band.dpm_amp_top_row + base_band.dpm_amp_left_column;
		auto shift_left = (output_width - base_band.dpm_amp_width) / 2;
		auto shift_top = (output_width - base_band.dpm_amp_width) / 2;
		const auto dst_offset = amp_dest + shift_left + shift_top * output_width;
		CUDA_DEBUG_SYNC();
		CUDASAFECALL(cudaMemcpy2D(dst_offset, output_width * sizeof(cuComplex), src_offset, size.width * sizeof(cuComplex), base_band.dpm_amp_width * sizeof(cuComplex), base_band.dpm_amp_width, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
		CUDA_DEBUG_SYNC();
		write_debug_gpu_complex(amp_dest, output_width, output_width, 1, "DPM_AMP_Part.tif", write_debug_complex_mode::log_one_pee, debuggle);
		small_ft_filter.take_ft(amp_dest, amp_dest, output_width, output_width, false);
		write_debug_gpu_complex(amp_dest, output_width, output_width, 1, "DPM_AMP_FT_Part.tif", write_debug_complex_mode::real, debuggle);
	}
	//Not we could probably merge these guys, but we're going to do this else we'd need to have logic to ensure the non-zero padded one is always selected as the otput (else would require a clear)
	thrust_safe_resize(out, output_width * output_width);
	thrust::transform(amp_demux.begin(), amp_demux.end(), phase_demux.begin(), out.begin(), mulitply_into());
	write_debug_gpu_complex(out, output_width, output_width, 1, "DPM_MERGED_PHASE.tif", write_debug_complex_mode::phase, debuggle);
	CUDA_DEBUG_SYNC();
}

template<bool south, bool west>
__global__ void _jones_demux(cuComplex* dst, const cuComplex* A, float c1, const cuComplex* B, float c2, const int cols_in, const int rows_in)
{
	const auto x = threadIdx.x + blockDim.x * blockIdx.x;//leave as int so that subtraction still works
	const auto y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((x < cols_in) && (y < rows_in))
	{
		const auto mult = [](cuComplex& in, const float& s)
		{
			in.x *= s;
			in.y *= s;
			return in;
		};
		auto idx_in = y * cols_in + x;
		const auto offset_y = south ? (rows_in) : 0;
		const auto offset_x = west ? (cols_in) : 0;
		auto idx_out = (y + offset_y) * (2 * cols_in) + (x + offset_x);
		auto a = A[idx_in], b = B[idx_in];
		auto div_factor = 1 / (2 * c1 * c2);
		cuComplex result = mult(cuCsubf(mult(a, c2), mult(b, c1)), div_factor);
		dst[idx_out] = result;
		//
		//const int odd_even = (y + x) & 1;
		//auto shift_factoid = 1 - 2 * (odd_even);
		//dst[idx_out] = mult(result, shift_factoid);
	}
}

void dpm_gpu_structs::merge_quad_for_pol(out_frame out, dpm_gpu_structs::quad_field field, const thrust::device_vector<cuComplex>& a, float c1_in, const thrust::device_vector<cuComplex>& b, float c2_in, int input_width, int input_height)
{
	const auto blocksize = 32;
	dim3 threads(blocksize, blocksize);
	const auto div = [](int W, int X) { return static_cast<int>(ceil(W / (1.0f * X))); };
	dim3 grid(div(input_width, threads.x), div(input_height, threads.y));
	auto complex_factor = sizeof(cuComplex) / sizeof(float);
	auto out_quad_ptr = reinterpret_cast<cuComplex*>(thrust_safe_get_pointer(out, 4 * complex_factor * input_width * input_height));
	auto a_ptr = thrust::raw_pointer_cast(a.data());
	auto b_ptr = thrust::raw_pointer_cast(b.data());
	CUDA_DEBUG_SYNC();
	switch (field)
	{
	case q00:
		_jones_demux<false, false> << <grid, threads >> > (out_quad_ptr, a_ptr, c1_in, b_ptr, c2_in, input_width, input_height);
		break;
	case q01:
		_jones_demux<false, true> << <grid, threads >> > (out_quad_ptr, a_ptr, c1_in, b_ptr, c2_in, input_width, input_height);
		break;
	case q11:
		_jones_demux<true, true> << <grid, threads >> > (out_quad_ptr, a_ptr, c1_in, b_ptr, c2_in, input_width, input_height);
		break;
	case q10:
		_jones_demux<true, false> << <grid, threads >> > (out_quad_ptr, a_ptr, c1_in, b_ptr, c2_in, input_width, input_height);
		break;
	}
	CUDA_DEBUG_SYNC();
}

struct add_complex
{
	__host__ __device__ cuComplex operator()(const cuComplex& A, const cuComplex& B) const
	{
		return cuCaddf(A, B);
	}

};

frame_size dpm_gpu_structs::compute_dpm_psi_octo(out_frame out, in_frame A, in_frame B, in_frame C, in_frame D, const frame_size& size, const dpm_settings& dpm_settings, const dpm_bg_update_functor& functor, bool update_bg, int channel_idx)
{
	auto debug = false;
	write_debug_gpu(A, size.width, size.height, 1, "A.tif", debug);
	write_debug_gpu(B, size.width, size.height, 1, "B.tif", debug);
	write_debug_gpu(C, size.width, size.height, 1, "C.tif", debug);
	write_debug_gpu(D, size.width, size.height, 1, "D.tif", debug);
	//A is 90 pixel at 45 ilumination
	//B is 0 pixel at 45 ilumination
	//C is 90 pixel at 135 ilumination
	//D is 0 pixel at 135 ilumination

	auto max_side = std::max(dpm_settings.dpm_phase_width, dpm_settings.dpm_amp_width);
	dpm_double_demux(Y12, A, size, dpm_settings, false);
	dpm_double_demux(Y11, B, size, dpm_settings, false);
	dpm_double_demux(Y22, C, size, dpm_settings, false);
	dpm_double_demux(Y21, D, size, dpm_settings, false);

	//
	const auto debug_mode = write_debug_complex_mode::phase;
	write_debug_gpu_complex(Y11, max_side, max_side, 1, "Y11.tif", debug_mode, debug);
	write_debug_gpu_complex(Y21, max_side, max_side, 1, "Y21.tif", debug_mode, debug);
	write_debug_gpu_complex(Y12, max_side, max_side, 1, "Y12.tif", debug_mode, debug);
	write_debug_gpu_complex(Y22, max_side, max_side, 1, "Y22.tif", debug_mode, debug);
	//
	frame_size output(max_side * 2, max_side * 2);
	auto get_average = [](const thrust::device_vector<cuComplex>& in)
	{
		return thrust::reduce(in.begin(), in.end(), make_cuComplex(0, 0), add_complex()).x / in.size();
	};

	const auto c2 = get_average(Y21) + get_average(Y22);
	const auto c1 = get_average(Y11) + get_average(Y12);

	merge_quad_for_pol(out, q00, Y11, 1 * c1, Y21, 1 * c2, max_side, max_side);
	merge_quad_for_pol(out, q01, Y11, 1 * c1, Y21, -1 * c2, max_side, max_side);
	merge_quad_for_pol(out, q10, Y12, 1 * c1, Y22, 1 * c2, max_side, max_side);
	merge_quad_for_pol(out, q11, Y12, 1 * c1, Y22, -1 * c2, max_side, max_side);

	write_debug_gpu_complex(reinterpret_cast<const cuComplex*>(thrust::raw_pointer_cast(out.data())), 2 * max_side, 2 * max_side, 1, "octo_phase.tif", write_debug_complex_mode::phase, debug);
	write_debug_gpu_complex(reinterpret_cast<const cuComplex*>(thrust::raw_pointer_cast(out.data())), 2 * max_side, 2 * max_side, 1, "octo_real.tif", write_debug_complex_mode::real, debug);
#if _DEBUG
	if (out.size() != 2 * max_side * 2 * max_side * 2)
	{
		qli_runtime_error("Wrong Size");
	}
#endif
	return output;
}

void dpm_gpu_structs::pre_allocate_dpm_structs(const frame_size& output_size)
{
	//this function is incomplete becasuse it doesn't cover some of the more exotic modes but may be good enough for now
	const auto elements = output_size.n();
	auto dpm_out_temp_buffer_ptr = reinterpret_cast<float2*>(thrust_safe_get_pointer(dpm_in_d_, elements));
	big_ft_.take_ft(dpm_out_temp_buffer_ptr, dpm_out_temp_buffer_ptr, output_size.width, output_size.height, false);
	thrust_safe_resize(dpm_in_d_, 2 * elements);
}
