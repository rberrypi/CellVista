//#include "stdafx.h"
#include "fourier_filter.h"
#include "write_debug_gpu.h"
#include "cuda_error_check.h"
#include "thrust_resize.h"
#include "cufft_shared.h"
#include <algorithm>
#include "cufft_error_check.h"
#include <functional>
//#include <numeric>

#ifndef M_PI
#define M_PI      (3.14159265358979323846 /* pi */)
#endif

template<typename T>
void ifft_shit_2D(std::vector<T>& data, const size_t xdim, const size_t ydim)
{
	//http://stackoverflow.com/questions/5915125/fftshift-ifftshift-c-c-source-code
	auto xshift = xdim / 2;
	if (xdim % 2 != 0)
	{
		xshift++;
	}
	auto yshift = ydim / 2;
	if (ydim % 2 != 0)
	{
		yshift++;
	}
	if ((xdim * ydim) % 2 != 0)
	{
		// temp output array
		std::vector<T > out;
		out.resize(xdim * ydim);
		for (auto x = 0; x < xdim; x++)
		{
			const auto out_x = (x + xshift) % xdim;
			for (auto y = 0; y < ydim; y++)
			{
				const auto out_y = (y + yshift) % ydim;
				// row-major order
				out[out_x + xdim * out_y] = data[x + xdim * y];
			}
		}
		// copy out back to data
		copy(out.begin(), out.end(), &data[0]);
	}
	else {
		// in and output array are the same,
		// values are exchanged using swap
		for (auto x = 0; x < xdim; x++)
		{
			const auto out_x = (x + xshift) % xdim;
			for (auto y = 0; y < yshift; y++)
			{
				const auto out_y = (y + yshift) % ydim;
				// row-major order
				std::swap(data[out_x + xdim * out_y], data[x + xdim * y]);
			}
		}
	}
}

class scale_by_constant
{//stolen from SO
	float c_;

public:
	explicit scale_by_constant(const float c) { c_ = c; };

	__host__ __device__ float operator()(float& a) const
	{
		const auto output = a * c_;
		return output;
	}

};

__global__ void _loadImageC1(cuComplex* dst, const float* src, int w_in, const int h_in, const int w_out, const int h_out)
{
	const int c = threadIdx.x + blockIdx.x * blockDim.x;
	const int r = threadIdx.y + blockIdx.y * blockDim.y;
	// Make sure we do not go out of bounds
	if ((r < h_out) && (c < w_out))
	{
		const auto c_pad = (w_out - w_in) / 2;
		const auto r_pad = (h_out - h_in) / 2;
		//
		auto r_new = r - r_pad;//todo this can be optomized
		const auto r_box = r_new / h_in;
		r_new = abs(r_new % h_in);
		if (!(r_box % 2 == 0))
		{
			r_new = h_in - r_new - 1;
		}
		//
		auto c_new = c - c_pad;//todo this can be optomized
		const auto c_box = c_new / w_in;
		c_new = abs(c_new % w_in);
		if (!(c_box % 2 == 0))
		{
			c_new = w_in - c_new - 1;
		}
		//
		assert(c_new < w_in);
		const auto in_idx = r_new * w_in + c_new;
		const auto out_idx = r * w_out + c;//I think there was a typo here...
		//
		dst[out_idx].x = src[in_idx];
		dst[out_idx].y = 0;
	}
}

__global__ void _loadImageC3(cuComplex* dst, const float* src, const int w_in, const  int h_in, const  int sample, const int w_out, const  int h_out)
{
	const auto sample_per_pixel = 3;
	const int c = threadIdx.x + blockIdx.x * blockDim.x;
	const int r = threadIdx.y + blockIdx.y * blockDim.y;
	// Make sure we do not go out of bounds
	if ((r < h_out) && (c < w_out))
	{
		const auto c_pad = (w_out - w_in) / 2;
		const auto r_pad = (h_out - h_in) / 2;
		//
		auto r_new = r - r_pad;//todo this can be optomized
		const auto r_box = r_new / h_in;
		r_new = abs(r_new % h_in);
		if (!(r_box % 2 == 0))
		{
			r_new = h_in - r_new - 1;
		}
		//
		auto c_new = c - c_pad;//todo this can be optomized
		auto c_box = c_new / w_in;
		c_new = abs(c_new % w_in);
		if (!(c_box % 2 == 0))
		{
			c_new = w_in - c_new - 1;
		}
		//
		//assert(c_new < w_in);
		const auto in_idx = sample_per_pixel * (r_new * w_in + c_new) + sample;
		const auto out_idx = r * w_out + c;//I think there was a typo here...
								  //
		dst[out_idx].x = src[in_idx];
		dst[out_idx].y = 0;
	}
}

void loadImage(thrust::device_vector<cuComplex>& img_big, out_frame src, const frame_size& frame_in, const  int pixel_sample, const  int samples_per_pixel, const frame_size& frame_out)
{
	auto dst_ptr = thrust_safe_get_pointer(img_big, frame_out.n());
	//
	dim3 bs2d(16, 16);//not sure if optimal
	dim3 gs2d;
	gs2d.x = static_cast<unsigned int>(ceil(frame_out.width / (1.f * bs2d.x)));
	gs2d.y = static_cast<unsigned int>(ceil(frame_out.height / (1.f * bs2d.y)));
	auto src_ptr = thrust::raw_pointer_cast(src.data());
	if (samples_per_pixel == 1)
	{
		_loadImageC1 << <gs2d, bs2d >> > (dst_ptr, src_ptr, frame_in.width, frame_in.height, frame_out.width, frame_out.height);
	}
	else
	{
		_loadImageC3 << <gs2d, bs2d >> > (dst_ptr, src_ptr, frame_in.width, frame_in.height, pixel_sample, frame_out.width, frame_out.height);
	}
}

__global__ void _getBackImageC1(float* dst, const cuComplex* src, const  int w_in, const int h_in, const int w_out, const int h_out)
{
	const int c = threadIdx.x + blockIdx.x * blockDim.x;
	const int r = threadIdx.y + blockIdx.y * blockDim.y;
	// Make sure we do not go out of bounds
	if ((r < h_out) && (c < w_out))
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
			dst[out_idx] = val.x;//This scaling is done at the output / (w_in*w_out);
			//dst[out_idx] = -1;
		}
	}
}

__global__ void _getBackImageC3(float* dst, int sample, const cuComplex* src, const int w_in, const int h_in, const int w_out, const  int h_out)
{
	const auto samples_per_pixel = 3;
	const int c = threadIdx.x + blockIdx.x * blockDim.x;
	const int r = threadIdx.y + blockIdx.y * blockDim.y;
	// Make sure we do not go out of bounds
	if ((r < h_out) && (c < w_out))
	{
		const auto c_pad = (w_in - w_out) / 2;
		const auto r_pad = (h_in - h_out) / 2;
		const auto r_new = r_pad + r;
		const auto c_new = c_pad + c;
		//
		if ((r_new < h_in) && (c_new < w_in))//dont think this can happen..
		{
			const auto in_idx = r_new * w_in + c_new;
			const auto out_idx = samples_per_pixel * (r * w_out + c) + sample;
			//
			auto val = src[in_idx];
			dst[out_idx] = val.x;//This scaling is done at the output / (w_in*w_out);
								 //dst[out_idx] = -1;
		}
	}
}

__global__ void _getBackImage_imagC1(float* dst, const cuComplex* src, const int w_in, const int h_in, const int w_out, const int h_out)
{
	const int c = threadIdx.x + blockIdx.x * blockDim.x;
	const int r = threadIdx.y + blockIdx.y * blockDim.y;
	// Make sure we do not go out of bounds
	if ((r < h_out) && (c < w_out))
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

void getBackImage(out_frame img, bool real_part, thrust::device_vector<cuComplex>& img_big, const frame_size& frame_in, const int pixel_sample, const  int samples_per_pixel, const frame_size& frame_out)
{
	//needs to be replaced with a cudamemcpy2D
	auto src_ptr = thrust::raw_pointer_cast(img_big.data());
	auto img_ptr = thrust::raw_pointer_cast(img.data());
	//
	dim3 bs2d(16, 16);//not sure if optimal
	dim3 gs2d;
	gs2d.x = static_cast<unsigned int>(ceil(frame_out.width / (1.f * bs2d.x)));
	gs2d.y = static_cast<unsigned int>(ceil(frame_out.height / (1.f * bs2d.y)));
	if (samples_per_pixel == 1)
	{
		if (real_part)
		{
			_getBackImageC1 << <gs2d, bs2d >> > (img_ptr, src_ptr, frame_in.width, frame_in.height, frame_out.width, frame_out.height);
		}
		else
		{
			_getBackImage_imagC1 << <gs2d, bs2d >> > (img_ptr, src_ptr, frame_in.width, frame_in.height, frame_out.width, frame_out.height);//An alternative is multiply by i or something
		}
	}
	else if (samples_per_pixel == 3)
	{
		if (real_part)
		{
			_getBackImageC3 << <gs2d, bs2d >> > (img_ptr, pixel_sample, src_ptr, frame_in.width, frame_in.height, frame_out.width, frame_out.height);
		}
		else
		{
			qli_not_implemented();
		}
	}
	else
	{
		qli_not_implemented();
	}
}

int pow2_roundup(int x)
{
	if (x < 0)
	{
		return 0;
	}
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x + 1;
}

struct inplace_mult
{
	// tell CUDA that the following code can be executed on the CPU and the GPU
	__host__ __device__ cuComplex operator()(const cuComplex& x, const float& y) const
	{
		return{ x.x * y, x.y * y };
	}
};

void fourier_filter::bandfilter_cpu(thrust::host_vector<float>& filter, const band_pass_settings& band, const frame_size& frame)
{
	if (band.do_band_pass == false)
	{
		return;
	}
	if (frame.width != frame.height)
	{
		qli_invalid_arguments();
	}
	const auto maxN = static_cast<int>(std::max(frame.width, frame.height));//tood make sure they are the same
	//
	const auto filterLargeC = 2.0f * band.max_dx / maxN;
	const auto filterSmallC = 2.0f * band.min_dx / maxN;
	const auto scaleLargeC = filterLargeC * filterLargeC;
	const auto scaleSmallC = filterSmallC * filterSmallC;
	//
	const auto filterLargeR = 2.0f * band.max_dy / maxN;
	const auto filterSmallR = 2.0f * band.min_dy / maxN;
	const auto scaleLargeR = filterLargeR * filterLargeR;
	const auto scaleSmallR = filterSmallR * filterSmallR;
	//float FactStripes;
	// loop over rows
	for (auto j = 1; j < maxN / 2; j++)
	{
		const auto row = j * maxN;
		const auto backrow = (maxN - j) * maxN;
		const auto rowFactLarge = exp(-(j * j) * scaleLargeR);
		const auto rowFactSmall = exp(-(j * j) * scaleSmallR);
		// loop over columns
		for (auto col = 1; col < maxN / 2; col++)
		{
			const auto backcol = maxN - col;
			const auto colFactLarge = exp(-(col * col) * scaleLargeC);
			const auto colFactSmall = exp(-(col * col) * scaleSmallC);
			const auto factor = (((1 - rowFactLarge * colFactLarge) * rowFactSmall * colFactSmall));
			filter[col + row] *= factor;
			filter[col + backrow] *= factor;
			filter[backcol + row] *= factor;
			filter[backcol + backrow] *= factor;
		}
	}
	const auto fixy = [&](float t) {return isinf(t) ? 0 : t; };
	const auto rowmid = maxN * (maxN / 2);
	auto rowFactLarge = fixy(exp(-(maxN / 2) * (maxN / 2) * scaleLargeR));
	auto rowFactSmall = fixy(exp(-(maxN / 2) * (maxN / 2) * scaleSmallR));
	//auto factStripes = exp(-(maxN / 2)*(maxN / 2) * scaleStripes);
	filter[maxN / 2] *= ((1 - rowFactLarge) * rowFactSmall); // (maxN/2,0)
	filter[rowmid] *= ((1 - rowFactLarge) * rowFactSmall); // (0,maxN/2)
	filter[maxN / 2 + rowmid] *= ((1 - rowFactLarge * rowFactLarge) * rowFactSmall * rowFactSmall); //
	//loop along row 0 and maxN/2	
	rowFactLarge = fixy(exp(-(maxN / 2) * (maxN / 2) * scaleLargeR));
	rowFactSmall = fixy(exp(-(maxN / 2) * (maxN / 2) * scaleSmallR));
	//dump();
	for (auto col = 1; col < maxN / 2; col++) {
		const auto backcol = maxN - col;
		const auto colFactLarge = exp(-(col * col) * scaleLargeC);
		const auto colFactSmall = exp(-(col * col) * scaleSmallC);
		filter[col] *= ((1 - colFactLarge) * colFactSmall);
		filter[backcol] *= ((1 - colFactLarge) * colFactSmall);
		filter[col + rowmid] *= ((1 - colFactLarge * rowFactLarge) * colFactSmall * rowFactSmall);
		filter[backcol + rowmid] *= ((1 - colFactLarge * rowFactLarge) * colFactSmall * rowFactSmall);
	}
	//dump();
	//SaveImage::write("second_filt.tif", w, h, filter.data());
	// loop along column 0 and maxN/2
	const auto colFactLarge = fixy(exp(-(maxN / 2) * (maxN / 2) * scaleLargeC));
	const auto colFactSmall = fixy(exp(-(maxN / 2) * (maxN / 2) * scaleSmallC));
	for (auto j = 1; j < maxN / 2; j++) {
		const auto row = j * maxN;
		const auto backrow = (maxN - j) * maxN;
		rowFactLarge = exp(-(j * j) * scaleLargeC);
		rowFactSmall = exp(-(j * j) * scaleSmallC);
		filter[row] *= ((1 - rowFactLarge) * rowFactSmall);
		filter[backrow] *= ((1 - rowFactLarge) * rowFactSmall);
		filter[row + maxN / 2] *= ((1 - rowFactLarge * colFactLarge) * rowFactSmall * colFactSmall);
		filter[backrow + maxN / 2] *= ((1 - rowFactLarge * colFactLarge) * rowFactSmall * colFactSmall);
	}
	filter[0] = (band.remove_dc) ? 0 : filter[0];
}

void fourier_filter::dic_filter_cpu(thrust::host_vector<float>& filter, const float shear_angle, const float coherence_length, const float pixel_ratio, const bool do_derivative, const frame_size& frame)
{
	//todo check this for off by one errors, etc
	const auto coherence_length_in_pixels = std::min(frame.height, frame.width) * (1 / pixel_ratio) / coherence_length;
	{
		dic_filter_cpu_temp_h.resize(frame.n(), 1);
		const auto degtorad = [](float deg) {return (acosf(-1) / 180) * deg; };
		const auto rads = degtorad(shear_angle);
		const auto n = frame.width;
		const auto m = frame.height;
		const auto tolerance = 0.001;
		const auto midx = ceilf(m / 2.0f);
		const auto midy = ceilf(n / 2.0f);
		const auto facto = 2 * M_PI / m;
		for (auto j = 0; j < m; j++)
		{
			for (auto i = 0; i < n; i++)
			{
				const auto idx = i + j * n;
				auto x = (i - midx) * cos(rads) + (j - midy) * sin(rads);
				// auto y = -(i - midx)*sin(rads) + (j - midy)*cos(rads);
				if (do_derivative)
				{

					x = std::min(x, coherence_length_in_pixels);
					dic_filter_cpu_temp_h[idx] = (x * facto) * ((x + tolerance) >= 0 ? 1 : -1);
				}
				else
				{
					dic_filter_cpu_temp_h[idx] = (x + tolerance) >= 0 ? 1 : -1;
				}
			}
		}
		ifft_shit_2D(dic_filter_cpu_temp_h, static_cast<size_t>(frame.width), static_cast<size_t>(frame.height));
	}
	std::transform(dic_filter_cpu_temp_h.begin(), dic_filter_cpu_temp_h.end(), filter.begin(), filter.begin(), std::multiplies<float>());
}

void fourier_filter::filter_gen_cpu(const bool force_regeneration, const phase_retrieval mode, const band_pass_settings& bandpass, const scope_compute_settings& qdic, const frame_size& frame)
{
	auto regen_filter = force_regeneration;
	regen_filter |= !(old_band_ == bandpass);
	old_band_ = bandpass;

	regen_filter |= (old_qdic_settings_ != qdic);
	old_qdic_settings_ = static_cast<qdic_scope_settings>(qdic);

	regen_filter |= (old_mode_ != mode);
	old_mode_ = mode;

	regen_filter |= (old_dimensions_ != qdic) && (mode == phase_retrieval::slim_demux);
	old_dimensions_ = static_cast<pixel_dimensions>(qdic);

	if (regen_filter)
	{
		std::cout << "Regenerating Filter" << std::endl;
		if (mode == phase_retrieval::slim_demux)
		{
			//prepare four directional demux filters
			for (auto i = 0; i < angles.size(); ++i)
			{
				const auto angle = angles[i];
				auto& filter_h = filters_h_[i];
				filter_h.assign(frame.n(), 1);
				if (angle >= 0)
				{
					dic_filter_cpu(filter_h, angle, qdic.coherence_length, qdic.pixel_ratio, true, frame);
				}
				if (bandpass.do_band_pass)
				{
					bandfilter_cpu(filter_h, bandpass, frame);
				}
				auto actual_fix = 1.0f / ((angle >= 0) ? 1 : frame.n());
				std::transform(filter_h.begin(), filter_h.end(), filter_h.begin(), std::bind1st(std::multiplies<float>(), actual_fix));
				auto& filter_g = filters_g_[i];
				thrust_safe_resize(filter_g, frame.n());
				thrust::copy(filter_h.begin(), filter_h.end(), filter_g.begin());
			}
		}
		else
		{
			auto& first_h = filters_h_[0];
			first_h.assign(frame.n(), 1);
			if (bandpass.do_band_pass)
			{
				bandfilter_cpu(first_h, bandpass, frame);
			}
			if (mode == phase_retrieval::glim_demux)
			{
				dic_filter_cpu(first_h, qdic.qsb_qdic_shear_angle, qdic.coherence_length, qdic.pixel_ratio, false, frame);
			}
			const auto fix_up = 1.0f / (frame.n());
			std::transform(first_h.begin(), first_h.end(), first_h.begin(), std::bind1st(std::multiplies<float>(), fix_up));
			auto& first_g = filters_g_[0];
			thrust_safe_resize(first_g, first_h.size());
			thrust::copy(first_h.begin(), first_h.end(), first_g.begin());
		}
	}
}

__global__ void _SLIMDemux_MergeNoCrop(cuComplex* dst, const cuComplex* __restrict__ src_A, const cuComplex* __restrict__ src_B, const cuComplex* __restrict__ src_C, const cuComplex* __restrict__ src_D, const  int numel)
{
	auto in_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (in_idx < numel)
	{
		auto val_a = src_A[in_idx], val_B = src_B[in_idx], val_C = src_C[in_idx], val_D = src_D[in_idx];
		auto final_value = thrust::max(thrust::max(val_a.x, val_B.x), thrust::max(val_C.x, val_D.x));
		dst[in_idx].x = final_value;
		dst[in_idx].y = 0;
	}
}

void SlimDemux_MergeNoCrop(thrust::device_vector<cuComplex>& output, const thrust::device_vector<cuComplex>& inA, const thrust::device_vector<cuComplex>& inB, const thrust::device_vector<cuComplex>& inC, const thrust::device_vector<cuComplex>& inD)
{
	const auto numel = inA.size();
	auto dst_ptr = thrust_safe_get_pointer(output, numel);
	auto srcA_ptr = thrust::raw_pointer_cast(inA.data());
	auto srcB_ptr = thrust::raw_pointer_cast(inB.data());
	auto srcC_ptr = thrust::raw_pointer_cast(inC.data());
	auto srcD_ptr = thrust::raw_pointer_cast(inD.data());
	//
	static int gridSize, blockSize;
	static auto old_size = 0;
	const auto size_changed = old_size != numel;
	old_size = numel;
	if (size_changed)
	{
		int minGridSize;//unused?
		CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, _SLIMDemux_MergeNoCrop, 0, 0));//todo bug here on the type!!!
		gridSize = (numel + blockSize - 1) / blockSize;
	}
	//
	_SLIMDemux_MergeNoCrop << <gridSize, blockSize >> > (dst_ptr, srcA_ptr, srcB_ptr, srcC_ptr, srcD_ptr, numel);
}

__global__ void _SlimDemux_CropAndMerge(float* dst, const cuComplex* __restrict__ src_A, const cuComplex* __restrict__ src_B, const cuComplex* __restrict__ src_C, const cuComplex* __restrict__ src_D, int w_in, int h_in, int w_out, int h_out)
{
	//here A must be the initial image
	int c = threadIdx.x + blockIdx.x * blockDim.x;
	int r = threadIdx.y + blockIdx.y * blockDim.y;
	// Make sure we do not go out of bounds
	if ((r < h_out) && (c < w_out))
	{
		auto c_pad = (w_in - w_out) / 2;
		auto r_pad = (h_in - h_out) / 2;
		auto r_new = r_pad + r;
		auto c_new = c_pad + c;
		//
		if ((r_new < h_in) && (c_new < w_in))//dont think this can happen..
		{
			const auto in_idx = r_new * w_in + c_new;
			const auto out_idx = r * w_out + c;
			//grabs the imaginary part, actuallt this is the real part...
			const auto val_A = src_A[in_idx], val_B = src_B[in_idx], val_C = src_C[in_idx], val_D = src_D[in_idx];
			const auto final_value = thrust::max(thrust::max(val_A.x, val_B.x), thrust::max(val_C.x, val_D.x));
			dst[out_idx] = final_value;//This scaling is done at the output / (w_in*w_out);
		}
	}
}

void SlimDemux_CropAndMerge(thrust::device_vector<float>& output, const frame_size& frame_out, const thrust::device_vector<cuComplex>& inA, const thrust::device_vector<cuComplex>& inB, const thrust::device_vector<cuComplex>& inC, const thrust::device_vector<cuComplex>& inD, const frame_size& frame_in)
{
	// ReSharper disable once CppLocalVariableMayBeConst
	auto dst_ptr = thrust::raw_pointer_cast(output.data());
	const auto srcA_ptr = thrust::raw_pointer_cast(inA.data());
	const auto srcB_ptr = thrust::raw_pointer_cast(inB.data());
	const auto srcC_ptr = thrust::raw_pointer_cast(inC.data());
	const auto srcD_ptr = thrust::raw_pointer_cast(inD.data());
	//
	dim3 bs2d(16, 16);//not sure if optimal
	dim3 gs2d;
	gs2d.x = static_cast<unsigned int>(ceil(frame_in.width / (1.f * bs2d.x)));
	gs2d.y = static_cast<unsigned int>(ceil(frame_in.height / (1.f * bs2d.y)));
	//
	_SlimDemux_CropAndMerge << <gs2d, bs2d >> > (dst_ptr, srcA_ptr, srcB_ptr, srcC_ptr, srcD_ptr, frame_in.width, frame_in.height, frame_out.width, frame_out.height);
}

struct divide_by_count
{
	// tell CUDA that the following code can be executed on the CPU and the GPU
	__host__ __device__ float operator()(const float& value, const int& count) const
	{
		return value / static_cast<float>(count);
	}
};

__global__ void RadialBinAverage(float* dst, const cuComplex* src, int* counter, float binspace, int full_width, int ft_width, int height, int ft_height)
{
	//Bullshit with a ton of atomic add for floats
	const int c = threadIdx.x + blockIdx.x * blockDim.x;
	const int r = threadIdx.y + blockIdx.y * blockDim.y;
	// Make sure we do not go out of bounds
	if ((r < height) && (c <= ft_width))
	{
		//auto c_fix = (c > ft_width) ? c - ft_width : c;//not needed
		const auto r_fix = (r > ft_height) ? height - r : r;
		const auto q_r = hypotf(r_fix, c);
		const auto bin = static_cast<int>(roundf(q_r / binspace));
		//
		auto bin_counter = counter + bin;
		atomicAdd(bin_counter, 1);
		auto dst_values = dst + bin;
		const auto c_value = src[c + r * full_width];
		const auto h = hypotf(c_value.x, c_value.y);
		auto src_value = log1p(h * h);
		atomicAdd(dst_values, src_value);
	}
}

void fourier_filter::lazyRadialAverage(thrust::device_vector<float>& output_lines, const thrust::device_vector<cuComplex>& input, int width_full, int height_full, int bins)
{
	// ReSharper disable once CppLocalVariableMayBeConst
	auto output_ptr = thrust_safe_get_pointer(output_lines, bins);
	thrust::fill(output_lines.begin(), output_lines.end(), 0.0f);
	const auto input_ptr = thrust::raw_pointer_cast(input.data());
	dim3 bs2d(16, 16);//not sure if optimal
	dim3 gs2d;
	const auto width_half = (width_full / 2);
	const auto height_half = (height_full / 2);
	gs2d.x = static_cast<unsigned int>(ceil(width_half / (1.f * bs2d.x)));
	gs2d.y = static_cast<unsigned int>(ceil(height_full / (1.f * bs2d.y)));
	// ReSharper disable once CppLocalVariableMayBeConst
	auto counter_ptr = thrust_safe_get_pointer(counter, bins);
	thrust::fill(counter.begin(), counter.end(), 0);
	const auto binspace = static_cast<float>((hypotf(static_cast<float>(width_half), static_cast<float>(height_half)) / (bins - 1)));
	RadialBinAverage << <gs2d, bs2d >> > (output_ptr, input_ptr, counter_ptr, binspace, width_full, width_half, height_full, height_half);
	thrust::transform(output_lines.begin(), output_lines.end(), counter.begin(), output_lines.begin(), divide_by_count());
}

void fourier_filter::do_filter(camera_frame_internal frame, const phase_retrieval a_la_mode, const scope_compute_settings& qdic, const band_pass_settings& band)
{
#if 0
	static auto calls = 0;
	calls = calls + 1;
	auto force_debuggle = calls == 5;
#else
	const auto force_debuggle = false;
#endif
	const auto mode = phase_processing_setting::settings.at(frame.processing).is_raw_mode ? phase_retrieval::camera : a_la_mode;
	if (frame.is_valid() && ( band.do_band_pass || (mode == phase_retrieval::glim_demux) || (mode == phase_retrieval::slim_demux)))
	{
		const auto regen_ft = (old_size_ != frame);
		old_size_ = static_cast<frame_size>(frame);
		const auto maxE = std::max(frame.width, frame.height);
		max_n_ = pow2_roundup(1.5 * maxE);
		for (auto pixel_sample = 0; pixel_sample < frame.samples_per_pixel; ++pixel_sample)
		{
			write_debug_gpu(*frame.buffer, frame.width, frame.height, 1, "input_image.tif", force_debuggle);
			const auto big_image_size = frame_size(max_n_, max_n_);
			filter_gen_cpu(regen_ft, mode, band, qdic, big_image_size);
			auto& first_big_image = big_imgs_[0];
			loadImage(first_big_image, *frame.buffer, frame, pixel_sample, frame.samples_per_pixel, big_image_size);
			write_debug_gpu_complex(big_imgs_.front(), max_n_, max_n_, 1, "BigImage.tif", write_debug_complex_mode::real, force_debuggle);
			// ReSharper disable once CppLocalVariableMayBeConst
			plan_.take_ft(first_big_image, first_big_image, max_n_, max_n_, true);
			auto& first_filter = filters_g_[0];
			thrust::transform(first_big_image.begin(), first_big_image.end(), first_filter.begin(), first_big_image.begin(), inplace_mult());
			write_debug_gpu(first_filter, max_n_, max_n_, frame.samples_per_pixel, "FirstFilter.tif", force_debuggle);
			const auto max_aux_angles = (mode == phase_retrieval::slim_demux) ? 4 : 1;
			for (auto i = 1; i < max_aux_angles; i++)
			{
				auto& big_img = big_imgs_[i];
				auto& filter = filters_g_[i];
				thrust_safe_resize(big_img, first_big_image.size());
				thrust::copy(first_big_image.begin(), first_big_image.end(), big_img.begin());
				{
					auto test_name = "FT_Input_" + std::to_string(i) + ".tif";
					write_debug_gpu_complex(big_img, max_n_, max_n_, frame.samples_per_pixel, test_name.c_str(), write_debug_complex_mode::absolute, force_debuggle);
				}
				thrust::transform(big_img.begin(), big_img.end(), filter.begin(), big_img.begin(), inplace_mult());
				{
					auto test_name = "FT_Filter_" + std::to_string(i) + ".tif";
					write_debug_gpu(filter, max_n_, max_n_, frame.samples_per_pixel, test_name.c_str());
				}
				{
					auto test_name = "FT_Output_" + std::to_string(i) + ".tif";
					write_debug_gpu_complex(big_img, max_n_, max_n_, frame.samples_per_pixel, test_name.c_str(), write_debug_complex_mode::absolute, force_debuggle);
				}
				plan_.take_ft(big_img, big_img, max_n_, max_n_, false);
			}
			
			write_debug_gpu_complex(big_imgs_[0], max_n_, max_n_, frame.samples_per_pixel, "GLIM_DEMUX_ft.tif", write_debug_complex_mode::log_one_pee, force_debuggle);
			plan_.take_ft(first_big_image, first_big_image, max_n_, max_n_, false);
			
			if (mode == phase_retrieval::slim_demux)
			{
				if (frame.samples_per_pixel != 1)
				{
					qli_invalid_arguments();
				}
				
				write_debug_gpu_complex(big_imgs_[0], max_n_, max_n_, frame.samples_per_pixel, "SLIMDemux_Output0.tif", write_debug_complex_mode::real, force_debuggle);
				
				write_debug_gpu_complex(big_imgs_[1], max_n_, max_n_, frame.samples_per_pixel, "SLIMDemux_Output1.tif", write_debug_complex_mode::real, force_debuggle);
				write_debug_gpu_complex(big_imgs_[2], max_n_, max_n_, frame.samples_per_pixel, "SLIMDemux_Output2.tif", write_debug_complex_mode::real, force_debuggle);
				write_debug_gpu_complex(big_imgs_[3], max_n_, max_n_, frame.samples_per_pixel, "SLIMDemux_Output3.tif", write_debug_complex_mode::real, force_debuggle);
				write_debug_gpu(*frame.buffer, frame.width, frame.height, frame.samples_per_pixel, "SLIMDemux_Input.tif", force_debuggle);
				SlimDemux_CropAndMerge(*frame.buffer, frame, big_imgs_[0], big_imgs_[1], big_imgs_[2], big_imgs_[3], big_image_size);
				
				write_debug_gpu(*frame.buffer, frame.width, frame.height, frame.samples_per_pixel, "SLIMDemux_After.tif", force_debuggle);
				
			}
			else
			{
				const auto is_real = (phase_retrieval::glim_demux != mode);
				getBackImage(*frame.buffer, is_real, first_big_image, big_image_size, pixel_sample, frame.samples_per_pixel, frame);//scale here
				write_debug_gpu_complex(big_imgs_[0], max_n_, max_n_, frame.samples_per_pixel, "GLIM_DEMUX_img.tif", write_debug_complex_mode::imaginary, force_debuggle);
				write_debug_gpu_complex(big_imgs_[0], max_n_, max_n_, frame.samples_per_pixel, "GLIM_DEMUX_real.tif", write_debug_complex_mode::real, force_debuggle);
			}
			
		}
	}
}

void fourier_filter::pre_allocate_fourier_filters(const frame_size& output_size)
{
	const auto maxE = std::max(output_size.width, output_size.height);
	max_n_ = pow2_roundup(1.5 * maxE);
	const auto big_image_size = frame_size(max_n_, max_n_).n();
	for (auto& frame : big_imgs_)
	{
		thrust_safe_resize(frame, big_image_size);
	}
	for (auto& frame : filters_g_)
	{
		thrust_safe_resize(frame, big_image_size);
	}
}