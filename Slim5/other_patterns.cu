#include "other_patterns.h"
#include "thrust_resize.h"
#include "line_scale.h"
#include "write_debug_gpu.h"
#include "cuda_error_check.h"
#include "cuda_runtime.h"
#include "channel_settings.h"
//todo convert these guys to 1D transforms

template<typename T>
__global__ void forIntensity(const T A, const T B, const T C, const T D,
	const float Acomp, const float Bcomp, const float Ccomp, const float Dcomp,
	const float* out, const int numel)
{
	const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numel)
	{
		const auto a = A[idx] * Acomp, b = B[idx] * Bcomp, c = C[idx] * Ccomp, d = D[idx] * Dcomp;
		const auto a2 = (b - d) / 2;
		const auto a1 = (a - c) / 2;
		const auto a0 = (a + b + c + d) / 4;
		const auto inte = hypot(a1, a2) / a0;
		out[idx] = inte;
	}
}

template<typename T>
__global__ void forDel_Phi(
	const T A, const T B, const T C, const T D,
	float Acomp, float Bcomp, float Ccomp, float Dcomp,
	const float* out, const int numel)
{
	const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numel)
	{
		const auto Gs = (D[idx] * Dcomp) - (B[idx] * Bcomp);
		const auto Gc = (A[idx] * Acomp) - (C[idx] * Ccomp);
		out[idx] = atan2f(Gs, Gc);
	}
}

//todo replace with 1D
template<typename T>
__global__ void forDel_Phi_with_scale(
	const T* A, const T* B, const T* C, const T* D,
	float Acomp, float Bcomp, float Ccomp, float Dcomp,
	float* out, const int numel, const line_scale scale)
{
	// Get our global thread ID
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numel)
	{
		const auto Gs = (D[idx] * Dcomp) - (B[idx] * Bcomp);
		const auto Gc = (A[idx] * Acomp) - (C[idx] * Ccomp);
		const auto phase = scale.b + (atan2f(Gs, Gc) * scale.m);
		out[idx] = ((scale.threshold) && (phase < 0)) ? 0.0f : phase;;
	}
}

template<typename T>
struct frame_package
{
	const T* img;
	float comp, top_weight, bot_weight, const_weight;
};

template<typename T>
__global__ void four_frame_psi_c1(
	const frame_package<T> A, const frame_package<T> B, const frame_package<T> C, const frame_package<T> D,
	float* out, float inv_scale, const int numel)
{
	// Get our global thread ID
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numel)
	{
		const auto top = (A.img[idx] * A.top_weight) + (B.img[idx] * B.top_weight) + (C.img[idx] * C.top_weight) + (D.img[idx] * D.top_weight);
		const auto bot = (A.img[idx] * A.bot_weight) + (B.img[idx] * B.bot_weight) + (C.img[idx] * C.bot_weight) + (D.img[idx] * D.bot_weight);
		const auto phase = atan2f(top, bot) * inv_scale;
		out[idx] = phase;
	}
}

template<typename T >
__global__ void four_frame_nonint_c1(
	const frame_package<T> A, const frame_package<T> B, const frame_package<T> C, const frame_package<T> D,
	float* out, const int numel)
{
	// Get our global thread ID
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numel)
	{
		const auto a = A.img[idx]; const auto b = B.img[idx]; const auto c = C.img[idx]; const auto d = D.img[idx];
		const auto top = (a * A.top_weight) + (b * B.top_weight) + (c * C.top_weight) + (d * D.top_weight);
		const auto bot = (a * A.bot_weight) + (b * B.bot_weight) + (c * C.bot_weight) + (d * D.bot_weight);
		const auto constant = (a * A.const_weight) + (b * B.const_weight) + (c * C.const_weight) + (d * D.const_weight);
		const auto phase = atan2f(top, bot);
		const auto gamma = hypotf(top, bot) / constant;
		const auto sum = a + b + c + d;
		out[idx] = (sum) / (1 + cosf(phase) * gamma);
	}
}

template<typename T, int C>
struct FramePackageC
{
	const T* __restrict__ img;
	float comp, top[C], bot[C];
};

template<typename T, int frames, int channels>
struct frame_package_holder
{
	FramePackageC<T, channels> packages[frames];
	//explicit frame_package_holder(FramePackageC<T, channels> Packages[4]) : packages(Packages){};
};

template<typename T, int frames, int channels>
__global__ void four_frame_psi_c3(float* out, int numel, const frame_package_holder<T, frames, channels> four_frame_psi_c3_packages, float inv_scale)
{
	//todo figure out if doing a thre channel pixel each instance is optimal
	// Get our global thread ID
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;//does three colors at once?
	if (idx < numel)
	{
		for (auto c = 0; c < channels; ++c)
		{
			auto out_idx = channels * idx + c;
			float top = 0, bot = 0;//this assign can be avoided if we loop unrolled harder, I think
			for (auto i = 0; i < frames; ++i)
			{
				auto value = four_frame_psi_c3_packages.packages[i].img[out_idx];
				top += value * four_frame_psi_c3_packages.packages[i].top[c];
				bot += value * four_frame_psi_c3_packages.packages[i].bot[c];
			}
			const auto phase = atan2f(top, bot) * inv_scale;
			out[out_idx] = phase;
		}
	}
}

template<typename T, int frames, int channels>
__global__ void four_frame_nonint_c3(float* out, int numel, const frame_package_holder<T, frames, channels> four_frame_psi_c3_packages)
{
	/*
	//todo figure out if doing a thre channel pixel each instance is optimal
	// Get our global thread ID
	auto idx = threadIdx.x + blockIdx.x*blockDim.x;//does three colors at once?
	if (idx < numel)
	{
		for (auto c = 0; c < channels; ++c)
		{
			auto out_idx = channels * idx + c;
			float top = 0, bot = 0, constant;//this assign can be avoided if we loop unrolled harder, I think
			for (auto i = 0; i < frames; ++i)
			{
				auto value = four_frame_psi_c3_packages.packages[i].img[out_idx];
				top += value*four_frame_psi_c3_packages.packages[i].top[c];
				bot += value*four_frame_psi_c3_packages.packages[i].bot[c];
				constant += value*four_frame_psi_c3_packages.packages[i].
			}

			 * 		const auto constant = (a * A.const_weight) + (b * B.const_weight) + (c * C.const_weight) + (d * D.const_weight);
		const auto phase = atan2f(top, bot);
		const auto gamma = hypotf(top, bot) / constant;
		const auto sum = a + b + c + d;
		out[idx] = (sum) / (1 + cosf(phase)*gamma);

			const auto phase = atan2f(top, bot);
			out[out_idx] = phase;
		}
	}
	*/
}


void other_patterns::ac_dc_c1(out_frame out, in_frame A, in_frame B, in_frame C, in_frame D, const channel_settings& in)
{
	//todo convert to 1D version
	const auto numel = A.size();
	static int grid_size, block_size;
	static auto old_size = (-1);
	const auto size_changed = (numel != old_size);
	if (size_changed)
	{
		int min_grid_size;//unused?
		CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, four_frame_psi_c1<unsigned short>, 0, 0));//todo bug here on the type!!!
		grid_size = (numel + block_size - 1) / block_size;
	}
	const auto ad = thrust::raw_pointer_cast(A.data());
	const auto bd = thrust::raw_pointer_cast(B.data());
	const auto cd = thrust::raw_pointer_cast(C.data());
	const auto dd = thrust::raw_pointer_cast(D.data());
	const auto result_d = thrust_safe_get_pointer(out, numel);
	const auto comps = in.get_compensations();
	//const auto a_comp = comps[0], b_comp = comps[1], c_comp = comps[2], d_comp = comps[3];
	const auto ch = 0;
	const auto& patterns = in.modulator_settings.begin()->patterns;
	{
		const auto items = patterns.size();
		if (items < 4)
		{
			qli_runtime_error("Not enough weights");
		}
	}
	std::array<frame_package<unsigned short>, 4> F = { {
		{ ad, comps[0], patterns[0].weights[ch].top,patterns[0].weights[ch].bot, patterns[0].weights[ch].constant },
		{ bd, comps[1], patterns[1].weights[ch].top,patterns[1].weights[ch].bot, patterns[1].weights[ch].constant },
		{ cd, comps[2], patterns[2].weights[ch].top,patterns[2].weights[ch].bot, patterns[2].weights[ch].constant },
		{ dd, comps[3], patterns[3].weights[ch].top,patterns[3].weights[ch].bot, patterns[3].weights[ch].constant }
		} };
	const auto scale = line_scale::compute_qdic(in.qsb_qdic_shear_dx);
	four_frame_psi_c1 << <grid_size, block_size >> > (F[0], F[1], F[2], F[3], result_d, scale.m, numel);
}

void other_patterns::non_int_c1(out_frame out, in_frame A, in_frame B, in_frame C, in_frame D, const channel_settings& in)
{
		//todo convert to 1D version
		const auto numel = A.size();
		static int grid_size, blockSize;
		static auto old_size = (-1);
		const auto size_changed = (numel != old_size);
		if (size_changed)
		{
			int min_grid_size;//unused?
			CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &blockSize, four_frame_psi_c1<unsigned short>, 0, 0));//todo bug here on the type!!!
			grid_size = (numel + blockSize - 1) / blockSize;
		}
		const auto ad = thrust::raw_pointer_cast(A.data());
		const auto bd = thrust::raw_pointer_cast(B.data());
		const auto cd = thrust::raw_pointer_cast(C.data());
		const auto dd = thrust::raw_pointer_cast(D.data());
		const auto result_d = thrust_safe_get_pointer(out, numel);
		const auto comps = in.get_compensations();
		//const auto a_comp = comps[0], b_comp = comps[1], c_comp = comps[2], d_comp = comps[3];
		const auto ch = 0;
	const auto& patterns = in.modulator_settings.begin()->patterns;
		std::array<frame_package<unsigned short>, 4> f = { {
			{ ad, comps[0], patterns[0].weights[ch].top,patterns[0].weights[ch].bot, patterns[0].weights[ch].constant },
			{ bd, comps[1], patterns[1].weights[ch].top,patterns[1].weights[ch].bot, patterns[1].weights[ch].constant },
			{ cd, comps[2], patterns[2].weights[ch].top,patterns[2].weights[ch].bot, patterns[2].weights[ch].constant },
			{ dd, comps[3], patterns[3].weights[ch].top,patterns[3].weights[ch].bot, patterns[3].weights[ch].constant }
			} };
		four_frame_nonint_c1 << <grid_size, blockSize >> > (f[0], f[1], f[2], f[3], result_d, numel);
}

void other_patterns::non_int_c3(out_frame out, in_frame A, in_frame B, in_frame C, in_frame D, const channel_settings& in)
{
	std::cout << "Warning this mode isn't implemented so we're putitng in a shim for the teset cases" << std::endl;
	ac_dc_c3(out, A, B, C, D, in);
}

void other_patterns::ac_dc_c3(out_frame out, in_frame A, in_frame B, in_frame C, in_frame D, const channel_settings& in)
{
	//todo convert to 1D version
	const auto numel = A.size() / 3;
	static int gridSize, block_size;
	static auto old_size = (-1);
	const auto size_changed = (numel != old_size);
	const auto scale = line_scale::compute_qdic(in.qsb_qdic_shear_dx);
	if (size_changed)
	{
		int minGridSize;//unused?
		CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &block_size, four_frame_psi_c3<unsigned short, 4, 3>, 0, 0));//todo bug here on the type!!!
		gridSize = (numel + block_size - 1) / block_size;
	}
	const auto ad = thrust::raw_pointer_cast(A.data());
	const auto bd = thrust::raw_pointer_cast(B.data());
	const auto cd = thrust::raw_pointer_cast(C.data());
	const auto dd = thrust::raw_pointer_cast(D.data());
	const auto result_d = thrust_safe_get_pointer(out, A.size());
	const auto comps = in.get_compensations();
	const auto a_comp = comps[0], BComp = comps[1], CComp = comps[2], DComp = comps[3];
	//cudaMemcpytoSymbol
	const auto& patterns = in.modulator_settings.begin()->patterns;
	FramePackageC<unsigned short, 3> four_frame_package_h[4] = {
		{ ad,a_comp,{ patterns[0].weights[0].top,patterns[0].weights[1].top,patterns[0].weights[2].top },{ patterns[0].weights[0].bot,patterns[0].weights[1].bot,patterns[0].weights[2].bot } },
		{ bd,BComp,{ patterns[1].weights[0].top,patterns[1].weights[1].top,patterns[1].weights[2].top },{ patterns[1].weights[0].bot,patterns[1].weights[1].bot,patterns[1].weights[2].bot } },
		{ cd,CComp,{ patterns[2].weights[0].top,patterns[2].weights[1].top,patterns[2].weights[2].top },{ patterns[2].weights[0].bot,patterns[2].weights[1].bot,patterns[2].weights[2].bot } },
		{ dd,DComp,{ patterns[3].weights[0].top,patterns[3].weights[1].top,patterns[3].weights[2].top },{ patterns[3].weights[0].bot,patterns[3].weights[1].bot,patterns[3].weights[2].bot } }
	};
	//auto size_bytes = sizeof(four_frame_package_h);
	//CUDASAFECALL( cudaMemcpyToSymbol(four_frame_psi_c3_packages, four_frame_package_h, size_bytes));
	frame_package_holder<unsigned short, 4, 3> holder;
	std::copy(four_frame_package_h, four_frame_package_h + 4, holder.packages);
	four_frame_psi_c3<unsigned short, 4, 3> << <gridSize, block_size >> > (result_d, numel, holder, scale.m);
	CUDASAFECALL(cudaDeviceSynchronize());
}


void other_patterns::ac_dc(out_frame out, in_frame A, in_frame B, in_frame  C, in_frame D, const channel_settings& in, const int samples_per_pixel)
{
	const auto processing = in.processing;
	if (processing == phase_processing::phase)
	{
		const auto processor = (samples_per_pixel == 3) ? ac_dc_c3 : ac_dc_c1;
		processor(out, A, B, C, D, in);
	}
	else if (processing == phase_processing::non_interferometric)
	{
		const auto processor = (samples_per_pixel == 3) ? non_int_c3 : non_int_c1;
		processor(out, A, B, C, D, in);
	}
	else if (processing == phase_processing::mutual_intensity)
	{
		for_intensity(out, A, B, C, D, in);
	}
	else
	{
		qli_not_implemented();
	}
}

template<typename T>
__global__ void _forIntensity(const T* A, const T* B, const T* C, const T* D,
	const float Acomp, const float Bcomp, const float Ccomp, const float Dcomp,
	float* out, const int numel)
{
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numel)
	{
		const float a = A[idx] * Acomp, b = B[idx] * Bcomp, c = C[idx] * Ccomp, d = D[idx] * Dcomp;
		const auto a2 = (b - d);
		const auto a1 = (a - c);
		const auto a0 = (a + b + c + d);
		const auto inte = hypot(a1, a2) / a0;
		out[idx] = inte;
	}
}

void other_patterns::for_intensity(out_frame out, in_frame A, in_frame B, in_frame C, in_frame D, const channel_settings& in)
{
	const auto numel = A.size();
	static int gridSize, blockSize;
	static auto old_size = (-1);
	const auto size_changed = (numel != old_size);
	if (size_changed)
	{
		int minGridSize;//unused?
		CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, _forIntensity<unsigned short>, 0, 0));//todo bug here on the type!!!
		gridSize = (numel + blockSize - 1) / blockSize;
	}
	const auto ad = thrust::raw_pointer_cast(A.data());
	const auto bd = thrust::raw_pointer_cast(B.data());
	const auto cd = thrust::raw_pointer_cast(C.data());
	const auto dd = thrust::raw_pointer_cast(D.data());
	const auto result_d = thrust_safe_get_pointer(out, numel);
	auto comps = in.get_compensations();
	const auto a_comp = comps[0], b_comp = comps[1], c_comp = comps[2], d_comp = comps[3];
	_forIntensity << <gridSize, blockSize >> > (ad, bd, cd, dd, a_comp, b_comp, c_comp, d_comp, result_d, numel);
}

struct _PassThru
{
	__host__ __device__ float operator()(const unsigned short in) const
	{
		return in;
	}
};

void other_patterns::pass_thru(out_frame out, const in_frame A)
{
	//but also maybe thrust copy?
	const auto numel = A.size();
#if _DEBUG
	if (numel == 0)
	{
		qli_not_implemented();
	}
#endif
	thrust_safe_resize(out, numel);
	thrust::transform(A.begin(), A.end(), out.begin(), _PassThru());
}