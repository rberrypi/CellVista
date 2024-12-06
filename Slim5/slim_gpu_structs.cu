#include "SLIM_GPU_Structs.h"
#include "thrust_resize.h"
#include "line_scale.h"
#include "write_debug_gpu.h"
#include "cuda_error_check.h"
#include "channel_settings.h"
// note these can be made faster by switching to float4
template<typename T>
__global__ void for_del_phi(
	const T* __restrict__ A, const T* __restrict__ B, const T* __restrict__ C, const T* __restrict__ D,
	const float Acomp, const float Bcomp, const float Ccomp, const float Dcomp,
	const float* out, const int numel)
{
	// Get our global thread ID
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	// Make sure we do not go out of bounds
	if (idx < numel)
	{
		const auto Gs = (D[idx] * Dcomp) - (B[idx] * Bcomp);
		const auto Gc = (A[idx] * Acomp) - (C[idx] * Ccomp);
		out[idx] = atan2f(Gs, Gc);
	}
}

template<typename T>
__global__ void Slim_Part1_src(
	const T* __restrict__ A, const T* __restrict__ B, const T* __restrict__ C, const T* __restrict__ D,
	float Acomp, float Bcomp, float Ccomp, float Dcomp,
	float* Beta, float* L, int numel)
{
	// Get our global thread ID
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	// Make sure we do not go out of bounds
	if (idx < numel)
	{
		const float a = A[idx] * Acomp; const float b = B[idx] * Bcomp; const float c = C[idx] * Ccomp; const float d = D[idx] * Dcomp;
		const auto gs = d - b;
		const auto gc = a - c;
		L[idx] = hypot(gs, gc);
		const auto t1 = a + c;
		const auto t2 = sqrt((4 * a * c) - gs * gs);
		const auto top = t1 - t2;
		const auto bot = t1 + t2;
		// HERE BE DRAGONS, you can try blocking the NaNs in other places but on some systems those blocks will optomize out, this is the safest place
		// SPECIFICALLY, this term should never be negative
		Beta[idx] = fmax(sqrt(top / bot), 0.0f);
	}
}

template<typename T>
__global__ void Slim_Part2_scale_src(const T* __restrict__ A, const T* __restrict__ B, const T* __restrict__ C, const T* __restrict__ D, const float acomp, const float bcomp, const float ccomp, const float dcomp, const float lat, const float disp, const float m, const bool thold, float* Phi, const int numel)
{
	// Get our global thread ID
	const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	// Make sure we do not go out of bounds
	if (idx < numel)
	{
		const float a = A[idx] * acomp; const float b = B[idx] * bcomp; const float c = C[idx] * ccomp; const float d = D[idx] * dcomp;
		const auto top = (d - b) * lat;
		const auto bot = (a - c) * lat + 1;
		const auto phase = disp + (atan2(top, bot) * m);
		Phi[idx] = ((thold) && (phase < 0)) ? 0.0f : phase;
	}
}

void slim_gpu_structs::compute_slim_phase(out_frame out, in_frame a, in_frame b, in_frame c, in_frame d, const channel_settings& settings, const bool update_bg, const int channel_idx, bool is_fpm)
{
	//Unbox
	static auto numel = 0;
	const auto size_changed = numel != a.size();
	numel = a.size();
	//
	//reintpret cases are redunant...
	auto ad = (thrust::raw_pointer_cast(is_fpm ? a.data() : d.data()));//360
	auto bd = (thrust::raw_pointer_cast(is_fpm ? b.data() : a.data()));//90 (aka phase contrast, because it has no modulator pattern)
	auto cd = (thrust::raw_pointer_cast(is_fpm ? c.data() : b.data()));//180
	auto dd = (thrust::raw_pointer_cast(is_fpm ? d.data() : c.data()));//270
#if _DEBUG
	if ((ad == nullptr) || (bd == nullptr) || (cd == nullptr) || (dd == nullptr))
	{
		qli_not_implemented();
	}
#endif
	auto extra_d = thrust_safe_get_pointer(extra_, numel);
	auto result_d = thrust_safe_get_pointer(out, numel);
	//
	const auto compensation = settings.get_compensations();
	const auto a_comp = compensation[0], b_comp = compensation[1], c_comp = compensation[2], d_comp = compensation[3];
	const static auto prefered_channel = 0;
	const auto skale = line_scale::compute_filter(settings, prefered_channel);
	const auto slim_bg_is_valid = settings.is_slim_bg_settings_valid();
	float slim_average_background_factor = 0;
	//std::cout << "compensation for ABCD: " << a_comp << "; " << b_comp << "; " << c_comp << "; " << d_comp << " ." << std::endl;
	if (slim_bg_is_valid)
	{
		slim_average_background_factor = settings.slim_bg_value;
	}
	else
	{
		static int gridSize, blockSize;
		if (size_changed)
		{
			int min_grid_size;//unused?
			CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &blockSize, Slim_Part1_src<unsigned short>, 0, 0));//todo bug here on the type!!!
			gridSize = (numel + blockSize - 1) / blockSize;
		}
		Slim_Part1_src << <gridSize, blockSize >> > (ad, bd, cd, dd, a_comp, b_comp, c_comp, d_comp, extra_d, result_d, numel);
		const auto bval = thrust::reduce(extra_.begin(), extra_.end(), 0.0);
		const auto lval = thrust::reduce(out.begin(), out.end(), 0.0);
		slim_average_background_factor = static_cast<float>(bval / lval);
	}
	//
	{
		static int gridSize, blockSize;
		if (size_changed)
		{
			int min_grid_size;//unused?
			CUDASAFECALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &blockSize, Slim_Part2_scale_src<unsigned short>, 0, 0));//todo bug here on the type!!!
			gridSize = (numel + blockSize - 1) / blockSize;
		}
		const auto attenuation_magic = (is_fpm) ? 1 : settings.objective_attenuation;
		const auto background_factor = attenuation_magic * slim_average_background_factor;
		Slim_Part2_scale_src << <gridSize, blockSize >> > (ad, bd, cd, dd, a_comp, b_comp, c_comp, d_comp, background_factor, skale.b, skale.m, skale.threshold, result_d, numel);
	}
	//
	if (slim_update && update_bg)
	{
		slim_bg_settings slim_bg(slim_average_background_factor);
		slim_update(slim_bg, channel_idx);
	}
	if (!slim_update && update_bg)
	{
		auto msg = "BG update called without a listening functor, this is most likely a bug";
		qli_runtime_error(msg);
	}
}
