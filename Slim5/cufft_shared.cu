#include "stdafx.h"
#include "cufft_shared.h"
#include "thrust_resize.h"
#include "cufft_error_check.h"

void cufft_wrapper::take_ft(thrust::device_vector<float2>& input, thrust::device_vector<float2>& out, int width, int height, bool is_forward)
{
	const auto elements = width * height;
	if (input.size() != elements)
	{
		qli_invalid_arguments();
	}
	auto input_ptr = thrust::raw_pointer_cast(input.data());
	auto output_ptr = thrust_safe_get_pointer(out, elements);
	take_ft(input_ptr, output_ptr, width, height, is_forward);
}

void cufft_wrapper::take_ft(float2* input, float2* out, int width, int height, bool is_forward)
{
	take_ft(input, out, ft_settings(width, height), is_forward);
}

void cufft_wrapper::take_ft(float2* input, float2* out, const ft_settings& settings, bool is_forward)
{
	CUDA_DEBUG_SYNC();
	auto changed_dimensions = settings != static_cast<ft_settings&>(*this);
	static_cast<ft_settings&>(*this) = settings;
	if (changed_dimensions)
	{
		free_handle();
		if (is_simple_case())
		{
			CUFFT_SAFE_CALL(cufftPlan2d(&id_, height, width, CUFFT_C2C));
		}
		else
		{
			int n[2] = { settings.height, settings.width };
			int inembed[2] = { settings.src.pitch_height_numel, settings.src.pitch_width_numel };
			int onembed[2] = { settings.dst.pitch_height_numel,settings.dst.pitch_width_numel };
			int istride = 1, ostride = 1;
			int idist = settings.src.pitch_height_numel * settings.src.pitch_width_numel, odist = settings.dst.pitch_height_numel * settings.dst.pitch_width_numel;
			CUFFT_SAFE_CALL(cufftPlanMany(&id_, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, 1));
		}
		is_initialized_=true;
	}
	auto shift_ptr = [](auto ptr, const ft_rectangle& rect)
	{
		return ptr + rect.left + rect.pitch_width_numel * rect.top;
	};
	auto direction = is_forward ? CUFFT_FORWARD : CUFFT_INVERSE;
	CUFFT_SAFE_CALL(cufftExecC2C(id_, shift_ptr(input, settings.src), shift_ptr(out, settings.dst), direction));
	CUDA_DEBUG_SYNC();
}

void cufft_wrapper::free_handle()
{
	if (is_initialized_)
	{
		CUFFT_SAFE_CALL(cufftDestroy(id_));
	}
}

cufft_wrapper::~cufft_wrapper()
{
	free_handle();
}
