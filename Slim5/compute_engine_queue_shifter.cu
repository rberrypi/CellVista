#include "compute_engine.h"
#include "thrust_resize.h"
#include "npp_error_check.h"
camera_frame_internal compute_engine::get_shifted_data(const camera_frame_internal& phase, const render_shifter& shifter)
{
	const auto do_shift = shifter.do_shift();
	if (!phase.is_valid() || !do_shift)
	{
		return phase;
	}
	if (phase.is_complex() || phase.samples_per_pixel != 1)
	{
		qli_not_implemented();
	}
	const auto output_elements = phase.n();
	{
		auto& img_out = *shifter_buffer_ptr;
		//
		auto& img_in = *phase.buffer;
		//
		{
			/*
			 const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
			 Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
			 int nTopBorderHeight, int nLeftBorderWidth,
			 Npp32s nValue
			 */
			float* pSrc = thrust::raw_pointer_cast(img_in.data());
			const auto nSrcStep = phase.width * sizeof(float);
			const NppiSize oSrcSizeROI = { phase.width,phase.height };
			float* pDst = thrust_safe_get_pointer(img_out, phase.n());
			const auto nDstStep = phase.width * sizeof(float);
			const NppiSize oDstSizeROI = { phase.width,phase.height };
			int nTopBorderHeight = shifter.ty;
			int nLeftBorderWidth = shifter.tx;
			const float nValue = 0;
			NPP_SAFE_CALL(nppiCopyConstBorder_32f_C1R(
				pSrc, nSrcStep, oSrcSizeROI,
				pDst, nDstStep, oDstSizeROI,
				nTopBorderHeight, nLeftBorderWidth,
				nValue));
		}
		//
	}
	const camera_frame_internal return_me(shifter_buffer_ptr, { phase, phase });
	shifter_buffer_ptr = phase.buffer;
	return return_me;
}