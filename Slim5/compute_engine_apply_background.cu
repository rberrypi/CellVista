#include "compute_engine.h"
#include "thrust_resize.h"
#include "cuda_runtime_api.h"
#include "channel_settings.h"
#include "qli_runtime_error.h"

struct SingleAngle
{
	__host__ __device__ float operator()(const cuFloatComplex& a) const
	{
		const auto angle = atan2f(cuCimagf(a), cuCrealf(a));
		return angle;
	}
};

struct AngleWithBackground
{
	__host__ __device__ float operator()(const cuFloatComplex& a, const cuFloatComplex& b) const
	{
		//safely ignore the amplitude
		auto rx = (a.x * b.x + a.y * b.y);
		auto ry = (a.y * b.x - a.x * b.y);
		return atan2f(ry, rx);
	}
};


void complex_bg_merge(thrust::device_vector<float>& img_out, const thrust::device_vector<float>& img_in, background_frame* bg_buffer)
{
	auto size = img_in.size() / 2;
	thrust_safe_resize(img_out, size);
	auto in_elements = thrust::device_pointer_cast(reinterpret_cast<const cuFloatComplex*>(thrust::raw_pointer_cast(img_in.data())));
	if (bg_buffer == nullptr)
	{
		thrust::transform(in_elements, in_elements + size, img_out.begin(), SingleAngle());
	}
	else
	{
		const auto image_data_ptr = thrust::raw_pointer_cast(bg_buffer->buffer.data());
		auto buffer_as_ptr = thrust::device_pointer_cast(reinterpret_cast<const cuFloatComplex*>(image_data_ptr));
		thrust::transform(in_elements, in_elements + size, buffer_as_ptr, img_out.begin(), AngleWithBackground());
	}
}



struct bg_subtract_functor
{
	__host__ __device__ float operator()(const float& a, const float& b) const
	{
		return a - b;
	}
};

camera_frame_internal compute_engine::apply_background_and_decomplexify(camera_frame_internal& phase, const channel_settings& settings, const live_compute_options& processing_options)
{
	/*
#if _DEBUG
	if (processing_options.show_mode == live_compute_options::background_show_mode::show_bg)
	{
		qli_runtime_error("Should have been returned at an earlier step in the pipeline, for example immediately");
	}
#endif
*/
	if (!phase.is_valid())
	{
		return camera_frame_internal();
	}
	//Step 1 : For regular return the image
	if (processing_options.show_mode == live_compute_options::background_show_mode::regular)
	{
		if (phase.is_complex())
		{
			complex_bg_merge(*decomplexify_buffer_ptr, *phase.buffer, nullptr);
			camera_frame_internal return_me(decomplexify_buffer_ptr, { phase, phase });
			decomplexify_buffer_ptr = phase.buffer;
			return_me.complexity = image_info::complex::no;
			return return_me;
		}
		return phase;
	}
	//Step 2: for set_bg set the bg (first one is zero)
	if (processing_options.show_mode == live_compute_options::background_show_mode::set_bg)
	{
		//change not guaranteeds as const channel_settings
		phase_update(phase);
		return camera_frame_internal();
	}
	auto& bg = settings.background_;
	const auto valid_bg = bg ? phase.info_matches_except_complexity(bg->info()) : false;
	if (!valid_bg)
	{
		if (!processing_options.is_live)
		{
			qli_runtime_error("BG should always match?");
		}
		return camera_frame_internal();
	}
	//Step 3: For show BG return the BG image
	if (processing_options.show_mode == live_compute_options::background_show_mode::show_bg)
	{
		//could be moved to start of processing?
		static_cast<internal_frame_meta_data&>(phase) = *bg.get();
		if (bg->is_complex())
		{
			complex_bg_merge(*phase.buffer, bg->buffer, nullptr);
			phase.complexity = image_info::complex::no;
		}
		else
		{
			thrust::copy(bg->buffer.begin(), bg->buffer.end(), phase.buffer->begin());
		}
		return phase;
	}
	//Step 4: perform background subtraction
	const auto get_decomplexified_bg_subtracted = [&]
	{
		auto& img_out = *decomplexify_buffer_ptr;
		auto& img_in = *phase.buffer;
		auto bg_ptr = bg ? bg.get() : nullptr;
		complex_bg_merge(img_out, img_in, bg_ptr);
		camera_frame_internal return_me(decomplexify_buffer_ptr, { phase, phase });
		//
		decomplexify_buffer_ptr = phase.buffer;
		return_me.complexity = image_info::complex::no;
		return return_me;
	};
	const auto get_inplace = [&]
	{
		if (bg)
		{
			auto img = phase.buffer;
			const auto& bg_buffer = bg->buffer;
			thrust::transform(img->begin(), img->end(), bg_buffer.begin(), img->begin(), bg_subtract_functor());
		}
		return phase;
	};
	auto fixed_frame = phase.is_complex() ? get_decomplexified_bg_subtracted() : get_inplace();
#if _DEBUG
	if (!fixed_frame.is_valid())
	{
		qli_runtime_error("Somehow you made an invalid frame which is bad");
	}
#endif
	return fixed_frame;

}

[[nodiscard]] bool channel_settings::has_valid_background() const
{
	if (!background_)
	{
		return false;
	}
	//ideally also check phase processing, but oh welp maybe later (!)
	const auto processing_matches = background_->processing == this->processing;
	if (!processing_matches)
	{
		return false;
	}
	//note that the background is applied after the processing, which is wacky
	auto expected_size = image_info_per_capture_item_on_disk();
	auto size_match = background_->info_matches_except_complexity(expected_size);
	return size_match && processing_matches;
}
