#include "render_widget.h"
#include "cuda_error_check.h"
#include <cuda_runtime_api.h>
#include "chrono_converters.h"
#include "compute_engine.h"
#include "write_debug_gpu.h"
#include "qli_runtime_error.h"
bool histogram_meta_info::is_valid() const
{
	return  abs(bot_idx - top_idx) > 10;
}

void render_widget::calc_histogram(const display_settings::display_ranges& expected_range, histogram_info& histogram_to_fill, unsigned char* out_d_8_bit_img, const float* input_image, const frame_size& frame_size, const int samples_per_pixel)
{
#if _DEBUG
	if (expected_range.empty())
	{
		qli_runtime_error();
	}
#endif
	//13 ms for 4 MP on GTX 1080
	comp_->move_clamp_and_scale(out_d_8_bit_img, input_image, img_size, samples_per_pixel, expected_range);
	comp_->calc_histogram(histogram, out_d_8_bit_img, img_size.n(), samples_per_pixel, expected_range, true);
}

void render_widget::move_to_old_and_calculate_histogram(const bool is_live, const float* img_d, const phase_processing processing)
{
	GLubyte* ogl_device_ptr;
	size_t cuda_buffer_size;//Perhaps because you might not know the OpenGL size?
	CUDASAFECALL(cudaGraphicsMapResources(1, &phase_resource_));
	CUDASAFECALL(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ogl_device_ptr), &cuda_buffer_size, phase_resource_));
	const auto valid_size = cuda_buffer_size == img_size.n() * samples_per_pixel * sizeof(unsigned char);
	//if invalid should runtime error?
	if (valid_size)
	{
		//histograming takes some time (~15 ms) so it shouldn't be done every frame
		const static auto histogram_min_interval = ms_to_chrono(150);
		static auto time_since_last_histogram = ms_to_chrono(0);
		const auto histogram_time_elapsed = ((timestamp() - time_since_last_histogram) > histogram_min_interval);
		const auto do_live_auto_contrast = render_settings_.live_auto_contrast && histogram_time_elapsed;
		const auto final_display_range = [&] {
			if (do_live_auto_contrast)
			{
				const auto expected_range = phase_processing_setting::settings.at(processing).display_range.predict_max_possible();
				display_settings::display_ranges expected_ranges(samples_per_pixel, expected_range);
				constexpr auto max_zoom_attempts = 2;
				for (auto zoom_attempt = 0; zoom_attempt < max_zoom_attempts; ++zoom_attempt)
				{
					calc_histogram(expected_ranges, histogram, ogl_device_ptr, img_d, img_size, samples_per_pixel);
					//ideally we'd do each range separately, but this isn't implemented (also dirty hack around)
					auto all_ranges_valid = true;
					for (auto histogram_meta_info_idx = 0; histogram_meta_info_idx < expected_ranges.size(); ++histogram_meta_info_idx)
					{
						auto& info = histogram.info.at(histogram_meta_info_idx);
						if (!info.is_valid())
						{
							all_ranges_valid = true;
							auto& expected_range = expected_ranges.at(histogram_meta_info_idx);
							const auto delta = 5 * abs(expected_range.max - expected_range.min) / 256;
							const auto min_value = info.bot - delta;
							const auto max_value = info.top + delta;
							expected_range = { min_value,max_value };
						}
					}
					if (!all_ranges_valid)
					{
						break;
					}
				}
				const auto new_display_range = histogram.predict_display_ranges();
				emit load_auto_contrast_settings(new_display_range);
				return  new_display_range;
			}
			return render_settings_.ranges;
		}();
		//std::cout << render_settings_.ranges.front().min << "->" << final_display_range.front().min << std::endl;
		comp_->move_clamp_and_scale(ogl_device_ptr, img_d, img_size, samples_per_pixel, final_display_range);
		if (histogram_time_elapsed)
		{
			std::unique_lock<std::mutex> lk_histogram(histogram_m);
			comp_->calc_histogram(histogram, ogl_device_ptr, img_size.n(), samples_per_pixel, final_display_range, false);
			emit load_histogram();
			time_since_last_histogram = timestamp();
		}
	}
	CUDASAFECALL(cudaGraphicsUnmapResources(1, &phase_resource_));
	CUDA_DEBUG_SYNC();
}