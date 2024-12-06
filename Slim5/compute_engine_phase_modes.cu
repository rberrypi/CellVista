#include "compute_engine.h"
#include "thrust_resize.h"
#include "channel_settings.h"
#include "write_debug_gpu.h"

camera_frame_internal compute_engine::phase_retrieval_camera(const internal_frame_meta_data& meta_data, phase_retrieval retrieval)
{
	auto img = get_out_frame();
	if (img == nullptr)
	{
		return camera_frame_internal();
	}
	const auto frame_to_load = pattern_to_load(retrieval, meta_data.pattern_idx);
	pass_thru(*img, input_frames.at(frame_to_load));
	return camera_frame_internal(img, meta_data);
}

frame_size compute_engine::phase_retrieval_pol(out_frame output_frame, const channel_settings& settings, const internal_frame_meta_data& meta_data, const live_compute_options&)
{
	const auto processing = settings.processing;
	frame_size meta_data_changed = meta_data;
	//so actually conver the set? AKA m0,m1,m2,

	switch (processing)
	{
		//pol_quads, degree_of_linear_polarization, angles_of_linear_polarization, hue_saturation_value
	case phase_processing::phase:
		ac_dc(output_frame, input_frames.at(0), input_frames.at(1), input_frames.at(2), input_frames.at(3), settings, meta_data.samples_per_pixel);
		break;
	case phase_processing::stoke_0:
	case phase_processing::stoke_1:
	case phase_processing::stoke_2:
	case phase_processing::angles_of_linear_polarization:
	case phase_processing::degree_of_linear_polarization:
		polarization_merger(output_frame, input_frames.at(0), input_frames.at(1), input_frames.at(2), input_frames.at(3), processing);
		break;
	case phase_processing::quad_pass_through:
		meta_data_changed = merge_quads(output_frame, input_frames.at(0), input_frames.at(1), input_frames.at(2), input_frames.at(3), meta_data);
		break;
	default:
		qli_not_implemented();
	}
	return meta_data_changed;
}
// Correct solution is to push? or just return the frames that are available?
camera_frame_internal compute_engine::phase_retrieval_psi(const channel_settings& settings, const internal_frame_meta_data& meta_data, const live_compute_options& processing_options)
{
	const auto retreival = settings.retrieval;
	const auto& retrieval_settings = phase_retrieval_setting::settings.at(retreival);
	const auto match_first_size = [&](const auto& in)
	{
		return in.size() == input_frames.front().size();
	};
	const auto pattern_idx = retrieval_settings.processing_patterns;
	auto input_same_size = std::all_of(input_frames.begin(), input_frames.begin() + pattern_idx, match_first_size) && !input_frames.begin()->empty();
	const auto is_final_frame = meta_data.pattern_idx == (pattern_idx - 1);
	const auto can_do_live = retrieval_settings.do_live(pattern_idx);
	const auto do_live = processing_options.is_live && can_do_live;
	const auto do_compute = input_same_size && (is_final_frame || do_live);
	//
	//
	if (!do_compute)
	{
		return camera_frame_internal();
	}
	//
	const auto processing = settings.processing;
	//
	auto img = get_out_frame();
	auto meta_data_after = meta_data;
	if (img == nullptr)
	{
		return camera_frame_internal();
	}
	{
		constexpr auto force_debug = false;
		write_debug_gpu(input_frames.at(0), meta_data.width, meta_data.height, meta_data.samples_per_pixel, "A.tif", force_debug);
		write_debug_gpu(input_frames.at(1), meta_data.width, meta_data.height, meta_data.samples_per_pixel, "B.tif", force_debug);
		write_debug_gpu(input_frames.at(2), meta_data.width, meta_data.height, meta_data.samples_per_pixel, "C.tif", force_debug);
		write_debug_gpu(input_frames.at(3), meta_data.width, meta_data.height, meta_data.samples_per_pixel, "D.tif", force_debug);
	}
	//
	const auto sets_bg = processing_options.show_mode == live_compute_options::background_show_mode::set_bg;
	switch (retreival)
	{
	case phase_retrieval::slim:
	case phase_retrieval::slim_demux:
	case phase_retrieval::fpm:
		compute_slim_phase(*img, input_frames.at(0), input_frames.at(1), input_frames.at(2), input_frames.at(3), settings, sets_bg, meta_data.channel_route_index, retreival == phase_retrieval::fpm);
		break;
	case phase_retrieval::glim:
	case phase_retrieval::glim_demux:
		ac_dc(*img, input_frames.at(0), input_frames.at(1), input_frames.at(2), input_frames.at(3), settings, meta_data.samples_per_pixel);
		break;
	case phase_retrieval::polarizer_demux_single:
	case phase_retrieval::polarizer_demux_psi:
		static_cast<frame_size&>(meta_data_after) = phase_retrieval_pol(*img, settings, meta_data, processing_options);
		break;
	case phase_retrieval::diffraction_phase:
		static_cast<frame_size&>(meta_data_after) = compute_dpm_phase(*img, input_frames.at(0), settings.processing, meta_data, settings, dpm_update_, sets_bg, meta_data.channel_route_index);
		meta_data_after.complexity = image_info::complex::yes;
		break;
	case phase_retrieval::polarizer_demux_two_frame_dpm:
	{
		switch (processing)
		{
		case phase_processing::quad_phase:
			static_cast<frame_size&>(meta_data_after) = compute_dpm_phase_quads(*img, input_frames.at(0), input_frames.at(1), input_frames.at(2), input_frames.at(3), meta_data, settings, dpm_update_, sets_bg, meta_data.channel_route_index);
			meta_data_after.complexity = image_info::complex::yes;
			break;
		case phase_processing::quad_pass_through:
			static_cast<frame_size&>(meta_data_after) = merge_quads(*img, input_frames.at(0), input_frames.at(1), input_frames.at(2), input_frames.at(3), meta_data);
			break;
		case phase_processing::pol_psi_octo_compute:
			static_cast<frame_size&>(meta_data_after) = compute_dpm_psi_octo(*img, input_frames.at(0), input_frames.at(1), input_frames.at(2), input_frames.at(3), meta_data, settings, dpm_update_, sets_bg, meta_data.channel_route_index);
			meta_data_after.complexity = image_info::complex::yes;
			break;
		default:
			qli_not_implemented();
		}
		break;
	}
	default:
		qli_not_implemented();
	}
	auto frame = camera_frame_internal(img, meta_data_after);
	if (!frame.is_valid())
	{
		volatile	const auto elements = frame.samples();
		volatile const auto frames_samples_height = frame.height;
		volatile const auto frames_samples_width = frame.width;
		volatile 	const auto elements_we_have = static_cast<int>(frame.buffer->size());
		std::cout << elements << "," << elements_we_have << " " << frames_samples_width << "," << frames_samples_height << std::endl;
	}
	return frame;
}