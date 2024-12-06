#include "stdafx.h"
#include "channel_settings.h"
#include "settings_file.h"
#include "device_factory.h"
#include "camera_device.h"
#include "scope.h"
#include <array>
#include "qli_runtime_error.h"

capture_iterator_view channel_settings::iterator() const
{
	const auto frames = exposures_and_delays.size();
	return compute_and_scope_settings::iterator(frames);
}

bool channel_settings::is_valid() const noexcept
{
	const auto live_gui_settings_are_valid = live_gui_settings::is_valid();
	const auto fixed_hardware_settings_are_valid = fixed_hardware_settings::is_valid();
	const auto valid = live_gui_settings_are_valid && fixed_hardware_settings_are_valid;
#if _DEBUG
	if (!valid)
	{
		const auto volatile here = 0;
		Q_UNUSED(here);
	}
#endif
	return valid;
}

channel_settings::compensations channel_settings::get_compensations() const
{
#if _DEBUG
	if (exposures_and_delays.empty())
	{
		qli_invalid_arguments();
	}
#endif
	const auto first_time = exposures_and_delays.front().exposure_time;
	const auto comps_to_return = exposures_and_delays.size();
	compensations return_me(comps_to_return, 1);
	const auto functor = [first_time](const phase_shift_exposure_and_delay& frame)
	{
		const auto comp = first_time.count() / (1.0f * frame.exposure_time.count());
		return comp;
	};
	std::transform(exposures_and_delays.begin(), exposures_and_delays.end(), return_me.begin(), functor);
#if _DEBUG
	{
		const auto validity_checker = [](const float value)
		{
			return std::isfinite(value) && value != 0;
		};
		const auto valid = std::all_of(return_me.begin(), return_me.end(), validity_checker);
		if (!valid)
		{
			qli_runtime_error();
		}
	}
#endif
	return return_me;
}

int channel_settings::output_files_per_compute(const bool is_live) const
{
	const auto patterns = exposures_and_delays.size();
	return compute_and_scope_settings::output_files_per_compute(patterns, is_live);
}

channel_settings::channel_settings(const fixed_hardware_settings& fixed_hardware_settings, const live_gui_settings& live_gui_settings) noexcept : fixed_hardware_settings(fixed_hardware_settings), live_gui_settings{ live_gui_settings }
{

}

size_t channel_settings::bytes_per_capture_item_on_disk() const
{
	const auto output_image_size = image_info_per_capture_item_on_disk().samples();
	const auto is_zero_compute = is_native_sixteen_bit();
	const auto bytes_per_pixel = is_zero_compute ? sizeof(unsigned short) : sizeof(float);
	//should also have paths for radial average when written to the disk
	const auto roi_size = output_image_size * bytes_per_pixel;
	const static size_t tiff_over_head = 0;// We could probably measure this?
	return roi_size + tiff_over_head;
}

image_info channel_settings::image_info_per_capture_item_on_disk() const
{
	const auto& camera = D->cameras.at(camera_idx);
	const auto sensor_size = camera->get_sensor_size(*this);
	const auto demosaic_binning = demosaic_setting::info.at(demosaic).binning;
	const auto demosaiced_size = frame_size(sensor_size.width / demosaic_binning, sensor_size.height / demosaic_binning);
	const auto dpm_size = frame_size(dpm_phase_width, dpm_phase_width);
	const auto is_dpm = phase_processing_setting::settings.at(processing).is_a_dpm;
	const auto retrieval_frame_size = is_dpm ? dpm_size : demosaiced_size;
	const auto processing_scale = phase_processing_setting::settings.at(this->processing).upscale;
	const auto processing_frame_size = frame_size(retrieval_frame_size.width * processing_scale, retrieval_frame_size.height * processing_scale);
	const auto samples_per_pixel = camera_device::samples_per_pixel(camera->chroma, demosaic);
	return image_info(processing_frame_size, samples_per_pixel, image_info::complex::no);
}

void channel_settings::assert_validity() const
{
#if _DEBUG
	if (!is_valid())
	{
		qli_runtime_error("Invalid Compute Settings");
	}
#endif
}

per_modulator_saveable_settings per_modulator_saveable_settings::generate_per_modulator_saveable_settings(int patterns, int samples_per_pixel)
{
	const std::array<int, 4> weight_sequence = { 1,0,-1,0 };//some variation of this
	const std::array<float, 4> slm_samples = { 200,230,250,255 };
	modulator_configuration::four_frame_psi_settings four_frame_psi_setting_holder;
	for (auto pattern_idx = 0; pattern_idx < four_frame_psi_setting_holder.size(); ++pattern_idx)
	{
		//todo use an auto lambda, but KISS for now
		auto& pattern = four_frame_psi_setting_holder.at(pattern_idx);
		pattern.slm_value = slm_samples.at(pattern_idx);
		pattern.weights.resize(samples_per_pixel);
		for (auto color_idx = 0; color_idx < pattern.weights.size(); ++color_idx)
		{
			pattern.weights.at(color_idx).top = weight_sequence.at((pattern_idx + 1) % weight_sequence.size());//Sine
			pattern.weights.at(color_idx).bot = weight_sequence.at((pattern_idx + 0) % weight_sequence.size());//Cosine	
		}
	}

	const illumination_power_settings illumination_power_settings;
	const darkfield_pattern_settings darkfield_pattern_settings;
	constexpr auto voltage_max = 7.2f;
	const distorted_donut beam_settings(256, 230, 90, 120, 1.07, 1.00);
	modulator_configuration modulator_configuration(beam_settings, darkfield_pattern_settings, four_frame_psi_setting_holder, illumination_power_settings, voltage_max);
	if (!modulator_configuration.is_valid())
	{
		qli_runtime_error();
	}
	const phase_shift_pattern phase_shift_pattern;
	psi_function_pairs weights;
	for (auto idx = 0; idx < samples_per_pixel; ++idx)
	{
		const psi_function_pair psi_function_pair;
		weights.push_back(psi_function_pair);
	}
	const per_pattern_modulator_settings per_pattern_modulator_settings(phase_shift_pattern, beam_settings, weights);
	per_pattern_modulator_settings_patterns per_pattern_modulator_settings_patterns(patterns, per_pattern_modulator_settings);
	for (auto pattern_idx = 0; pattern_idx < patterns; ++pattern_idx)
	{
		auto& pattern = per_pattern_modulator_settings_patterns.at(pattern_idx);
		const auto slm_value = pattern_idx / 2.0f;
		pattern.slm_background = slm_value;
		pattern.slm_value = slm_value;
		pattern.pattern_mode = slm_pattern_mode::checkerboard;
		for (auto color_idx = 0; color_idx < pattern.weights.size(); ++color_idx)
		{
			pattern.weights.at(color_idx).top = weight_sequence.at((pattern_idx + 1) % weight_sequence.size());//Sine
			pattern.weights.at(color_idx).bot = weight_sequence.at((pattern_idx + 0) % weight_sequence.size());//Cosine	
		}
	}
	constexpr auto actual_patterns = false;
	per_modulator_saveable_settings per_modulator_saveable_settings(modulator_configuration, per_pattern_modulator_settings_patterns, actual_patterns);
	if (!per_modulator_saveable_settings.is_valid())
	{
		qli_runtime_error();
	}
	return per_modulator_saveable_settings;
}

fixed_hardware_settings fixed_hardware_settings::generate_fixed_hardware_settings(const slm_mode slm_mode, const int samples_per_pixel, const int slms)
{
	constexpr  auto attenuation = 2.45f;
	constexpr auto stage_overlap = 0.1f;
	const pixel_dimensions pixel_dimensions(1.5, 7.4);
	const qdic_scope_settings qdic_scope_settings(45, 0.3f);
	const wave_length_package wavelengths = { 0.470f,0.55f,0.670f };
	const scope_compute_settings scope_compute_settings(attenuation, stage_overlap, pixel_dimensions, qdic_scope_settings, wavelengths);
	const dpm_settings dpm_settings;//needs to be filled out on a per frame basis
	const auto pattern_count = [&] {
		const auto& slm_settings = slm_mode_setting::settings.at(slm_mode);
		switch (slm_mode)
		{
		case slm_mode::darkfield:
		{
			return 0;
		}
		case slm_mode::custom_patterns:
		{
			//lets pretend its a calibration sequence
			return typical_calibration_patterns;
		}
		default:
			return slm_settings.patterns;
		}
	}();

	auto per_modulator_saveable_settings = per_modulator_saveable_settings::generate_per_modulator_saveable_settings(pattern_count, samples_per_pixel);
	fixed_modulator_settings fixed_modulator_settings(slms, per_modulator_saveable_settings);
	//okay so for multiple SLMS the second SLM is the illuminator which behaves in a special way
	if (fixed_modulator_settings.size() == 2)
	{
		//todo use enum lookup
		auto& illuminator_settings = fixed_modulator_settings.at(1);
		// const auto value = illuminator_settings.brightfield_scale_factor*std::numeric_limits<unsigned char>::max();
		 //const auto value = illuminator_settings.brightfield_scale_factor*illuminator_settings.illumination_power;
		constexpr auto value = 250.f;
		for (auto& pattern : illuminator_settings.patterns)
		{
			pattern.slm_background = value;
			pattern.slm_value = value;
			pattern.pattern_mode = slm_pattern_mode::checkerboard;
		}
	}
	fixed_hardware_settings fixed_hardware_settings(fixed_modulator_settings, scope_compute_settings, dpm_settings);
	if (!fixed_hardware_settings.is_valid())
	{
		qli_runtime_error();
	}
	return fixed_hardware_settings;
}

channel_settings channel_settings::generate_test_channel(const processing_quad& testing_quad)
{
	const auto minimum_modulators = D->get_slm_count();
	const auto samples = D->max_samples_per_pixels();
	return generate_test_channel(testing_quad, minimum_modulators, samples);
}

channel_settings channel_settings::generate_test_channel(const processing_quad& testing_quad, const int slms, const int samples_per_pixel)
{
#if _DEBUG
	if (!testing_quad.is_supported_quad())
	{
		qli_invalid_arguments();
	}
#endif
	const camera_config camera_config;
	const auto slm_mode = phase_retrieval_setting::settings.at(testing_quad.retrieval).slm_mode;
	const auto fixed_hardware_settings = generate_fixed_hardware_settings(slm_mode, samples_per_pixel, slms);
	const render_modifications render_modifications;
	const auto display_settings = phase_processing_setting::settings.at(testing_quad.processing).get_display_settings();
	const ml_remapper ml_remapper;
	const render_shifter render_shifter;
	const render_settings render_settings(render_modifications, display_settings, ml_remapper, render_shifter);
	const microscope_light_path microscope_light_path;
	const auto is_raw_frame = testing_quad.is_raw_frame();
	const band_pass_settings band_pass_settings(0.2, 600, true, !is_raw_frame);

	const slim_bg_settings slim_bg_settings;
	const background_frame_ptr background_frame_holder;
	const material_info material_info(1.3f, 1.38f, 0.3f, 1.0f);
	const compute_and_scope_settings compute_and_scope_settings(testing_quad, render_settings, microscope_light_path, camera_config, band_pass_settings, slim_bg_settings, background_frame_holder, material_info);

	const phase_shift_exposure_and_delay phase_shift_exposure_and_delay(ms_to_chrono(30), ms_to_chrono(60));
	const auto patterns = fixed_hardware_settings.modulator_settings.front().patterns.size();
	phase_shift_exposures_and_delays live_pattern_settings(patterns, phase_shift_exposure_and_delay);
	live_gui_settings live_gui_settings(compute_and_scope_settings, live_pattern_settings);
	//
	auto settings = channel_settings(fixed_hardware_settings, live_gui_settings);
	settings.fixup_label_suffix();
	settings.assert_validity();
	return settings;
}

void channel_settings::fixup_label_suffix()
{
	//auto
	const auto has_scope = D && D->scope;
	const auto label = has_scope ? D->scope->get_channel_settings_names().at(scope_channel).toStdString() : "Camera";
	compute_and_scope_settings::fixup_label_suffix(label);
}

void channel_settings::fixup_channel()
{
	const auto minimum_modulators = D->get_slm_count();
	const auto samples = D->max_samples_per_pixels();
	const auto retrieval_patterns = phase_retrieval_setting::settings.at(retrieval).modulator_patterns();
	const auto patterns = retrieval_patterns == pattern_count_from_file ? modulator_settings.size() : retrieval_patterns;
	modulator_settings.resize(minimum_modulators);
	for (auto& modulator : modulator_settings)
	{
		modulator.patterns.resize(patterns);
		for (auto& pattern : modulator.patterns)
		{
			pattern.set_samples_per_pixel(samples);
		}
	}
#if _DEBUG
	assert_validity();
#endif
}

[[nodiscard]] bool channel_settings::difference_clears_background(const channel_settings& channel_settings) const noexcept
{
	const auto processing_quad_changed = !(static_cast<const processing_quad&>(*this) == channel_settings);
	const auto camera_changed = !(static_cast<const processing_quad&>(*this) == channel_settings);
	return processing_quad_changed || camera_changed;
}


bool channel_settings::difference_requires_camera_reload(const channel_settings& channel_settings) const noexcept
{
	return static_cast<const camera_config&>(*this) != channel_settings;
}