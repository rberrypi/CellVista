#pragma once

#ifndef COMPACT_LIGHT_PATH_H
#define COMPACT_LIGHT_PATH_H

#include "phase_processing.h"
#include "instrument_configuration.h"
#include "camera_config.h"
#include "compute_and_scope_state.h"
#include "phase_shift_exposure_and_delay.h"

struct compact_light_path : processing_quad, microscope_light_path, display_settings, camera_config, band_pass_settings
{
	phase_shift_exposures_and_delays frames;
	float zee_offset;
	std::string label_suffix;

	compact_light_path() noexcept : compact_light_path(processing_quad(), microscope_light_path(), display_settings(), camera_config(), 0.0f, phase_shift_exposures_and_delays(), band_pass_settings(), std::string()) {};

	compact_light_path(const processing_quad& processing_quad, const microscope_light_path& microscope_light_path, const display_settings& display_settings, const camera_config& camera_config, const float zee_offset, const phase_shift_exposures_and_delays& delays, const band_pass_settings& band_pass_settings, const std::string& custom_label) noexcept: processing_quad(processing_quad), microscope_light_path(microscope_light_path), display_settings(display_settings), camera_config(camera_config), band_pass_settings(band_pass_settings), frames(delays), zee_offset(zee_offset), label_suffix(custom_label) {};

	[[nodiscard]] bool is_valid() const noexcept
	{
		return !frames.empty() && is_supported_quad();
	}

	[[nodiscard]] bool item_approx_equals(const compact_light_path& new_light_path) const noexcept
	{
		return approx_equals(zee_offset, new_light_path.zee_offset)
			&& denoise == new_light_path.denoise &&
			std::equal(frames.begin(), frames.end(), new_light_path.frames.begin(), new_light_path.frames.end()) &&
			static_cast<const processing_quad&>(*this) == new_light_path &&
			microscope_light_path::item_approx_equals(new_light_path) &&
			static_cast<const display_settings&>(new_light_path) == new_light_path &&
			static_cast<const camera_config&>(new_light_path) == new_light_path &&
			band_pass_settings::item_approx_equals(new_light_path);
	}

};
#endif 