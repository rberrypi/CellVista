#pragma once
#ifndef SETTINGS_FILE_H
#define SETTINGS_FILE_H
#include "fixed_hardware_settings.h"
#include "modulator_configuration.h"

struct processing_double;

struct slm_pattern_generation
{
	slm_mode modulator_mode;
	darkfield_mode darkfield;
	int darkfield_samples;
	[[nodiscard]] bool operator== (const slm_pattern_generation& b) const noexcept
	{
		return modulator_mode == b.modulator_mode && darkfield == b.darkfield && darkfield_samples == b.darkfield_samples;
	}
	//
	slm_pattern_generation(const slm_mode mode, const darkfield_mode darkfield, const int samples) noexcept : modulator_mode(mode), darkfield(darkfield), darkfield_samples(samples) {}
	slm_pattern_generation() noexcept :slm_pattern_generation(slm_mode::single_shot, darkfield_mode::dots, 4) {};

	[[nodiscard]] bool is_valid() const noexcept
	{
		if (modulator_mode == slm_mode::darkfield)
		{
			return darkfield_samples > 0;
		}
		return modulator_mode != slm_mode::unset;
	}
};

struct settings_file final : fixed_hardware_settings, slm_pattern_generation
{
	//dirty hack for now, get rid of this bullshit later
	std::string file_path;
	//
	//todo move to fixed hardware settings, later
	settings_file()=default;
	settings_file(const fixed_hardware_settings& settings, const slm_pattern_generation& modulation_mode, const std::string& filepath) noexcept;
	[[nodiscard]] static settings_file generate_default_settings_file(phase_retrieval retrieval);
	[[nodiscard]] bool is_valid() const;
	[[nodiscard]] int pattern_count() const noexcept;
	void assert_valid() const;
	void regenerate_pattern();
	void regenerate_pattern(const slm_dimensions& slm_dimensions, int samples_per_pixel);
	static void fill_dot_list(per_pattern_modulator_settings_patterns& changed, const modulator_configuration& per_modulator_saveable_settings, const frame_size& slm_size, int darkfield_samples, bool is_illumination, darkfield_mode darkfield_mode, int samples_per_pixel);
	static void fill_paired_dot_list(per_pattern_modulator_settings_patterns& changed, const modulator_configuration& per_modulator_saveable_settings, const frame_size& slm_size, const int samples, const bool is_illumination, const darkfield_mode darkfield_mode, const int samples_per_pixel);
	static void fill_circle_list(per_pattern_modulator_settings_patterns& changed, const modulator_configuration& per_modulator_saveable_settings, const frame_size& slm_size, int darkfield_samples, bool is_illumination, darkfield_mode darkfield_mode, int samples_per_pixel);
	static void fill_four_frame_psi(per_pattern_modulator_settings_patterns& changed, const modulator_configuration& per_modulator_saveable_settings, const frame_size& slm_size,  slm_pattern_mode slm_mode,  bool is_illumination, int patterns, int samples_per_pixel);
	[[nodiscard]] static settings_file read(const std::string& filename, bool& okay) ;
	[[nodiscard]] bool write() const;
	[[nodiscard]] bool operator== (const settings_file& b) const noexcept
	{
		return static_cast<const fixed_hardware_settings&>(*this) == b && static_cast<const slm_pattern_generation&>(*this) == b;
	}

	[[nodiscard]] bool item_approx_equals(const settings_file& b) const noexcept
	{

		return fixed_hardware_settings::item_approx_equals(b) && static_cast<const slm_pattern_generation&>(*this) == b;
	}
	[[nodiscard]] bool operator!= (const settings_file& b) const noexcept
	{
		return !(*this == b);
	}

	[[nodiscard]] bool is_complete() const noexcept;
};
#endif