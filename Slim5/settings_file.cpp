#include "stdafx.h"
#include "settings_file.h"
#include "device_factory.h"
#include "phase_processing.h"
#include "qli_runtime_error.h"

bool settings_file::is_complete() const noexcept
{
	return is_valid() && !file_path.empty() && scope_compute_settings::is_complete();
}

settings_file::settings_file(const fixed_hardware_settings& settings, const slm_pattern_generation& modulation_mode, const std::string& filepath) noexcept : fixed_hardware_settings{ settings }, slm_pattern_generation(modulation_mode), file_path{ filepath }
{

}

void settings_file::assert_valid() const
{
#if _DEBUG
	if (!is_valid())
	{
		qli_invalid_arguments();
	}
#endif
}

[[nodiscard]] int settings_file::pattern_count() const noexcept
{
	return modulator_settings.front().patterns.size();
}

bool settings_file::is_valid() const
{
	const auto hw_settings_valid = fixed_hardware_settings::is_valid();
	const auto slm_pattern_settings_valid = slm_pattern_generation::is_valid();
	const auto required_patterns = slm_mode_setting::settings.at(modulator_mode).patterns;
	const auto pattern_count_check = [&](const per_modulator_saveable_settings& setting)
	{
		const auto right_number_of_patterns =  setting.patterns.size() == required_patterns;
		return right_number_of_patterns;
	};
	const auto all_modulators_correctly_sized = required_patterns==pattern_count_from_file || std::all_of(modulator_settings.begin(), modulator_settings.end(), pattern_count_check);
	const auto valid = hw_settings_valid && slm_pattern_settings_valid && all_modulators_correctly_sized;
#if _DEBUG
	if (!valid)
	{
		const auto volatile test = 0;
	}
#endif
	return valid;
}

settings_file settings_file::generate_default_settings_file(const phase_retrieval retrieval)
{
	const auto samples_per_pixel = D->max_samples_per_pixels();
	const auto slms = D->get_slm_count();
	const auto slm_mode = phase_retrieval_setting::settings.at(retrieval).slm_mode;
	const auto fixed_hardware_settings = fixed_hardware_settings::generate_fixed_hardware_settings(slm_mode, samples_per_pixel, slms);
	const slm_pattern_generation slm_pattern_generation(slm_mode,darkfield_mode::dots,10);
	auto settings = settings_file(fixed_hardware_settings, slm_pattern_generation, "");
#if _DEBUG
	if (!settings.is_valid())
	{
		qli_invalid_arguments();
	}
#endif
	return settings;
}
