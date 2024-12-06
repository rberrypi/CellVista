#include "stdafx.h"
#include "modulator_configuration.h"
#include "qli_runtime_error.h"

bool per_modulator_saveable_settings::is_valid() const
{
	const auto has_patterns = !patterns.empty();
	const auto all_valid = std::all_of(patterns.begin(), patterns.end(), [&](const per_pattern_modulator_settings& pattern)
	{
		return pattern.is_valid();
	});
	const auto modulator_config_valid = modulator_configuration::is_valid();
	const auto valid = has_patterns && all_valid && modulator_config_valid;
#if _DEBUG
	if (!valid)
	{
		auto volatile what = 0;
	}
#endif
	return valid;
}

bool illumination_power_settings::is_complete() const noexcept
{
	return std::isnormal(illumination_power) && std::isnormal(brightfield_scale_factor);
}

bool darkfield_pattern_settings::is_complete() const noexcept
{
	return std::isnormal(width_na) && std::isnormal(ref_ring_na) && std::isnormal(objective_na) && std::isnormal(max_na);
}

[[nodiscard]] bool distorted_donut::is_complete() const noexcept
{
	return std::isnormal(ellipticity_e) && std::isnormal(ellipticity_f) && outer_diameter > inner_diameter;
}

bool modulator_configuration::is_valid() const noexcept
{
	const auto four_frame_psi_settings_functor = [](const four_frame_psi_setting& four_frame)
	{
		return four_frame.is_valid();
	};
	const auto has_patterns = !four_frame_psi.empty();
	const auto valid = has_patterns && std::all_of(four_frame_psi.begin(), four_frame_psi.end(), four_frame_psi_settings_functor);
#if _DEBUG
	if (!valid)
	{
		auto volatile what = 0;
	}
#endif
	return valid;
}

const darkfield_mode_settings::darkfield_mode_settings_holder darkfield_mode_settings::settings = {
	{darkfield_mode::dots,darkfield_mode_settings("Dots",false,slm_pattern_mode::checkerboard)},
	{darkfield_mode::dots_with_ring_psi,darkfield_mode_settings("Dots & Ring PSI",true,slm_pattern_mode::donut)},
	{darkfield_mode::dots_with_dic_psi,darkfield_mode_settings("Dots & QDIC PSI",true,slm_pattern_mode::checkerboard)},
	{darkfield_mode::pdots_with_dic_psi,darkfield_mode_settings("Pair Dots & QDIC PSI",true,slm_pattern_mode::checkerboard)},
	{darkfield_mode::rings,darkfield_mode_settings("Rings",false,slm_pattern_mode::checkerboard)},
	{darkfield_mode::rings_with_psi,darkfield_mode_settings("Rings & QDIC PSI",true,slm_pattern_mode::checkerboard)}
};

const darkfield_pattern_settings::darkfield_pattern_settings_map darkfield_pattern_settings::darkfield_display_mode_settings = {
	{darkfield_pattern_settings::darkfield_display_align_mode::darkfield,"Darkfield"},
	{darkfield_pattern_settings::darkfield_display_align_mode::align_ref_ring_na,"Reference Ring"},
	{darkfield_pattern_settings::darkfield_display_align_mode::align_objective_na,"Objective Ring"},
	{darkfield_pattern_settings::darkfield_display_align_mode::align_max_na,"Max Ring"}
};

const slm_pattern_mode_names_map slm_pattern_mode_names = {
{slm_pattern_mode::donut,"Donut"},
{slm_pattern_mode::file,"File"},
{slm_pattern_mode::checkerboard,"Checkerboard"},
{slm_pattern_mode::alignment,"Alignment"}

};

const slm_mode_setting::slm_mode_setting_map slm_mode_setting::settings = {
{ slm_mode::slim, slm_mode_setting(typical_psi_patterns,"SLIM Ring")},
{ slm_mode::qdic, slm_mode_setting(typical_psi_patterns,"GLIM Piston")},
{ slm_mode::single_shot, slm_mode_setting(single_shot,"Single Shot")},
{ slm_mode::two_shot_lcvr, slm_mode_setting(2,"Two Shot")},
{ slm_mode::darkfield, slm_mode_setting(pattern_count_from_file,"Darkfield")},
{ slm_mode::custom_patterns, slm_mode_setting(pattern_count_from_file,"Custom")},
};

void per_pattern_modulator_settings::assert_valid() const
{
#if _DEBUG
	if (!is_valid())
	{
		qli_runtime_error("Invalid per pattern modulator settings");
	}
#endif
}

[[nodiscard]] bool per_pattern_modulator_settings::is_valid() const noexcept
{
	const auto weights_valid = weights.size() == 1 || weights.size() == 3;
	const auto patterns_valid = slm_levels::is_valid();
	const auto valid = weights_valid && patterns_valid;
#if _DEBUG
	if (!valid)
	{
		qli_runtime_error();
	}
#endif
	return valid;
}

const double_spin_box_settings psi_function_pair::spin_box_settings = double_spin_box_settings(-1, 1, 5);