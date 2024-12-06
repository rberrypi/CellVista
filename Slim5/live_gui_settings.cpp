#include "stdafx.h"
#include "live_gui_settings.h"

#include "qli_runtime_error.h"

bool live_gui_settings::is_valid() const
{
	const auto has_frames = !exposures_and_delays.empty();
	const auto base_class_valid = compute_and_scope_settings::is_valid();
	const auto expected_patterns = phase_retrieval_setting::settings.at(retrieval).modulator_patterns();
	const auto actual_patterns = exposures_and_delays.size();
	const auto has_expected_patterns = expected_patterns == pattern_count_from_file ? true : actual_patterns >= expected_patterns;
	const auto valid_pattern_to_display = this->current_pattern<exposures_and_delays.size();
	const auto valid = has_frames && base_class_valid && has_expected_patterns && valid_pattern_to_display;
#if _DEBUG
	if (!valid)
	{
		qli_runtime_error("Shouldn't happen");
	}
#endif
	return valid;

}

void live_gui_settings::assert_validity() const
{
#if _DEBUG
	if (!is_valid())
	{
		qli_runtime_error("Invalid Compute Settings");
	}
#endif
}

void live_gui_settings::fill_exposure_settings(const std::chrono::microseconds exposure)
{
#if _DEBUG
	if (exposures_and_delays.empty())
	{
		qli_runtime_error("Has no effect");
	}
#endif
	for (auto& item : exposures_and_delays)
	{
		item.exposure_time = exposure;
	}
}

