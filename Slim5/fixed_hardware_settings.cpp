#include "stdafx.h"
#include "fixed_hardware_settings.h"
bool fixed_hardware_settings::is_valid() const noexcept
{
	const auto all_valid = std::all_of(modulator_settings.begin(),modulator_settings.end(),[](const per_modulator_saveable_settings& per_modulator_saveable_settings)
	{
		return per_modulator_saveable_settings.is_valid();
	});
	const auto all_same_size =std::all_of(modulator_settings.begin(),modulator_settings.end(),[&](const per_modulator_saveable_settings& per_modulator_saveable_settings)
	{
		return per_modulator_saveable_settings.patterns.size() == modulator_settings.front().patterns.size();
	});
	const auto has_items = !modulator_settings.empty();
	const auto is_valid =  has_items && all_valid && dpm_settings::is_valid() && all_same_size;
#if _DEBUG
	if (!is_valid)
	{
		const auto volatile here=0;
	}
#endif
	return is_valid;
}