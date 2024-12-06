#include "stdafx.h"
#include "phase_processing.h"
bool processing_quad::is_supported_quad() const noexcept
{
	const auto retrieval_modes = demosaic_setting::info.at(demosaic).supported_retrieval_modes;
	const auto valid_retrieval = std::find(retrieval_modes.begin(), retrieval_modes.end(), retrieval) != retrieval_modes.end();
	const auto processing_modes = phase_retrieval_setting::settings.at(retrieval).supported_processing_modes;
	const auto valid_processing = std::find(processing_modes.begin(), processing_modes.end(), processing) != processing_modes.end();
	const auto denoise_modes = phase_retrieval_setting::settings.at(retrieval).supported_denoise_modes;
	const auto valid_denoise = std::find(denoise_modes.begin(), denoise_modes.end(), denoise) != denoise_modes.end();
	const auto valid = valid_retrieval && valid_processing && valid_denoise;
#if _DEBUG
	if (!valid)
	{
		const volatile auto what = 0;
	}
#endif
	return valid;
}

[[nodiscard]] bool processing_quad::modulates_slm() const noexcept
{
	//this needs to be fixed to also check the processing modes
	return !phase_processing_setting::settings.at(processing).is_raw_mode;
}


[[nodiscard]] bool processing_quad::has_dpm() noexcept
{
	static auto has_dpm = [] {
		for (const auto& setting : demosaic_setting::info)
		{
			for (auto mode : setting.second.supported_retrieval_modes)
			{
				for (auto processing : phase_retrieval_setting::settings.at(mode).supported_processing_modes)
				{
					const auto is_dpm = phase_processing_setting::settings.at(processing).is_a_dpm;
					if (is_dpm)
					{
						return true;
					}
				}
			}
		}
		return false;
	}();
	return has_dpm;
}

[[nodiscard]] int processing_quad::max_retrieval_input_frames() noexcept
{
	static auto patterns = [] {
		auto max_reasonable_patterns = std::numeric_limits<int>::min();
		for (const auto& setting : demosaic_setting::info)
		{
			for (auto mode : setting.second.supported_retrieval_modes)
			{
				max_reasonable_patterns = std::max(max_reasonable_patterns, phase_retrieval_setting::settings.at(mode).processing_patterns);
			}
		}
		return max_reasonable_patterns;
	}();
	return patterns;
}
