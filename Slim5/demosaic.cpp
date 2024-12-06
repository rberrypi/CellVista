#include "stdafx.h"

#include "phase_processing.h"


const demosaic_setting::demosaic_settings_map demosaic_setting::info = [] {
	demosaic_setting::demosaic_settings_map  map;
	constexpr auto bw_mode = false, color_mode = true;
	constexpr auto one_pattern = 1, two_patterns = 2, four_patterns = 4;
	constexpr auto no_binning = 1, two_by_two_binning = 2;
	const auto fix_up = [](demosaic_setting::retrieval_modes& list)
	{
		if constexpr (REMOVE_SLIM)
		{
			list.erase(phase_retrieval::slim);
		}
		if constexpr (REMOVE_HRSLIM)
		{
			list.erase(phase_retrieval::slim_demux);
		}
		if constexpr (REMOVE_GLIM)
		{
			list.erase(phase_retrieval::glim);
		}
		if constexpr (REMOVE_IGLIM)
		{
			list.erase(phase_retrieval::glim_demux);
		}
		if constexpr (REMOVE_FPM)
		{
			list.erase(phase_retrieval::fpm);
		}
		if constexpr (REMOVE_CUSTOM)
		{
			list.erase(phase_retrieval::custom_patterns);
		}
		if constexpr (REMOVE_POL_PSI)
		{
			list.erase(phase_retrieval::polarizer_demux_psi);
		}
		if constexpr (REMOVE_POL_DPM)
		{
			list.erase(phase_retrieval::polarizer_demux_two_frame_dpm);
		}
		if constexpr (REMOVE_DPM)
		{
			list.erase(phase_retrieval::diffraction_phase);
		}
		if constexpr (REMOVE_POL)
		{
			list.erase(phase_retrieval::polarizer_demux_single);
		}
	};
	demosaic_setting::retrieval_modes all_retrieval_modes;
	for (auto idx = static_cast<int>(phase_retrieval::camera); idx < static_cast<int>(phase_retrieval::last_retrieval); ++idx)
	{
		all_retrieval_modes.insert(static_cast<phase_retrieval>(idx));
	}
	demosaic_setting::retrieval_modes color_modes = { phase_retrieval::camera,phase_retrieval::slim,phase_retrieval::fpm,phase_retrieval::glim };
	demosaic_setting::retrieval_modes polarizer_quad_modes = { phase_retrieval::camera,phase_retrieval::diffraction_phase,phase_retrieval::polarizer_demux_single,phase_retrieval::polarizer_demux_psi };
	demosaic_setting::retrieval_modes polarizer_two_frame_dpm = { phase_retrieval::camera,phase_retrieval::diffraction_phase,phase_retrieval::polarizer_demux_two_frame_dpm };
	//
	fix_up(all_retrieval_modes);
	fix_up(color_modes);
	fix_up(polarizer_quad_modes);
	fix_up(polarizer_two_frame_dpm);
	//
	map.insert({ demosaic_mode::no_processing,demosaic_setting("Off",bw_mode,one_pattern,no_binning,all_retrieval_modes) });
	map.insert({ demosaic_mode::rggb_14_native,demosaic_setting("RGGB",color_mode,one_pattern,no_binning,color_modes) });
	//
	map.insert({ demosaic_mode::polarization_0_45_90_135,demosaic_setting("PolQuad",bw_mode,four_patterns,two_by_two_binning,polarizer_quad_modes) });
	map.insert({ demosaic_mode::polarization_0_90,demosaic_setting("PolHV",bw_mode,two_patterns,two_by_two_binning,polarizer_two_frame_dpm) });
	map.insert({ demosaic_mode::polarization_45_135,demosaic_setting("PolCross",bw_mode,two_patterns,two_by_two_binning,polarizer_two_frame_dpm) });
	return map;
}();

