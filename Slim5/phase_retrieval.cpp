#include "stdafx.h"

#include "phase_processing.h"
#include "qli_runtime_error.h"

const phase_retrieval_setting::phase_retrieval_settings phase_retrieval_setting::settings = []
{
	phase_retrieval_setting::phase_retrieval_settings map;
	const auto slim_processing = { phase_processing::phase,phase_processing::mass,phase_processing::height,phase_processing::refractive_index, phase_processing::raw_frames };
	const auto glim_processing = { phase_processing::phase,phase_processing::mutual_intensity,phase_processing::non_interferometric,phase_processing::raw_frames };
	const auto dpm_processing = { phase_processing::diffraction_phase_larger,phase_processing::diffraction_phase, phase_processing::raw_frames };
	const auto polarizer_modes = { phase_processing::quad_pass_through,phase_processing::angles_of_linear_polarization, phase_processing::degree_of_linear_polarization, phase_processing::stoke_0, phase_processing::stoke_1, phase_processing::stoke_2, phase_processing::raw_frames };
	const auto dpm_pol_modes = { phase_processing::quad_pass_through, phase_processing::quad_phase,phase_processing::pol_psi_octo_compute, phase_processing::raw_frames };
	const auto camera_modes = { phase_processing::raw_frames };
	const phase_retrieval_setting_processing::denoise_modes no_denoise_modes = { denoise_mode::off };
	const auto all_denoise_modes = []
	{
		phase_retrieval_setting_processing::denoise_modes modes;
		for (auto idx = 0; idx < static_cast<int>(denoise_mode::count); ++idx)
		{
			modes.insert(static_cast<denoise_mode>(idx));
		}
		return modes;
	}();
	const auto no_denoise = { denoise_mode::off };
	const auto no_demosaic_required = demosaic_requirements();
	const auto four_element_pol = demosaic_requirements(demosaic_mode::polarization_0_45_90_135);
	const auto two_element_pol = demosaic_requirements(demosaic_mode::polarization_0_90);
	map.insert({ phase_retrieval::camera,phase_retrieval_setting("Camera",phase_retrieval_setting_processing(no_demosaic_required,camera_modes,all_denoise_modes),{ slm_mode::single_shot,single_shot,live_cycle_mode::every_frame, false },true) });

	map.insert({ phase_retrieval::darkfield,phase_retrieval_setting("Darkfield",phase_retrieval_setting_processing(no_demosaic_required, camera_modes,no_denoise),{ slm_mode::darkfield,pattern_count_from_file,live_cycle_mode::every_frame,false },false) });

	map.insert({ phase_retrieval::slim,phase_retrieval_setting("SLIM",phase_retrieval_setting_processing(no_demosaic_required, slim_processing,all_denoise_modes),{ slm_mode::slim,typical_psi_patterns,live_cycle_mode::every_frame,false },true) });
	map.insert({ phase_retrieval::slim_demux,phase_retrieval_setting("hrSLIM",phase_retrieval_setting_processing(no_demosaic_required,slim_processing,all_denoise_modes),{ slm_mode::slim,typical_psi_patterns,live_cycle_mode::half_cycle,false },false) });
	map.insert({ phase_retrieval::fpm,phase_retrieval_setting("FPM",phase_retrieval_setting_processing(no_demosaic_required,slim_processing,all_denoise_modes),{ slm_mode::slim,typical_psi_patterns,live_cycle_mode::every_frame,false },true) });
	map.insert({ phase_retrieval::diffraction_phase,phase_retrieval_setting("DPM",phase_retrieval_setting_processing(no_demosaic_required,dpm_processing,all_denoise_modes),{ slm_mode::single_shot,single_shot,live_cycle_mode::every_frame,false },false) });
	map.insert({ phase_retrieval::glim,phase_retrieval_setting("GLIM",phase_retrieval_setting_processing(no_demosaic_required,glim_processing,all_denoise_modes),{ slm_mode::qdic,typical_psi_patterns,live_cycle_mode::every_frame,false },true) });
	map.insert({ phase_retrieval::glim_demux,phase_retrieval_setting("iGLIM",phase_retrieval_setting_processing(no_demosaic_required,glim_processing,all_denoise_modes),{ slm_mode::qdic,typical_psi_patterns,live_cycle_mode::every_frame,false },false) });
	map.insert({ phase_retrieval::polarizer_demux_single,phase_retrieval_setting("Pol",phase_retrieval_setting_processing(four_element_pol,polarizer_modes,all_denoise_modes),{ slm_mode::single_shot,typical_psi_patterns,live_cycle_mode::every_frame,true },false) });
	map.insert({ phase_retrieval::polarizer_demux_psi,phase_retrieval_setting("PolPSI",phase_retrieval_setting_processing(four_element_pol,{ phase_processing::raw_frames },all_denoise_modes),{ slm_mode::qdic,pol_psi_patterns,live_cycle_mode::every_frame,true },false) });
	map.insert({ phase_retrieval::polarizer_demux_two_frame_dpm,phase_retrieval_setting("PolDPM",phase_retrieval_setting_processing(two_element_pol,dpm_pol_modes,all_denoise_modes),{ slm_mode::qdic,pol_two_patterns,live_cycle_mode::every_frame,true },false) });
	map.insert({ phase_retrieval::custom_patterns,phase_retrieval_setting("Custom",phase_retrieval_setting_processing(no_demosaic_required,camera_modes,no_denoise),{ slm_mode::custom_patterns,pattern_count_from_file,live_cycle_mode::every_frame,false },true) });
	//lets make sure the map covered everything, so we don't got no dead code
	for (auto idx = 0; idx < static_cast<int>(phase_processing::count); ++idx)
	{
		const auto& item = static_cast<phase_processing>(idx);
		auto found_mode = false;
		for (const auto& setting : map)
		{
			const auto& processing_mode = setting.second.supported_processing_modes;
			found_mode = std::find(processing_mode.begin(), processing_mode.end(), item) != processing_mode.end();
			if (found_mode)
			{
				break;
			}
		}
		if (!found_mode)
		{
			const auto mode_name = phase_processing_setting::settings.at(item).label;
			qli_runtime_error(mode_name);
		}
	}
	return map;
}();
