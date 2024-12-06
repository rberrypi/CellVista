#include "stdafx.h"
#include <unordered_map>
#include "phase_processing.h"
#include "display_settings.h"
const display_range typical_phase_range = { -0.3f,0.9f };
const display_range typical_normalized_range = { 0,1 };
std::unordered_map<phase_processing, phase_processing_setting> phase_processing_setting::settings =
{
	{ phase_processing::phase ,{display_settings::jet_lut,typical_phase_range ,"Phase" ,false,false,1,false} },
	{ phase_processing::diffraction_phase_larger ,{ display_settings::jet_lut,typical_phase_range ,"Phase" ,false,false ,1,false } },
	{ phase_processing::diffraction_phase ,{ display_settings::jet_lut,typical_phase_range ,"Phase2" ,true,false ,1,false} },
	{ phase_processing::mass ,{ display_settings::jet_lut,{ 0.f,20.f }, "Mass" ,false ,true ,1,false} },
	{ phase_processing::height ,{ display_settings::jet_lut,{ 0.f,100.f }, "Height",false ,true ,1,false} },
	{ phase_processing::refractive_index ,{ display_settings::jet_lut,{ 0.f,100.f }, "RefIdx",false ,true,1,false} },
	{ phase_processing::mutual_intensity ,{ display_settings::bw_lut,typical_normalized_range, "Mutual" ,false  ,false,1,false} },
	{ phase_processing::quad_pass_through ,{ display_settings::bw_lut,camera_intensity_placeholder, "RawQuad",false  ,false,2,true} },
	{ phase_processing::non_interferometric ,{ display_settings::bw_lut,camera_intensity_placeholder, "NonInt",false ,true,1,false} },
	{ phase_processing::raw_frames ,{ display_settings::bw_lut,camera_intensity_placeholder, "Raw" ,false ,false,1,true} },
	{ phase_processing::degree_of_linear_polarization ,{ display_settings::jet_lut,{ 0.f,1.f }, "DLP" ,false ,false,1,false } },
	{ phase_processing::angles_of_linear_polarization ,{ display_settings::jet_lut,{ -3.15,3.15 }, "ALP" ,false  ,true,1,false } },
	{ phase_processing::stoke_0 ,{ display_settings::bw_lut,typical_normalized_range, "S0" ,false ,true ,1,false } },
	{ phase_processing::stoke_1 ,{ display_settings::bw_lut,typical_normalized_range, "S1",false ,true ,1,false } },
	{ phase_processing::stoke_2 ,{ display_settings::bw_lut,typical_normalized_range, "S2",false, false,1 ,false } },
	{ phase_processing::quad_phase ,{ display_settings::jet_lut,typical_phase_range, "PhaseQuad" ,true,false ,2 ,false } },
	{ phase_processing::pol_psi_octo_compute ,{ display_settings::jet_lut,typical_phase_range, "PolDPM",true ,false,2 ,false } },
};

	[[nodiscard]] bool processing_double::is_raw_frame() const noexcept
	{
		return phase_processing_setting::settings.at(processing).is_raw_mode ;
	}