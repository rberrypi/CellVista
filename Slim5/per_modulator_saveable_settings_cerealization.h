#pragma once
#ifndef PER_MODULATOR_SAVEABLE_SETTINGS_CEREALIZATION_H
#define PER_MODULATOR_SAVEABLE_SETTINGS_CEREALIZATION_H
#include "modulator_configuration.h"
#include <cereal/types/chrono.hpp>

template <class Archive>
void serialize(Archive& archive, distorted_donut& cc)
{
	archive(
		cereal::make_nvp("x_center", cc.x_center),
		cereal::make_nvp("y_center", cc.y_center),
		cereal::make_nvp("inner_diameter", cc.inner_diameter),
		cereal::make_nvp("outer_diameter", cc.outer_diameter),
		cereal::make_nvp("ellipticity_e", cc.ellipticity_e),
		cereal::make_nvp("ellipticity_f", cc.ellipticity_f)
	);
}

template <class Archive>
void serialize(Archive& archive, psi_function_pair& cc)
{
	archive(
		cereal::make_nvp("top", cc.top),
		cereal::make_nvp("bot", cc.bot),
		cereal::make_nvp("constant", cc.constant)
	);
}

template <class Archive>
void serialize(Archive& archive, slm_levels& cc)
{
	archive(
		cereal::make_nvp("slm_background", cc.slm_background),
		cereal::make_nvp("slm_value", cc.slm_value)
	);
}

template <class Archive>
void serialize(Archive& archive, phase_shift_pattern& cc)
{
	archive(
		cereal::make_nvp("slm_levels", cereal::base_class<slm_levels>(&cc)),
		cereal::make_nvp("pattern_mode", cc.pattern_mode),
		cereal::make_nvp("filepath", cc.filepath)
	);
}

template <class Archive>
void serialize(Archive& archive, four_frame_psi_setting& cc)
{
	archive(
		cereal::make_nvp("slm_levels", cereal::base_class<slm_levels>(&cc)),
		cereal::make_nvp("weights", cc.weights)
	);
}

template <class Archive>
void serialize(Archive& archive, per_pattern_modulator_settings& cc)
{
	archive(
		cereal::make_nvp("phase_shift_pattern", cereal::base_class<phase_shift_pattern>(&cc)),
		cereal::make_nvp("distorted_donut", cereal::base_class<distorted_donut>(&cc)),
		cereal::make_nvp("weights", cc.weights)
	);
}


template <class Archive>
void serialize(Archive& archive, darkfield_pattern_settings& cc)
{
	archive(
		cereal::make_nvp("modulator_width_na", cc.width_na),
		cereal::make_nvp("ref_ring_na", cc.ref_ring_na),
		cereal::make_nvp("objective_na", cc.objective_na),
		cereal::make_nvp("max_na", cc.max_na),
		cereal::make_nvp("darkfield_display_mode", cc.darkfield_display_mode),
		cereal::make_nvp("invert_modulator_x", cc.invert_modulator_x),
		cereal::make_nvp("invert_modulator_y", cc.invert_modulator_y)
	);
}

template <class Archive>
void serialize(Archive& archive, illumination_power_settings& cc)
{
	archive(
		cereal::make_nvp("brightfield_scale_factor", cc.brightfield_scale_factor),
		cereal::make_nvp("illumination_power", cc.illumination_power)
	);
}

template <class Archive>
void serialize(Archive& archive, modulator_configuration& cc)
{
	archive(
		
		cereal::make_nvp("distorted_donut", cereal::base_class<distorted_donut>(&cc)),
		cereal::make_nvp("darkfield_pattern_settings", cereal::base_class<darkfield_pattern_settings>(&cc)),
		cereal::make_nvp("illumination_power_settings", cereal::base_class<illumination_power_settings>(&cc)),
		cereal::make_nvp("four_frame_psi", cc.four_frame_psi),
		cereal::make_nvp("cycle_internal_delay_us", cc.cycle_internal_delay_us),
		cereal::make_nvp("voltage_max", cc.voltage_max)
	);
}

template <class Archive>
void serialize(Archive& archive, per_modulator_saveable_settings& cc)
{
	
	archive(
		cereal::make_nvp("modulator_configuration", cereal::base_class<modulator_configuration>(&cc)),
		cereal::make_nvp("patterns", cc.patterns),
		cereal::make_nvp("is_alignment", cc.is_alignment)
	);
	
}


#endif;