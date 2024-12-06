#pragma once
#ifndef FIXED_HARDWARE_SETTINGS_CEREALIZATION_H
#define FIXED_HARDWARE_SETTINGS_CEREALIZATION_H

#include "fixed_hardware_settings.h"
#include "per_modulator_saveable_settings_cerealization.h"
#include "boost_cerealization.h"
#include <cereal/types/array.hpp>

template <class Archive>
void serialize(Archive& archive, pixel_dimensions& cc)
{
	archive(
		cereal::make_nvp("pixel_ratio", cc.pixel_ratio),
		cereal::make_nvp("coherence_length", cc.coherence_length)
	);
}

template <class Archive>
void serialize(Archive& archive, qdic_scope_settings& cc)
{
	archive(
		cereal::make_nvp("qsb_qdic_shear_angle", cc.qsb_qdic_shear_angle),
		cereal::make_nvp("qsb_qdic_shear_dx", cc.qsb_qdic_shear_dx)
	);
}

template <class Archive>
void serialize(Archive& archive, scope_compute_settings& cc)
{
	archive(
		make_nvp("pixel_dimensions", cereal::base_class<pixel_dimensions>(&cc)),
		make_nvp("qdic_scope_settings", cereal::base_class<qdic_scope_settings>(&cc)),
		cereal::make_nvp("objective_attenuation", cc.objective_attenuation),
		cereal::make_nvp("stage_overlap", cc.stage_overlap),
		cereal::make_nvp("wave_lengths", cc.wave_lengths)
	);
}

template <class Archive>
void serialize(Archive& archive, dpm_settings& cc)
{
	//warning if you try to serialize volatile variables the thing crashes, why idk
	archive(
		cereal::make_nvp("dpm_phase_left_column", cc.dpm_phase_left_column),
		cereal::make_nvp("dpm_phase_top_row", cc.dpm_phase_top_row),
		cereal::make_nvp("dpm_phase_width", cc.dpm_phase_width),
		cereal::make_nvp("dpm_amp_left_column", cc.dpm_amp_left_column),
		cereal::make_nvp("dpm_amp_top_row", cc.dpm_amp_top_row),
		cereal::make_nvp("dpm_amp_width", cc.dpm_amp_width)
	);
	//todo check if recent upgrade to cereal fixes this oddity?
	cc.dpm_snap_bg = false;//WARNING if you enable snap bg it will throw a RapidJSONException :-/
}

template <class Archive>
void serialize(Archive& archive, fixed_hardware_settings& cc)
{
	archive(
		cereal::make_nvp("scope_compute_settings", cereal::base_class<scope_compute_settings>(&cc)),
		cereal::make_nvp("dpm_settings", cereal::base_class<dpm_settings>(&cc)),
		cereal::make_nvp("modulator_settings", cc.modulator_settings)
	);
}
#endif