#pragma once
#ifndef LIVE_GUI_SETTINGS_CEREALIZATION_H
#define LIVE_GUI_SETTINGS_CEREALIZATION_H
#include "live_gui_settings.h"
#include "boost_cerealization.h"

template <class Archive>
void serialize(Archive& archive, segmentation_feature_bounding& cc)
{
	archive(
		cereal::make_nvp("segmentation_bounding_min", cc.segmentation_bounding_min),
		cereal::make_nvp("segmentation_bounding_max", cc.segmentation_bounding_max)
	);
}

template <class Archive>
void serialize(Archive& archive, segmentation_feature_circularity& cc)
{
	archive(
		cereal::make_nvp("segmentation_circ_min", cc.segmentation_circ_min),
		cereal::make_nvp("segmentation_circ_max", cc.segmentation_circ_max)
	);
}

template <class Archive>
void serialize(Archive& archive, segmentation_feature_area& cc)
{
	archive(
		cereal::make_nvp("segmentation_area_min", cc.segmentation_area_min),
		cereal::make_nvp("segmentation_area_max", cc.segmentation_area_max)
	);
}

template <class Archive>
void serialize(Archive& archive, segmentation_save_settings& cc)
{
	archive(
		cereal::make_nvp("segmentation_keep_originals", cc.segmentation_keep_originals)
	);
}

template <class Archive>
void serialize(Archive& archive, segmentation_settings& cc)
{
	archive(
		make_nvp("segmentation_feature_bounding", cereal::base_class<segmentation_feature_bounding>(&cc)),
		make_nvp("segmentation_feature_circularity", cereal::base_class<segmentation_feature_circularity>(&cc)),
		make_nvp("segmentation_feature_area", cereal::base_class<segmentation_feature_area>(&cc)),
		make_nvp("segmentation_save_settings", cereal::base_class<segmentation_save_settings>(&cc)),
		cereal::make_nvp("segmentation", cc.segmentation),
		cereal::make_nvp("segmentation_min_value", cc.segmentation_min_value)
	);
}
template <class Archive>
void serialize(Archive& archive, display_range& cc)
{
	archive(
		cereal::make_nvp("min", cc.min),
		cereal::make_nvp("max", cc.max)
	);
}

template <class Archive>
void serialize(Archive& archive, display_settings& cc)
{
	archive(
		cereal::make_nvp("ranges", cc.ranges),
		cereal::make_nvp("display_lut", cc.display_lut)
	);
}

template <class Archive>
void serialize(Archive& archive, render_modifications& cc)
{
	archive(
		cereal::make_nvp("show_crosshair", cc.show_crosshair),
		cereal::make_nvp("live_auto_contrast", cc.live_auto_contrast),
		cereal::make_nvp("do_ft", cc.do_ft)
	);
}

template <class Archive>
void serialize(Archive& archive, render_settings& cc)
{
	archive(
		cereal::make_nvp("display_settings", cereal::base_class<display_settings>(&cc)),
		cereal::make_nvp("render_modifications", cereal::base_class<render_modifications>(&cc)));
}

template <class Archive>
void serialize(Archive& archive, condenser_position& cc)
{
	archive(
		cereal::make_nvp("nac", cc.nac),
		cereal::make_nvp("nac_position", cc.nac_position)
	);
}

template <class Archive>
void serialize(Archive& archive, microscope_light_path& cc)
{
	archive(
		cereal::make_nvp("condenser_position", cereal::base_class<condenser_position>(&cc)),
		cereal::make_nvp("scope_channel", cc.scope_channel),
		cereal::make_nvp("light_path", cc.light_path)
	);
}

template <class Archive>
void serialize(Archive& archive, camera_config& cc)
{

	archive(
		cereal::make_nvp("gain_index", cc.gain_index),
		cereal::make_nvp("bin_index", cc.bin_index),
		cereal::make_nvp("aoi_index", cc.aoi_index),
		cereal::make_nvp("camera_idx", cc.camera_idx),
		cereal::make_nvp("mode", cc.mode),
		cereal::make_nvp("enable_cooling", cc.enable_cooling)
	);
}

template <class Archive>
void serialize(Archive& archive, band_pass_settings& cc)
{

	archive(
		cereal::make_nvp("do_band_pass", cc.do_band_pass),
		cereal::make_nvp("remove_dc", cc.remove_dc),
		cereal::make_nvp("min_dx", cc.min_dx),
		cereal::make_nvp("max_dx", cc.max_dx),
		cereal::make_nvp("min_dy", cc.min_dy),
		cereal::make_nvp("max_dy", cc.max_dy)
	);
}

template <class Archive>
void serialize(Archive& archive, slim_bg_settings& cc)
{
	archive(
		cereal::make_nvp("slim_bg_value", cc.slim_bg_value));
}

template <class Archive>
void serialize(Archive& archive, processing_quad& cc)
{
	archive(
		cereal::make_nvp("demosaic", cc.demosaic),
		cereal::make_nvp("processing", cc.processing),
		cereal::make_nvp("retrieval", cc.retrieval),
		cereal::make_nvp("denoise", cc.denoise)
	);
}

template <class Archive>
void serialize(Archive& archive, material_info& cc)
{
	archive(
		cereal::make_nvp("n_cell", cc.n_cell),
		cereal::make_nvp("n_media", cc.n_media),
		cereal::make_nvp("mass_inc", cc.mass_inc),
		cereal::make_nvp("obj_height", cc.obj_height)
	);
}

template <class Archive>
void serialize(Archive& archive, compute_and_scope_settings& cc)
{
	archive(
		make_nvp("render_settings", cereal::base_class<render_settings>(&cc))
		, make_nvp("microscope_light_path", cereal::base_class<microscope_light_path>(&cc))
		, make_nvp("camera_config", cereal::base_class<camera_config>(&cc))
		, make_nvp("band_pass_settings", cereal::base_class<band_pass_settings>(&cc))
		, make_nvp("slim_bg_settings", cereal::base_class<slim_bg_settings>(&cc))
		, make_nvp("processing_quad", cereal::base_class<processing_quad>(&cc))
		, make_nvp("material_info", cereal::base_class<material_info>(&cc))
		, cereal::make_nvp("z_offset", cc.z_offset)
		, cereal::make_nvp("current_pattern", cc.current_pattern)
		, cereal::make_nvp("label_suffix", cc.label_suffix)
	);
}

template <class Archive>
void serialize(Archive& archive, phase_shift_exposure_and_delay& cc)
{
	archive(
		cereal::make_nvp("slm_stability", cc.slm_stability),
		cereal::make_nvp("exposure_time", cc.exposure_time)
	);
}


template <class Archive>
void serialize(Archive& archive, live_gui_settings& cc)
{
	archive(
		cereal::make_nvp("compute_and_scope_settings", cereal::base_class<compute_and_scope_settings>(&cc))
		,cereal::make_nvp("exposures_and_delays", cc.exposures_and_delays)
	);
}

#endif