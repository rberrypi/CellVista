#include "stdafx.h"
#include "full_interface_gui.h"
#include <QFileDialog>
#include <fstream> 
#include <cereal/types/array.hpp>
// ReSharper disable CppUnusedIncludeDirective
#include <cereal/types/vector.hpp>
#include <cereal/types/chrono.hpp>
// ReSharper restore CppUnusedIncludeDirective
#include "ui_full_interface_gui.h"

template <class Archive>
void serialize(Archive& archive, compact_light_path& cc)
{
	archive(
		cereal::make_nvp("processing_quad", cereal::base_class<processing_quad>(&cc)),
		cereal::make_nvp("microscope_light_path", cereal::base_class<microscope_light_path>(&cc)),
		cereal::make_nvp("display_settings", cereal::base_class<display_settings>(&cc)),
		cereal::make_nvp("camera_config", cereal::base_class<camera_config>(&cc)),
		cereal::make_nvp("band_pass_settings", cereal::base_class<band_pass_settings>(&cc)),
		cereal::make_nvp("frames", cc.frames),
		cereal::make_nvp("zee_offset", cc.zee_offset),
		cereal::make_nvp("denoise", cc.denoise),
		cereal::make_nvp("label_suffix", cc.label_suffix)
	);
}
template <class Archive>
void serialize(Archive& archive, full_interface_gui_settings& cc)
{
	archive(
		cereal::make_nvp("light_paths", cc.light_paths),
		cereal::make_nvp("cmb_acquire_modes", cc.cmb_acquire_modes),
		cereal::make_nvp("full_iteration_times", cc.full_iteration_times),
		cereal::make_nvp("interpolate_roi_enabled", cc.interpolate_roi_enabled),
		cereal::make_nvp("interpolate_roi_global_enabled", cc.interpolate_roi_global_enabled),
		cereal::make_nvp("meta_data", cc.meta_data),
		cereal::make_nvp("switch_channel_mode", cc.switch_channel_mode)
	);
}


void full_interface_gui::save_metadata() const
{
	const auto dir = QDir(get_dir());
	if (!dir.isEmpty() && dir.exists())
	{
		const auto file_path = dir.filePath("metadata.txt");
		std::ofstream file;
		file.open(file_path.toStdString());
		const auto data = ui_->metadata->toPlainText();
		file << data.toStdString();
		file.close();
	}

}