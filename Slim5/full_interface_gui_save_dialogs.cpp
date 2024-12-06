#include "stdafx.h"
#include "full_interface_gui.h"
#include <QFileDialog>
#include <fstream> 
#include <cereal/types/array.hpp>
// ReSharper disable CppUnusedIncludeDirective
#include <cereal/types/vector.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/archives/json.hpp>
// ReSharper restore CppUnusedIncludeDirective
#include "qli_runtime_error.h"
#include "ui_full_interface_gui.h"
#include "live_gui_settings_cerealization.h"

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
		cereal::make_nvp("switch_channel_mode", cc.switch_channel_mode),
		cereal::make_nvp("filename_grouping", cc.filename_grouping)
	);
}

const std::string full_interface_gui::default_scan_settings_name = "scan_settings.json";

void full_interface_gui::setup_load_save_dialog()
{

	const auto base_directory = get_dir(); // this->get_workspace_path();
	const auto default_file_path = QDir(base_directory).filePath(QString::fromStdString(full_interface_gui::default_scan_settings_name));
	ui_->load_save_dialog->set_text(default_file_path);

	const auto save_button_clicked = [&]()
	{
		const auto file_name = ui_->load_save_dialog->get_path();
		save_cereal_file(file_name);
	};
	QObject::connect(ui_->load_save_dialog, &path_load_save_selector::save_button_clicked, save_button_clicked);

	const auto load_button_clicked = [&]()
	{
		const auto file_name = ui_->load_save_dialog->get_path();
		load_cereal_file(file_name);
		verify_acquire_button();
	};
	QObject::connect(ui_->load_save_dialog, &path_load_save_selector::load_button_clicked, load_button_clicked);
}

void full_interface_gui::load_cereal_file(const QString& path)
{
	std::ifstream os(path.toStdString());
	if (os.is_open())
	{
		cereal::JSONInputArchive archive(os);
		full_interface_gui_settings gui;
		archive(cereal::make_nvp("gui", gui));
		rois_->load_xml(archive);
		set_saveable_settings(gui); //should be called after ROIs are loaded
		for (auto i = 0; i < rois_->data_view_.size(); ++i)
		{
			const auto& roi = rois_->data_view_.at(i);
			if (roi->grid_selected_)
			{
				select_roi(i);
				fit_roi_in_view();
				break;
			}
		}
#if _DEBUG
		{
			const auto what_we_got = get_saveable_settings();
			if (!what_we_got.item_approx_equals(gui))
			{
				qli_gui_mismatch();//FUCK
			}
		}
#endif
	}
}

void full_interface_gui::save_cereal_file(const QString& path)
{
	const auto file_name = path.toStdString();
	{
		std::ofstream os(file_name);
		if (os.is_open())
		{
			auto gui = get_saveable_settings();
			cereal::JSONOutputArchive archive(os);
			archive(cereal::make_nvp("gui", gui));
			rois_->save_xml(archive);
		}
	}
#if _DEBUG
	//test if what we saved matches?
	{
		load_cereal_file(path);
	}
#endif
}