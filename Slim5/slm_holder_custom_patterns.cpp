#include "stdafx.h"
#include "slm_holder.h"
#include "slm_device.h"
#include <fstream>
#include <filesystem>
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include "fixed_hardware_settings_cerealization.h"
#include <QProgressDialog>
#include "device_factory.h"
#include "write_tif.h"

// replace with json?

fixed_modulator_settings slm_holder::get_settings() const
{
	fixed_modulator_settings settings;
	for (auto& slm : slms)
	{
		const auto setting = slm->get_modulator_state();
		settings.push_back(setting);
	}
	return settings;
}

std::string get_pattern(const int slm, const int pattern)
{
	return "SLM_" + std::to_string(slm) + "_P" + std::to_string(pattern)+".tif";
}

void slm_holder::write_slm_directory(const std::string& directory, QProgressDialog& progress_dialog)
{
	auto settings = get_settings();
	const auto total_to_write = settings.front().patterns.size() * slms.size();
	progress_dialog.setMaximum(total_to_write);
	const std::filesystem::path file_list_directory(directory);
	const auto pattern_file = file_list_directory / "stored_patterns.json";
	const auto slm_sizes = D->get_slm_dimensions();
	for (auto slm_idx = 0; slm_idx < slms.size(); ++slm_idx)
	{
		const auto& slm = slms.at(slm_idx);
		const auto slm_size = slm_sizes.at(slm_idx);
		const auto patterns = slm->get_frame_number_total();
		for (auto pattern = 0; pattern < patterns; ++pattern)
		{
			auto full_path = file_list_directory / get_pattern(slm_idx,pattern);
			const auto* data = slm->get_frame(pattern);
			const auto actual_path = full_path.make_preferred().string();
			write_tif(actual_path,data,slm_size.width,slm_size.height,1,nullptr);
			//
			{
				if (progress_dialog.wasCanceled())
				{
					return;
				}
				progress_dialog.setValue(1 + progress_dialog.value());
			}
		}
	}
	std::ofstream os(pattern_file.string());
	if (os.is_open())
	{
		cereal::JSONOutputArchive archive(os);
		archive(settings);
	}
}
