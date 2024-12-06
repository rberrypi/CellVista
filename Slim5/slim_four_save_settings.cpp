#include "stdafx.h"
#include "slim_four.h"
#include <QFileInfo>
#include "ui_slim_four.h"

const std::string settings_name = "slim4_settings.json";

void slim_four::save_previous()
{
	const auto slm_text = ui_->slm_settings_file->get_file_path();
	const auto last_directory_text = ui_->txtOutputDir->text().toStdString();
	const auto button_idx = this->get_contrast_idx();
	auto settings = ephemeral_settings(slm_text, last_directory_text, button_idx);
	settings.write(settings_name);
	//also 
	current_contrast_settings_.at(button_idx) = get_live_gui_settings();
}

void slim_four::load_previous()
{
	ephemeral_settings settings;
	try
	{
		settings = ephemeral_settings(settings_name);

	}
	catch (...)
	{

	}
	ui_->slm_settings_file->set_file_path(settings.slm_text);
	{
		const auto last_directory = QString::fromStdString(settings.last_directory_text);
		const QFileInfo last_directory_check(last_directory);
		if (last_directory_check.exists() && last_directory_check.isDir())
		{
			ui_->txtOutputDir->setText(last_directory);
		}
	}
	press_contrast_button(settings.last_channel);
}
