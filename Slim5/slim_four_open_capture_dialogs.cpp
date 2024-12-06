#include "stdafx.h"
#include "slim_four.h"
#include "full_interface_gui.h"
#include "itaSettings.h"
#include "device_factory.h"
#include "scope.h"
#include "ui_slim_four.h"
#include "live_capture_engine.h"
#include "qt_layout_disable.h"
#include <filesystem>

void slim_four::setup_full_interface()
{
#if HIDE_AUTOMATED_SCANNING == 1
	ui_->btnFullInterface->setHidden(true);
#endif

	const auto open_full_interface_functor = [&]
	{
		using std::cout;
		using std::endl;

		if (full_interface_dialog_ == nullptr)
		{
			cout << "automated scanning dialog is not present, creating..." << endl;
			refresh_contrast_settings();
			const auto initial_settings = get_live_gui_settings();
			const auto file_settings = ui_->slm_settings_file->get_settings_file();
			full_interface_dialog_ = new full_interface_gui(initial_settings, file_settings, this, nullptr);
			connect(full_interface_dialog_, &full_interface_gui::start_acquisition, this, &slim_four::start_acquisition_and_write_settings_file);
			connect(full_interface_dialog_, &full_interface_gui::stop_acquisition, this, &slim_four::stop_acquisition);
			connect(this, &slim_four::microscope_state_changed, full_interface_dialog_, &full_interface_gui::set_microscope_state);
			connect(capture_engine, &live_capture_engine::gui_enable, full_interface_dialog_, &full_interface_gui::gui_enable);
			connect(this, &slim_four::settings_file_changed, full_interface_dialog_, &full_interface_gui::set_file_settings);
			connect(D->scope.get(), &microscope::focus_system_engaged, full_interface_dialog_, &full_interface_gui::focus_system_engaged);
			connect(full_interface_dialog_, &QMainWindow::destroyed, [&]
			{
				full_interface_dialog_ = nullptr;
			});

		}
		full_interface_dialog_->showMaximized();
		full_interface_dialog_->raise();
	};
	connect(ui_->btnFullInterface, &QPushButton::clicked, open_full_interface_functor);
}

void slim_four::setup_itaSettings()
{
#if FUCKFACEGABRIELPOPESCU
	LOGGER_INFO("setup_itaSettings");
	if (!popescu_the_romanian_retard)
		popescu_the_romanian_retard = new itaSettings(this, nullptr);

	connect(ui_->btn_ita, &QPushButton::clicked, [&]() {
		popescu_the_romanian_retard->showNormal();
		popescu_the_romanian_retard->raise();
		});

	LOGGER_INFO("setup_itaSettings Done!");
#endif
}

void slim_four::close_full_interface() const
{
	if (full_interface_dialog_)
	{
		full_interface_dialog_->close();
	}
}

void slim_four::setup_slm_configuration()
{
	QObject::connect(ui_->processing_quad, &processing_quad_selector::processing_quad_changed, ui_->slm_settings_file, &settings_file_holder::set_processing_double);
	const auto default_value = ui_->processing_quad->get_quad();
	ui_->slm_settings_file->set_processing_double(default_value);
	const auto set_settings_file_validity = [&](const bool is_valid)
	{
		enable_layout(ui_->grdContrast, is_valid);
	};
	const auto set_workspace_validity = [&]()
	{
		const auto settings_validity_hack = ui_->grdContrast->itemAt(0)->widget()->isEnabled();
		const auto validity = !ui_->txtOutputDir->text().isEmpty() && settings_validity_hack;
		ui_->btnFullInterface->setEnabled(validity);
		ui_->btnSnap->setEnabled(validity);
	};
	connect(ui_->slm_settings_file, &settings_file_holder::settings_file_is_complete, set_settings_file_validity);
	connect(ui_->slm_settings_file, &settings_file_holder::settings_file_is_complete, set_workspace_validity);
	connect(ui_->txtOutputDir, &folder_line_edit::textChanged, set_workspace_validity);
	set_settings_file_validity(false);
	set_workspace_validity();
	//
	const auto pattern_change_functor = [&](const int pattern)
	{
		this->hidden_properties.current_pattern = pattern;
	};
	connect(ui_->slm_settings_file, &settings_file_holder::current_pattern_changed, pattern_change_functor);
	connect(ui_->slm_settings_file, &settings_file_holder::do_calibration, this, &slim_four::do_calibration);
}
