#include "stdafx.h"
#include "settings_file_holder.h"
#include "ui_settings_file_holder.h"
#include "device_factory.h"
#include "qli_runtime_error.h"
#include "slm_control.h"

settings_file_holder::settings_file_holder(QWidget* parent) : QWidget(parent), slm_control_dialog_(nullptr)
{
	ui = std::make_unique<Ui::slm_settings_holder>();
	ui->setupUi(this);
	connect(ui->txtSLM, &QLineEdit::textChanged, [&](const QString& filename)
	{
		auto okay = false;
		auto settings = settings_file::read(filename.toStdString(), okay);
		if (okay)
		{
			settings.modulator_mode = phase_retrieval_setting::settings.at(internal_processing.retrieval).slm_mode;
			settings.regenerate_pattern();
			set_settings_file(settings);
		}
		else
		{
			stored_settings_file.file_path = filename.toStdString();
			set_settings_file(stored_settings_file);
		}
		ui->btnSLMSettings->setEnabled(!filename.isEmpty());
	});
	QObject::connect(ui->btnSLMSettings, &QPushButton::pressed, this, &settings_file_holder::open_slm_control);

}

void settings_file_holder::set_settings_file(const settings_file& file)
{
	if (slm_control_dialog_)
	{
		slm_control_dialog_->set_settings_file(file);
	}
	else
	{
		stored_settings_file = file;
		const auto is_complete = file.is_complete();
		reload_settings_file(is_complete);
	}
}

void settings_file_holder::close_slm_control()
{
	if (slm_control_dialog_)
	{
		slm_control_dialog_->close();
	}
}

void settings_file_holder::set_gui_for_acquisition(const bool enable)
{
	if (slm_control_dialog_)
	{
		slm_control_dialog_->setEnabled(enable);
	}
}

void settings_file_holder::open_slm_control()
{
	if (slm_control_dialog_ == nullptr)
	{
		const auto current_pattern_idx = D->get_slm_frame_idx();
		slm_control_dialog_ = new slm_control;
		connect(slm_control_dialog_, &QMainWindow::destroyed, [&]
		{
			slm_control_dialog_ = nullptr;
		});
		connect(slm_control_dialog_, &slm_control::settings_file_changed, [&](const settings_file& chan, const bool is_complete)
		{
			stored_settings_file = chan;
			reload_settings_file(is_complete);
			slm_control_dialog_->reload_modulator_surface();
		});
		connect(slm_control_dialog_, &slm_control::toggle_draw_dpm, this, &settings_file_holder::toggle_draw_dpm);
		connect(slm_control_dialog_, &slm_control::start_capture, this, &settings_file_holder::start_capture);
		connect(slm_control_dialog_, &slm_control::abort_capture, this, &settings_file_holder::abort_capture);
		connect(slm_control_dialog_, &slm_control::current_pattern_changed, this, &settings_file_holder::current_pattern_changed);
		connect(slm_control_dialog_, &slm_control::do_calibration, this, &settings_file_holder::do_calibration);		
		//connect(this, &settings_file_holder::scan_done, slm_control_dialog_, &slm_control::scan_done);
		slm_control_dialog_->set_settings_file(stored_settings_file);
		slm_control_dialog_->set_processing_double(internal_processing);
		slm_control_dialog_->set_pattern(current_pattern_idx);
	}
	slm_control_dialog_->showNormal();
	slm_control_dialog_->raise();
}

void settings_file_holder::reload_settings_file(const bool is_complete)
{
	D->load_slm_settings(stored_settings_file.modulator_settings, false);
	const auto* color_label = is_complete ? "" : "color: red;";
	const auto pattern_count = stored_settings_file.pattern_count();
	ui->txtSLM->setStyleSheet(color_label);
	emit resize_exposures(pattern_count);
	emit settings_file_changed(stored_settings_file);
	emit settings_file_is_complete(is_complete);
}

const settings_file& settings_file_holder::get_settings_file() const noexcept
{
	return stored_settings_file;
}

void settings_file_holder::set_processing_double(const processing_double& processing)
{
	internal_processing = processing;
	const auto slm_mode = phase_retrieval_setting::settings.at(processing.retrieval).slm_mode;
	stored_settings_file.modulator_mode = slm_mode;
	stored_settings_file.regenerate_pattern();
	const auto is_complete = stored_settings_file.is_complete();
	if (!slm_control_dialog_)
	{
		reload_settings_file(is_complete);
	}
	else
	{
		slm_control_dialog_->set_settings_file(stored_settings_file);
		slm_control_dialog_->set_processing_double(processing);		
	}
}

std::string settings_file_holder::get_file_path() const
{
	return ui->txtSLM->text().toStdString();
}

void settings_file_holder::set_file_path(const std::string& path)
{
	const auto file_path = QString::fromStdString(path);
	ui->txtSLM->setText(file_path);
}

void settings_file::regenerate_pattern()
{
	const auto slm_dimensions = D->get_slm_dimensions();
	const auto samples_per_pixel = D->max_samples_per_pixels();
	regenerate_pattern(slm_dimensions, samples_per_pixel);
}