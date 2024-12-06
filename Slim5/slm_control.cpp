#include "stdafx.h"
#include "slm_control.h"
// ReSharper disable once CppUnusedIncludeDirective
#include <QBitmap>
#include <QSizeGrip>
#include <iostream>
#include <QMessageBox>
#include "device_factory.h"
#include <QFileDialog>
#include <filesystem>
#include "settings_file.h"
#include "ui_slm_control.h"
#include "qli_runtime_error.h"
#include <QProgressDialog>

slm_control::slm_control(QWidget* parent) : QMainWindow(parent), handle(nullptr)
{
	setAttribute(Qt::WA_DeleteOnClose, true);
	ui_ = std::make_unique<Ui::slm_control>();
	ui_->setupUi(this);
	QObject::connect(ui_->fixed_modulator_settings, &fixed_modulator_settings_selector::fixed_modulator_settings_changed, this, &slm_control::regenerate_patterns);
	QObject::connect(ui_->scope_compute_settings, &scope_compute_settings_selector::scope_compute_settings_changed, this, &slm_control::update_settings_file);
	QObject::connect(ui_->dpm_settings, &dpm_settings_selector::dpm_settings_changed, this, &slm_control::update_settings_file);
	QObject::connect(ui_->slm_pattern_options, &slm_pattern_generation_selector::slm_pattern_generation_changed, [&](const slm_pattern_generation& settings)
	{
		ui_->fixed_modulator_settings->set_slm_mode(settings.modulator_mode);
	});
	QObject::connect(ui_->slm_pattern_options, &slm_pattern_generation_selector::slm_pattern_generation_changed, this, &slm_control::regenerate_patterns);
	setup_settings_buttons();
	setup_phase_channel_selection();
	setup_io_settings();
	setup_calibration_buttons();
	setup_pattern_selection();
}

slm_control::~slm_control() = default;

void slm_control::write_slm_directory_file() const
{
	const auto full_path = windowTitle();
	const QFileInfo file(full_path);
	const auto directory = file.absoluteDir();
	if (directory.exists())
	{
		const auto file_path = directory.absolutePath().toStdString();
		auto* widget = this->centralWidget();
		QProgressDialog progress("Writing Patterns", "Abort", 0, 0, widget);
		progress.setWindowModality(Qt::ApplicationModal);
		progress.setWindowFlags(progress.windowFlags() & ~Qt::WindowContextHelpButtonHint);
		D->write_slm_directory(file_path,progress);
	}
}

void slm_control::setup_pattern_selection()
{
	connect(ui_->btnSaveFiles, &QPushButton::clicked, this, &slm_control::write_slm_directory_file);
	const auto pattern_change_functor = [&](const int pattern)
	{
		emit this->current_pattern_changed(pattern);
		reload_modulator_surface();
	};
	connect(ui_->qsbPatternNumber, qOverload<int>(&QSpinBox::valueChanged), pattern_change_functor);
	connect(ui_->fixed_modulator_settings, &fixed_modulator_settings_selector::clicked_pattern, ui_->qsbPatternNumber, &QSpinBox::setValue);
}

void slm_control::reload_modulator_surface() const
{
	const auto pattern_count = D->get_slm_frames()-1;
	const auto pattern = ui_->qsbPatternNumber->value();
	ui_->fixed_modulator_settings->set_pattern(pattern);
	ui_->qsbPatternNumber->setMaximum(pattern_count);
}

void slm_control::set_pattern(const int pattern)
{
	ui_->qsbPatternNumber->setValue(pattern);
}

std::string slm_control::get_path() const
{
	auto path = windowTitle().toStdString();
	return path;
}

settings_file slm_control::get_settings_file() const
{
	const auto fixed_modulator_settings = ui_->fixed_modulator_settings->get_fixed_modulator_settings();
	const auto scope_compute_settings = ui_->scope_compute_settings->get_scope_compute_settings();
	const auto dpm_settings = ui_->dpm_settings->get_dpm_settings();
	const fixed_hardware_settings fixed_hardware_settings(fixed_modulator_settings, scope_compute_settings, dpm_settings);
	const auto slm_modes = ui_->slm_pattern_options->get_slm_pattern_generation();
	const auto path = get_path();
	settings_file settings(fixed_hardware_settings, slm_modes, path);
	return settings;
}

void slm_control::set_processing_double(const processing_double& processing)
{
	ui_->slm_pattern_options->set_processing_double(processing);
}

void slm_control::set_settings_file(const settings_file & settings)
{

#if _DEBUG
	settings.assert_valid();
#endif
	//Even if it triggers a regeneration we should still survive if we do it in the right order
	const auto as_q_string = QString::fromStdString(settings.file_path);
	setWindowTitle(as_q_string);
	{
		const auto current_settings = get_settings_file();
		const auto no_change = current_settings.item_approx_equals(settings);
		const auto debug_item = [](const settings_file& file, const std::string& label)
		{
			const auto pattern_count = file.modulator_settings.empty() ? 0 : file.modulator_settings.front().patterns.size() ;
			std::cout << label << " " << pattern_count<< std::endl;
		};
		debug_item(settings, "input");
		debug_item(current_settings, "Current Settings");
		if (no_change)
		{
			return;
		}
	}
	ui_->slm_pattern_options->set_slm_pattern_generation_silent(settings);
	ui_->fixed_modulator_settings->set_fixed_modulator_settings_silent(settings.modulator_settings);
	ui_->fixed_modulator_settings->set_slm_mode(settings.modulator_mode);	
	ui_->scope_compute_settings->set_scope_compute_settings_silent(settings);
	ui_->dpm_settings->set_dpm_settings_silent(settings);

#if _DEBUG
	{
		//its true a generator can trigger a regeneration but we don't want it to be triggered in this place
		const auto what_we_got = get_settings_file();
		if (!what_we_got.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
	const auto is_complete = settings.is_complete();
	emit settings_file_changed(settings,is_complete);
}

void slm_control::regenerate_patterns()
{
	std::cout << "Regenerating Patterns" << std::endl;
	auto current_data = get_settings_file();
	static_cast<slm_pattern_generation&>(current_data)  = ui_->slm_pattern_options->get_slm_pattern_generation();
	current_data.regenerate_pattern();
	set_settings_file(current_data);
}

void slm_control::setup_settings_buttons()
{
	QObject::connect(ui_->btnSaveSettings, &QPushButton::pressed, this, &slm_control::save_settings_file);
	const auto settings_reload_functor = [&] {
		const auto path = this->get_path();
		bool okay;
		const auto settings_file = settings_file::read(path, okay);
		if (okay)
		{
			set_settings_file(settings_file);
		}
	};
	QObject::connect(ui_->btnLoad, &QPushButton::pressed, settings_reload_functor);
	QObject::connect(ui_->btnRegenerate, &QPushButton::pressed, this,&slm_control::regenerate_patterns);

}

void slm_control::save_settings_file() const
{
	const auto settings = get_settings_file();
	const auto success = settings.write();
	if (!success)
	{
		QMessageBox msg_box;
		msg_box.setText("Failed to save settings");
		msg_box.exec();
	}
}

void slm_control::update_settings_file()
{
	const auto value = get_settings_file();
	const auto is_complete = value.is_complete();
	emit settings_file_changed(value,is_complete);
}

void slm_control::setup_calibration_buttons()
{
	QObject::connect(ui_->btnGrayLevelMatching, &QPushButton::pressed, [&]
	{
		emit do_calibration(calibration_study_kind::gray_level_matching);
	});
	QObject::connect(ui_->btnTakeOneRaw, &QPushButton::pressed, [&]
	{
		emit do_calibration(calibration_study_kind::take_one);
	});
	QObject::connect(ui_->btnGLIMShear, &QPushButton::pressed, [&]
	{
		emit do_calibration(calibration_study_kind::qdic_shear);
	});
}


