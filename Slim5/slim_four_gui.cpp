#include "stdafx.h"
#include "slim_four.h"
#include "device_factory.h"
#include "camera_device.h"
#include "scope.h"
#include <QMessageBox>
#include <QStringBuilder>
#include <QDirIterator>
#include "ml_transformer.h"
#include "band_pass_settings_selector.h"
#include "ui_slim_four.h"
#include "qli_runtime_error.h"
#include "live_capture_engine.h"
#include "compute_engine.h"

slim_four::slim_four(QWidget* parent) : QMainWindow(parent), capture_engine(nullptr), render_surface_(nullptr), slm_control_dialog_(nullptr), full_interface_dialog_(nullptr), material_picker_dialog_(nullptr)
#if FUCKFACEGABRIELPOPESCU
, popescu_the_romanian_retard(nullptr)
#endif
{
	const auto max_frame_size = D->max_camera_frame_size();
	compute = std::make_shared<compute_engine>(max_frame_size);
	ui_ = std::make_unique<Ui::slim_four>();
	ui_->setupUi(this);
	setFocusPolicy(Qt::WheelFocus);
	QObject::connect(ui_->btnQuit, &QPushButton::clicked, this, &QMainWindow::close);
	QObject::connect(ui_->btnHelp, &QPushButton::clicked, this, &slim_four::show_about);
	setup_render_widget_and_capture_engine();
	setup_scope_channel();
	setup_exposure_time();
	setup_progress_bars();
	setup_camera_config();
	setup_channel_reset();
	setup_auto_contrast();
	setup_live_snapshots();
	setup_location_update();
	setup_contrast_buttons();
	setup_take_phase_background();
	setup_microscope_move();
	setup_full_interface();
	setup_ml_transformer();
	setup_material_settings();
	setup_itaSettings();
	setup_slm_configuration();
	setup_channel_settings();
	//
	{
		const auto index = get_contrast_idx();//typically one
		contrast_button_toggle(true, index);
	}
	load_previous();
	{
		const auto channel = get_channel_settings();
		const auto options = get_live_compute_options();
		capture_engine->begin_live_capture(channel, options);
		QObject::connect(this, &slim_four::channel_settings_changed, capture_engine, &live_capture_engine::set_channel_settings);
		QObject::connect(this, &slim_four::live_compute_options_changed, capture_engine, &live_capture_engine::set_compute_options);
	}
}


int slim_four::get_contrast_idx() const
{
	if (ui_->btnSetOne->isChecked())
	{
		return 0;
	}
	if (ui_->btnSetTwo->isChecked())
	{
		return 1;
	}
	if (ui_->btnSetThree->isChecked())
	{
		return 2;
	}
	if (ui_->btnSetFour->isChecked())
	{
		return 3;
	}
	if (ui_->btnSetFive->isChecked())
	{
		return 4;
	}
	if (ui_->btnSetSix->isChecked())
	{
		return 5;
	}
	if (ui_->btnSetSeven->isChecked())
	{
		return 6;
	}
	if (ui_->btnSetEight->isChecked())
	{
		return 7;
	}
	if (ui_->btnSetNine->isChecked())
	{
		return 8;
	}
	return 0;//error!?
}

// ReSharper disable once CppInconsistentNaming
void slim_four::closeEvent(QCloseEvent* event)
{
	const auto reply = QMessageBox::question(this, "Quit", "Are you sure you want to quit?", QMessageBox::Yes | QMessageBox::No);
	if (reply == QMessageBox::Yes)
	{
		save_previous();
		//close all other windows
		ui_->slm_settings_file->close_slm_control();
		close_full_interface();
		close_material_picker();
		QMainWindow::closeEvent(event);
	}
	else
	{
		event->ignore();
	}
}

void slim_four::press_contrast_button(const unsigned int cont) const
{
	switch (cont)
	{
	case 0: ui_->btnSetOne->click(); break;
	case 1: ui_->btnSetTwo->click(); break;
	case 2: ui_->btnSetThree->click(); break;
	case 3: ui_->btnSetFour->click(); break;
	case 4: ui_->btnSetFive->click(); break;
	case 5: ui_->btnSetSix->click(); break;
	case 6: ui_->btnSetSeven->click(); break;
	case 7: ui_->btnSetEight->click(); break;
	case 8: ui_->btnSetNine->click(); break;
	default:
		qli_invalid_arguments();
	}
}

slim_bg_settings slim_four::get_slim_bg_settings() const
{
	const auto current_idx = get_contrast_idx();
	const auto slim_bg_value = current_contrast_settings_.at(current_idx);
	return slim_bg_settings(slim_bg_value);
}

render_modifications slim_four::get_render_modifications() const
{
	const auto show_crosshair = ui_->btn_cross_hairs->isChecked();
	const auto live_auto_contrast = ui_->btn_live_autocontrast->isChecked();
	const auto do_ft = ui_->btn_live_ft->isChecked();
	return render_modifications(show_crosshair, live_auto_contrast, do_ft);
}

render_settings slim_four::get_render_settings() const
{
	const auto display_settings = ui_->wdg_display_settings->get_display_settings();
	const auto remapper = ui_->wdg_ml_remapper->get_ml_remapper();
	const auto shifter = ui_->wdg_render_shifter->get_render_shifter();
	const auto render_modifications = get_render_modifications();
	auto render = render_settings(render_modifications, display_settings, remapper, shifter);
	return render;
}

compute_and_scope_settings slim_four::get_compute_and_scope_settings() const
{
	const auto live_compute_options = ui_->processing_quad->get_quad();
	const auto microscope_light_path = ui_->wdg_light_path->get_light_path();
	const auto camera_config = ui_->cmb_camera_config->get_camera_config();
	const auto band_pass_settings = ui_->wdg_band_pass_filter->get_band_pass_settings();
	const auto render_settings = get_render_settings();
	const auto slim_bg_settings = get_slim_bg_settings();
	const auto buffer = get_background_frame();
	compute_and_scope_settings compute_and_scope_settings(live_compute_options, render_settings, microscope_light_path, camera_config, band_pass_settings, slim_bg_settings, buffer, hidden_properties.material_info);
	compute_and_scope_settings.fixup_label_suffix();
#if _DEBUG
	if (!compute_and_scope_settings.is_valid())
	{
		qli_gui_mismatch();
	}
#endif
	return compute_and_scope_settings;
}

live_gui_settings slim_four::get_live_gui_settings() const
{
	const auto live_pattern_settings = ui_->wdg_phase_shift_exposures_and_delays->get_exposures_and_delays();
	const auto compute_and_scope_settings = get_compute_and_scope_settings();
	live_gui_settings live_gui_settings(compute_and_scope_settings, live_pattern_settings);
	live_gui_settings.current_pattern = std::min(hidden_properties.current_pattern, static_cast<int>(live_gui_settings.exposures_and_delays.size()) - 1);
#if _DEBUG
	if (!live_gui_settings.is_valid())
	{
		qli_invalid_arguments();
	}
#endif
	return live_gui_settings;
}

channel_settings slim_four::get_channel_settings() const
{
	const auto live_gui_settings = get_live_gui_settings();
	live_gui_settings.assert_validity();
	const auto& hardware_settings = ui_->slm_settings_file->get_settings_file();
	auto settings = channel_settings(hardware_settings, live_gui_settings);
	settings.assert_validity();// don't dump invalid junk
	return settings;
}

void slim_four::set_render_modifications(const render_modifications& render_modifications) const
{
	//show_crosshair, live_auto_contrast, do_radial_average, do_ft
	ui_->btn_cross_hairs->setChecked(render_modifications.show_crosshair);
	ui_->btn_live_autocontrast->setChecked(render_modifications.live_auto_contrast);
	ui_->btn_live_ft->setChecked(render_modifications.do_ft);
#if _DEBUG
	{
		const auto what_we_got = get_render_modifications();
		if (!what_we_got.item_approx_equals(render_modifications))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void slim_four::set_render_settings(const render_settings& render_settings) const
{
	ui_->wdg_display_settings->set_display_settings(render_settings);
	set_render_modifications(render_settings);
	ui_->wdg_ml_remapper->set_ml_remapper(render_settings);
	ui_->wdg_render_shifter->set_render_shifter(render_settings);
#if _DEBUG
	{
		const auto what_we_set = get_render_settings();
		if (!what_we_set.item_approx_equals(render_settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void slim_four::set_slim_bg_settings(const slim_bg_settings& settings)
{
	const auto current_idx = get_contrast_idx();
	current_contrast_settings_.at(current_idx).slim_bg_value = settings.slim_bg_value;
#if _DEBUG
	{
		const auto what_we_got = get_slim_bg_settings();
		if (!what_we_got.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

camera_config slim_four::get_camera_config() const
{
	auto ref_settings = ui_->cmb_camera_config->get_camera_config();
	ref_settings.enable_cooling = ui_->btnCooling->isChecked();
	return ref_settings;
}

void slim_four::set_camera_config(const camera_config& settings) const
{
	ui_->cmb_camera_config->set_camera_config(settings);
	ui_->btnCooling->setChecked(settings.enable_cooling);
#if _DEBUG
	{
		const auto what_we_got = get_camera_config();
		if (what_we_got != settings)
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void slim_four::set_background_frame(const std::shared_ptr<background_frame>& buffer)
{
	if (buffer)
	{
		qli_not_implemented();
	}
}

void slim_four::set_current_pattern(const int new_current_pattern)
{
	const auto current_index = get_contrast_idx();
#if _DEBUG
	{
		const auto valid_pattern = current_contrast_settings_.at(current_index).exposures_and_delays.size() >= new_current_pattern;
		if (!valid_pattern)
		{
			qli_runtime_error();
		}
	}
#endif
	current_contrast_settings_.at(current_index).current_pattern = new_current_pattern;
}

void slim_four::set_compute_and_scope_settings(const compute_and_scope_settings& settings)
{
	set_render_settings(settings);
	ui_->wdg_light_path->set_light_path_selector(settings);
	set_camera_config(settings);
	ui_->wdg_band_pass_filter->set_band_pass_settings(settings);
	set_slim_bg_settings(settings);
	ui_->processing_quad->set_processing(settings);
	set_material_info(settings);
	set_background_frame(settings.background_);
	hidden_properties.current_pattern = settings.current_pattern;
	hidden_properties.z_offset = settings.z_offset;
#if _DEBUG
	{
		const auto what_we_got = get_compute_and_scope_settings();
		if (!what_we_got.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void  slim_four::set_live_gui_settings(const live_gui_settings& settings)
{
	//maybe block all connections before this?
	set_compute_and_scope_settings(settings);
	ui_->wdg_phase_shift_exposures_and_delays->set_phase_shift_exposures_and_delays(settings.exposures_and_delays);
#if _DEBUG
	{
		const auto what_we_got = get_live_gui_settings();
		const auto min_exposure = ui_->wdg_phase_shift_exposures_and_delays->min_time();
		if (!what_we_got.item_approx_equals(settings, min_exposure))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void slim_four::setup_exposure_time()
{
	const auto value_change_functor = [&](const processing_quad& pair)
	{
		ui_->wdg_phase_shift_exposures_and_delays->current_phase_retrieval_changed(pair.retrieval);
	};
	connect(ui_->processing_quad, &processing_quad_selector::processing_quad_changed, value_change_functor);
	connect(ui_->slm_settings_file, &settings_file_holder::resize_exposures, ui_->wdg_phase_shift_exposures_and_delays, &exposure_sequence_control::resize_exposures);
}

void slim_four::setup_progress_bars() const
{
	ui_->progressBarCapture->reset();
	ui_->progressBarIO->reset();
}

void slim_four::setup_channel_reset()
{
	const auto reset_channel_functor = [&]
	{
		const auto idx = get_contrast_idx();
		const auto& contrast_settings = default_contrast_settings_.at(idx);
		set_live_gui_settings(contrast_settings);
	};
	connect(ui_->btn_channel_reset, &QPushButton::clicked, reset_channel_functor);
}

void slim_four::setup_camera_config()
{
	for (auto& camera : D->cameras)
	{
		//actually should be the the last camera...
		connect(camera, &camera_device::min_exp_changed_ts, ui_->wdg_phase_shift_exposures_and_delays, &exposure_sequence_control::set_minimum_exposure_time);// wil always apply the current camera (?)
	}
	{
		const auto& camera = D->cameras.front();
		const auto min_time = camera->get_min_exposure();
		ui_->wdg_phase_shift_exposures_and_delays->set_minimum_exposure_time(min_time);
	}
}

void slim_four::setup_live_snapshots()
{
	const auto clicked_functor = [&]
	{
		const auto dir = ui_->txtOutputDir->text();
		const auto snapshot_text = ui_->snapshot_text->text();
		const auto full_path = QDir(dir).filePath(snapshot_text);
		{
			const auto value = ui_->snapshot_text->get_capture_count();
			ui_->snapshot_text->set_capture_count(value + 1);
		}
		if (QFile(full_path).exists())
		{
			const auto override_text = QString("Are you sure you want to override? %1").arg(full_path);
			const auto reply = QMessageBox::question(this, "Override", override_text, QMessageBox::Yes | QMessageBox::No);
			if (reply != QMessageBox::Yes)
			{
				return;
			}
		}
		const auto msg = gui_message(gui_message_kind::live_image_to_file, QVariant(full_path));
		capture_engine->push_live_message(msg);
	};
	connect(ui_->processing_quad, &processing_quad_selector::processing_quad_changed, ui_->snapshot_text, &snapshot_label::set_processing);
	connect(ui_->btnSnap, &QPushButton::clicked, clicked_functor);

}

void slim_four::contrast_button_toggle(const bool down, const int button_idx)
{
	if (down)
	{
		const auto& new_settings = current_contrast_settings_.at(button_idx);
		set_live_gui_settings(new_settings);
	}
	else
	{
		//how does this actually work? what is the call order?
		const auto to_store = get_live_gui_settings();
		current_contrast_settings_.at(button_idx) = to_store;//store the live settings on release?
	}
}

void slim_four::refresh_contrast_settings()
{
	const auto current_index = get_contrast_idx();
	contrast_button_toggle(false, current_index);
}

void slim_four::setup_contrast_buttons()
{
	connect(ui_->btnSetOne, &QPushButton::toggled, [&](const bool enable) {contrast_button_toggle(enable, 0); });
	connect(ui_->btnSetTwo, &QPushButton::toggled, [&](const bool enable) {contrast_button_toggle(enable, 1); });
	connect(ui_->btnSetThree, &QPushButton::toggled, [&](const bool enable) {contrast_button_toggle(enable, 2); });
	connect(ui_->btnSetFour, &QPushButton::toggled, [&](const bool enable) {contrast_button_toggle(enable, 3); });
	connect(ui_->btnSetFive, &QPushButton::toggled, [&](const bool enable) {contrast_button_toggle(enable, 4); });
	connect(ui_->btnSetSix, &QPushButton::toggled, [&](const bool enable) {contrast_button_toggle(enable, 5); });
	connect(ui_->btnSetSeven, &QPushButton::toggled, [&](const bool enable) {contrast_button_toggle(enable, 6); });
	connect(ui_->btnSetEight, &QPushButton::toggled, [&](const bool enable) {contrast_button_toggle(enable, 7); });
	connect(ui_->btnSetNine, &QPushButton::toggled, [&](const bool enable) {contrast_button_toggle(enable, 8); });
}

void slim_four::setup_scope_channel()
{
	const auto move_channel = [](const microscope_light_path& settings) {
		D->scope->move_light_path(settings, true);
	};
	connect(ui_->wdg_light_path, &light_path_selector::light_path_selector_changed, move_channel);
}

QString slim_four::get_dir() const
{
	return ui_->txtOutputDir->text();
}

void slim_four::setup_ml_transformer()
{
#if INCLUDE_ML==0
	ui_->frame_ml->setHidden(true);
#endif
}

void slim_four::fuckupITA() {
#if FUCKFACEGABRIELPOPESCU
	ui_->btn_ita->setVisible(false);
#endif
}


void slim_four::do_calibration(const calibration_study_kind mode)
{
	const auto settings = get_channel_settings();
	auto [channels, acquisition] = [&] {
		const auto here = D->scope->get_state();
		switch (mode)
		{
		case calibration_study_kind::gray_level_matching:
			return acquisition::generate_gray_level_sequence(settings, here);
		case calibration_study_kind::qdic_shear:
			return acquisition::generate_qdic_shear_sequence(settings, here);
		case calibration_study_kind::take_one:
			return acquisition::generate_take_one_sequence(settings, here);
		default:
			qli_runtime_error();
		}
	}();
	acquisition.output_dir = ui_->txtOutputDir->text().toStdString();
	const auto preflight_info = acquisition.preflight_checks(channels);
	if (!preflight_info.pass)
	{
		return;
	}
	acquisition.assert_valid();
	D->route = acquisition;
	start_acquisition_and_write_settings_file(capture_mode::sync_capture_sync_io);
}

void slim_four::start_acquisition_and_write_settings_file(const capture_mode capture_mode)
{
	auto current_file_settings = ui_->slm_settings_file->get_settings_file();
	const std::filesystem::path current_path = current_file_settings.file_path;
	const auto current_filename = current_path.filename();
	const auto current_directory = std::filesystem::path(D->route.output_dir);
	const auto new_file_path = current_directory / current_filename;
	current_file_settings.file_path = new_file_path.string();
	if (current_file_settings.write())
	{
		capture_engine->start_acquisition(capture_mode);
	}
}
