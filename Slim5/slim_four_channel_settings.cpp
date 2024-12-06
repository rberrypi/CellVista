#include "stdafx.h"
#include "slim_four.h"
#include "ui_slim_four.h"
void slim_four::setup_channel_settings()
{
	QObject::connect(ui_->slm_settings_file, &settings_file_holder::current_pattern_changed, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->slm_settings_file, &settings_file_holder::settings_file_changed, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->wdg_phase_shift_exposures_and_delays, &exposure_sequence_control::phase_shift_exposures_and_delays_changed, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->processing_quad, &processing_quad_selector::processing_quad_changed, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->wdg_light_path, &light_path_selector::light_path_selector_changed, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->cmb_camera_config, &camera_config_selector::camera_config_changed, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->wdg_band_pass_filter, &band_pass_settings_selector::band_pass_settings_changed, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->wdg_display_settings, &display_selector::display_settings_changed, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->wdg_ml_remapper, &ml_remapper_selector::ml_remapper_changed, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->wdg_render_shifter, &render_shifter_selector::ml_render_shifter, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->btn_cross_hairs, &QPushButton::clicked, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->btn_live_autocontrast, &QPushButton::clicked, this, &slim_four::channel_settings_update);
	QObject::connect(ui_->btn_live_ft, &QPushButton::clicked, this, &slim_four::channel_settings_update);
	//SLIM bg value
}

void slim_four::channel_settings_update()
{
	const auto channel_settings = get_channel_settings();
	ui_->snapshot_text->set_label(channel_settings.label_suffix);
	emit channel_settings_changed(channel_settings);
}

void slim_four::live_compute_options_update()
{
	const auto compute_options = get_live_compute_options();
	emit live_compute_options_changed(compute_options);
}
