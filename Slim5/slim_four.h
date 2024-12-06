#pragma once
#ifndef SLIM_FOUR_H
#define SLIM_FOUR_H

//Todo put forward declarations 
#include "acquisition_framework.h"
#include <thread>// STD thread is much cleaner
#include <atomic>
#include "qcustomplot.h"
#include "live_settings_store.h"
#include "emphemeral_main_gui_settings.h"
#include "compute_engine_shared.h"
#include <QMainWindow>
#include <memory>
#include "background_update_functors.h"
// ReSharper disable CppInconsistentNaming
class QMediaPlayer;
class QAbstractScrollArea;
// ReSharper restore CppInconsistentNaming

class render_widget;
class full_interface_gui;
#if FUCKFACEGABRIELPOPESCU
class itaSettings;
#endif
class slm_control;
class material_picker;
struct dpm_bg_gpu;
class render_widget;
class camera_device;

namespace Ui {
	class slim_four;
}
class QTabBar;

struct slim_four_hidden_properties
{
	float z_offset;
	int current_pattern;
	material_info material_info;
	slim_four_hidden_properties() noexcept: z_offset(0),current_pattern(0)
	{
		
	}
};
class live_capture_engine;
class slim_four final : public QMainWindow, ephemeral_settings, public live_settings_store
{
	Q_OBJECT

public:
	slim_four_hidden_properties hidden_properties;
	explicit slim_four(QWidget* parent);
	virtual ~slim_four();
	[[nodsicard]] int get_contrast_idx() const;
	[[nodsicard]] live_gui_settings get_live_gui_settings() const;
	[[nodsicard]] QString get_dir() const;
	[[nodsicard]] live_compute_options get_live_compute_options() const;
	[[nodsicard]] live_channels get_current_live_channels();
	[[nodsicard]] live_channels get_common_channels_list() const;

	std::unique_ptr<Ui::slim_four> ui_;

protected:
	void closeEvent(QCloseEvent* event) override;

private:
	std::shared_ptr<compute_engine> compute;
	live_capture_engine* capture_engine;
	//
	
	//Live acquisition logic
	render_widget* render_surface_;
	void setup_render_widget_and_capture_engine();
	//
	[[nodsicard]] QTabBar* get_bg_tab_bar() const;
	//
	void setup_channel_reset();
	//
	void setup_material_settings();
	void setup_itaSettings();
	void close_material_picker() const;
	//
	//Live
	//
	// setup_gui_elements
	void setup_slm_configuration();
	void setup_camera_config();
	void setup_exposure_time();
	void setup_progress_bars() const;
	void setup_auto_contrast() const;
	void setup_live_snapshots();
	void setup_microscope_move();
	void setup_location_update() ;
	void contrast_button_toggle(bool down, int button_idx);
	void refresh_contrast_settings();
	void setup_contrast_buttons();
	void setup_scope_channel();
	void setup_take_phase_background();
	void setup_channel_settings();
	void setup_full_interface();
	void setup_ml_transformer();
	void fuckupITA();
	QTimer* scope_timer;
	//
	[[nodsicard]] render_modifications get_render_modifications() const;
	[[nodsicard]] camera_config get_camera_config() const;
	[[nodsicard]] slim_bg_settings get_slim_bg_settings() const;
	[[nodsicard]] render_settings get_render_settings() const;
	[[nodsicard]] compute_and_scope_settings get_compute_and_scope_settings() const;
	void set_background_frame(const std::shared_ptr<background_frame>& buffer);
	[[nodsicard]] compute_and_scope_settings::background_frame_ptr get_background_frame() const;
	[[nodsicard]] channel_settings get_channel_settings() const;
	void set_slim_bg_settings(const slim_bg_settings& settings);
	void set_render_modifications(const render_modifications& render_modifications) const;
	void set_render_settings(const render_settings& render_settings) const;
	void set_camera_config(const camera_config& settings) const;
	void set_current_pattern(int new_current_pattern);
	void set_compute_and_scope_settings(const compute_and_scope_settings& settings);
	void set_live_gui_settings(const live_gui_settings& settings);
	void press_contrast_button(unsigned int cont) const;
	void save_previous();
	void load_previous();
	//
	//GUI
	QCPItemText* label_;

	//Misc
	slm_control* slm_control_dialog_;
	full_interface_gui* full_interface_dialog_;
	material_picker* material_picker_dialog_;

#if FUCKFACEGABRIELPOPESCU
	itaSettings* popescu_the_romanian_retard;
#endif

signals:
	void repaint_render();
	void set_workspace_directory(const QString& dir);
	void settings_file_changed(const settings_file& settings_file);
	void channel_settings_changed(const channel_settings& settings);
	void live_compute_options_changed(const live_compute_options& options);
	void stop_acquisition();
	void microscope_state_changed(const microscope_state& state);
	
public slots:
	void set_gui_for_acquisition(bool enable_gui);
	//Settings file slots
	void channel_settings_update();
	void live_compute_options_update();
	void start_acquisition_and_write_settings_file(capture_mode capture_mode);

	void set_material_info(const material_info& info);
	//
	void load_auto_contrast_settings(const display_settings::display_ranges& range) const;
	void load_histogram();
	static void show_about();
	void do_calibration(calibration_study_kind mode);
	void close_full_interface() const;
	//
	void enable_bg_tabs(bool enable) const;
	//
	void set_capture_progress(size_t current) const;
	void set_capture_total(size_t total) const;
	void set_io_progress(size_t left) const;
	void set_io_progress_total(size_t total) const;
	void set_io_buffer_progress(size_t current) const;
};

#endif