#pragma once
#ifndef SLM_CONTROL_H
#define SLM_CONTROL_H
#include <QMainWindow>

#include "capture_modes.h"
#include "settings_file.h"
#include "acquisition.h"



namespace Ui
{
	class slm_control;
}

class slim_four;
class slm_control final : public QMainWindow
{
	Q_OBJECT

	//By design this doesn't hold SLMs or the device factor, so it needs to be reloaded externally?
	void regenerate_patterns() ;
	slim_four* handle;
	std::unique_ptr<Ui::slm_control> ui_;
	void setup_settings_buttons() ;
	void setup_phase_channel_selection() const;
	void setup_io_settings() const;
	void setup_calibration_buttons();
	void setup_pattern_selection() ;
	void update_settings_file() ;
	std::vector<modulator_configuration> old_modulator_configurations;
	//
	void save_settings_file() const;
	[[nodiscard]] std::string get_path() const;
public:
	virtual ~slm_control();
	explicit slm_control( QWidget* parent=nullptr);
	[[nodiscard]] settings_file get_settings_file() const;

public slots:

	void set_processing_double(const processing_double& processing);
	void set_settings_file(const settings_file& settings);
	void set_pattern(int pattern);
	void write_slm_directory_file() const;
	void reload_modulator_surface() const;
	
signals:
	void do_calibration(calibration_study_kind mode);
	void settings_file_changed(const settings_file& chan, bool is_complete);
	void toggle_draw_dpm(bool draw_dpm_mask);
	void start_capture(capture_mode mode);
	void abort_capture();
	void current_pattern_changed(int pattern);
};

#endif