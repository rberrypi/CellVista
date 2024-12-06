#pragma once
#ifndef SETTINGS_FILE_HOLDER_H
#define SETTINGS_FILE_HOLDER_H
#include <QWidget>
#include "settings_file.h"
#include "phase_processing.h"

enum class calibration_study_kind;
enum class capture_mode;

namespace Ui
{
	class slm_settings_holder;
}
class slm_control;
class settings_file_holder final : public QWidget
{
	Q_OBJECT
	settings_file stored_settings_file;
	std::unique_ptr<Ui::slm_settings_holder> ui;
	void reload_settings_file(bool is_complete);
	slm_control* slm_control_dialog_;
	processing_double internal_processing;//unused?
public: 
	explicit settings_file_holder(QWidget* parent);
	[[nodiscard]] const settings_file& get_settings_file() const noexcept;
	[[nodiscard]] std::string get_file_path() const;
	void set_file_path(const std::string& path);
	
public slots:
	void set_processing_double(const processing_double& processing);
	void open_slm_control();
	void close_slm_control();	
	void set_settings_file(const settings_file& file);
	void set_gui_for_acquisition(bool enable);
	
signals:
	void settings_file_is_complete(bool is_complete);
	void resize_exposures(int required_patterns);
	void toggle_draw_dpm(bool draw);
	void start_capture(capture_mode mode);
	void abort_capture();
	void current_pattern_changed(int pattern);
	void settings_file_changed(const settings_file& settings_file);
	void do_calibration(calibration_study_kind mode);
};

#endif