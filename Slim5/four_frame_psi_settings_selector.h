#pragma once
#ifndef FOUR_FRAME_PSI_SETTINGS_SELECTOR_H
#define FOUR_FRAME_PSI_SETTINGS_SELECTOR_H

#include "modulator_configuration.h"
#include <QWidget>
namespace Ui
{
	class four_frame_psi_settings_selector;
}
class four_frame_psi_setting_selector;
class four_frame_psi_settings_selector final : public QWidget
{
	Q_OBJECT
	std::unique_ptr<Ui::four_frame_psi_settings_selector> ui_;	
	void update_four_frame_psi_settings();
	std::array<four_frame_psi_setting_selector*,4> selectors;
public:
	explicit four_frame_psi_settings_selector(QWidget *parent = Q_NULLPTR);

	[[nodiscard]] modulator_configuration::four_frame_psi_settings get_four_frame_psi_settings() const;

public slots:
	void set_four_frame_psi_settings(const modulator_configuration::four_frame_psi_settings& modulator_settings);
	void set_four_frame_psi_settings_silent(const modulator_configuration::four_frame_psi_settings& modulator_settings);
	void set_four_layout_direction(bool horizontal);
	void set_slm_mode(slm_mode slm_mode);
signals:
	void four_frame_psi_settings_changed(const modulator_configuration::four_frame_psi_settings& settings);
};


#endif 
