#pragma once
#ifndef FOUR_FRAME_PSI_SETTING_SELECTOR_H
#define FOUR_FRAME_PSI_SETTING_SELECTOR_H

#include "modulator_configuration.h"
#include <QWidget>
namespace Ui
{
	class four_frame_psi_setting_selector;
}

class four_frame_psi_setting_selector final : public QWidget
{
	Q_OBJECT
	std::unique_ptr<Ui::four_frame_psi_setting_selector> ui_;	
	void update_four_frame_psi_setting();
	
public:
	explicit four_frame_psi_setting_selector(QWidget *parent = Q_NULLPTR);

	[[nodiscard]] four_frame_psi_setting get_four_frame_psi_setting() const;

public slots:
	void set_four_frame_psi_setting(const four_frame_psi_setting& modulator_settings);
	void set_slm_mode(const slm_mode& slm_mode);

signals:
	void four_frame_psi_setting_changed(const four_frame_psi_setting& settings);
};


#endif 
