#pragma once
#ifndef ILLUMINATION_POWER_SETTINGS_SELECTOR_H
#define ILLUMINATION_POWER_SETTINGS_SELECTOR_H

#include "modulator_configuration.h"
#include <QWidget>
namespace Ui
{
	class illumination_power_settings_selector;
}

class illumination_power_settings_selector final : public QWidget
{
	Q_OBJECT
	std::unique_ptr<Ui::illumination_power_settings_selector> ui_;	
	void update_illumination_power_settings();
	void set_complete(bool is_complete);
public:
	explicit illumination_power_settings_selector(QWidget *parent = Q_NULLPTR);

	[[nodiscard]] illumination_power_settings get_illumination_power_settings() const;

public slots:
	void set_illumination_power_settings(const illumination_power_settings& power_settings);
	
signals:
	void illumination_power_settings_changed(const illumination_power_settings& power_settings);
};


#endif 
