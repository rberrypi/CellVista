#pragma once
#ifndef DPM_SETTINGS_SELECTOR_H
#define DPM_SETTINGS_SELECTOR_H
#include "fixed_hardware_settings.h"
#include <QWidget>
namespace Ui
{
	class dpm_settings_selector;
}

class dpm_settings_selector final : public QWidget
{
	Q_OBJECT
	
	std::unique_ptr<Ui::dpm_settings_selector> ui_;	
	void update_dpm_settings_selector();
public:
	explicit dpm_settings_selector(QWidget *parent = Q_NULLPTR);

	[[nodiscard]] dpm_settings get_dpm_settings() const;

public slots:
	void set_dpm_settings_silent(const dpm_settings& settings);
	void set_dpm_settings(const dpm_settings& settings) const;
	void enable_buttons(bool enable) const;

signals:
	void dpm_settings_changed(const dpm_settings& settings);
};


#endif