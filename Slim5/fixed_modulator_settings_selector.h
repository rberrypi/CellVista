#pragma once
#ifndef FIXED_MODULATOR_SETTINGS_SELECTOR_H
#define FIXED_MODULATOR_SETTINGS_SELECTOR_H

#include "modulator_configuration.h"
#include "per_modulator_saveable_settings_selector.h"
#include <QWidget>
namespace Ui
{
	class fixed_modulator_settings_selector;
}

class fixed_modulator_settings_selector final : public QWidget
{
	Q_OBJECT
	std::unique_ptr<Ui::fixed_modulator_settings_selector> ui_;	
	boost::container::static_vector<per_modulator_saveable_settings_selector*, max_slms > slm_widgets;
	void update_fixed_modulator_settings();
public:
	explicit fixed_modulator_settings_selector(QWidget *parent = Q_NULLPTR);

	[[nodiscard]] fixed_modulator_settings get_fixed_modulator_settings() const;

public slots:
	void set_fixed_modulator_settings(const fixed_modulator_settings& modulator_settings);
	void set_fixed_modulator_settings_silent(const fixed_modulator_settings& modulator_settings);	
	void enable_buttons(bool enable) const;
	void set_pattern(int pattern);
	void set_slm_mode(slm_mode mode);
	void set_darkfield_mode(darkfield_mode mode);

	void reload_modulators();
private:
signals:
	void fixed_modulator_settings_changed(const fixed_modulator_settings& settings);
	void clicked_pattern(int idx);
};


#endif 
