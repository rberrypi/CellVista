#pragma once
#ifndef PER_MODULATOR_SETTINGS_SELECTOR_H
#define PER_MODULATOR_SETTINGS_SELECTOR_H
#include "modulator_configuration.h"
#include <QGroupBox>
namespace Ui
{
	class per_modulator_saveable_settings_selector;
}

class per_pattern_modulator_settings_patterns_model;
class per_modulator_saveable_settings_selector final : public QGroupBox
{
	Q_OBJECT

	std::unique_ptr<Ui::per_modulator_saveable_settings_selector> ui_;	
	void update_per_modulator_saveable_settings();
	void set_valid_voltage(bool is_valid);
	int slm_id;
	std::string secret_path;
	darkfield_mode darkfield_mode_;
	slm_mode mode_;
	std::unique_ptr<per_pattern_modulator_settings_patterns_model> pattern_model;
	bool block_update_hack;
public:
	explicit per_modulator_saveable_settings_selector(QWidget *parent = Q_NULLPTR);
	virtual ~per_modulator_saveable_settings_selector();
	void reload_modulator();
	[[nodiscard]] per_modulator_saveable_settings get_per_modulator_saveable_settings() const;
	void set_modulator_configuration(const modulator_configuration& configuration);
	void set_modulator_configuration_silent(const modulator_configuration& configuration);	
	[[nodiscard]] modulator_configuration get_modulator_configuration() const;
	void paintEvent(QPaintEvent *event) override;
	bool eventFilter(QObject *object, QEvent *event) override;

public slots:
	void set_darkfield_mode(const darkfield_mode& mode);
	void set_slm_mode(const slm_mode& mode);
	void set_slm_id(int id);
	void set_pattern(int pattern);
	void set_per_modulator_saveable_settings(const per_modulator_saveable_settings& settings);

signals:
	void per_modulator_saveable_settings_changed(const per_modulator_saveable_settings& settings);
	void clicked_pattern(int pattern_idx);
};


#endif 
