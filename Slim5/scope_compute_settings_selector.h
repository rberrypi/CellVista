#pragma once
#ifndef  SCOPE_COMPUTE_SETTINGS_SELECTOR_H
#define SCOPE_COMPUTE_SETTINGS_SELECTOR_H
class QDoubleSpinBox;
#include "fixed_hardware_settings.h"
#include <QWidget>
namespace Ui
{
	class scope_compute_settings_selector;
}

class scope_compute_settings_selector final : public QWidget
{
	Q_OBJECT

	std::unique_ptr<Ui::scope_compute_settings_selector> ui_;	
	std::array<QDoubleSpinBox*, max_samples_per_pixel > wavelength_widgets;
	void update_scope_compute_settings();
public:
	explicit scope_compute_settings_selector(QWidget *parent = Q_NULLPTR);

	scope_compute_settings get_scope_compute_settings() const;
	wave_length_package get_wave_length_package() const;

public slots:
	void set_scope_compute_settings_silent(const scope_compute_settings& settings);	
	void set_scope_compute_settings(const scope_compute_settings& settings);
	void set_color(bool color);
	void set_wave_length_package(const wave_length_package& package);

signals:
	void scope_compute_settings_changed(const scope_compute_settings& settings);
};


#endif