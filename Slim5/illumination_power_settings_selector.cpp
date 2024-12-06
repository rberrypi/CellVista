#include "stdafx.h"
#include "illumination_power_settings_selector.h"
#include "ui_illumination_power_settings_selector.h"

void illumination_power_settings_selector::update_illumination_power_settings()
{
	const auto value = this->get_illumination_power_settings();
	set_complete(get_illumination_power_settings().is_complete());
	illumination_power_settings_changed(value);
}

illumination_power_settings_selector::illumination_power_settings_selector(QWidget* parent) :QWidget(parent)
{
	ui_ = std::make_unique<Ui::illumination_power_settings_selector>();
	ui_->setupUi(this);
	for (auto* box : { ui_->brightfield_downscale,ui_->illumination_level })
	{
		QObject::connect(box, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &illumination_power_settings_selector::update_illumination_power_settings);
	}
	set_complete(get_illumination_power_settings().is_complete());
}

void illumination_power_settings_selector::set_complete(const bool is_complete)
{
	const auto* color_label = is_complete ? "" : "color: red;";
	for (auto* item :{ ui_->brightfield_downscale,ui_->illumination_level })
	{
		item->setStyleSheet(color_label);
	}
}

illumination_power_settings illumination_power_settings_selector::get_illumination_power_settings() const
{
	const auto power = ui_->illumination_level->value();
	const auto scale = ui_->brightfield_downscale->value();
	return illumination_power_settings(power, scale);
}

void illumination_power_settings_selector::set_illumination_power_settings(const illumination_power_settings& power_settings)
{
	ui_->brightfield_downscale->setValue(power_settings.brightfield_scale_factor);
	ui_->illumination_level->setValue(power_settings.illumination_power);
}
