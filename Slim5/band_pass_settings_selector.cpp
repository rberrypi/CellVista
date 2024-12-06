#include "stdafx.h"
#include "band_pass_settings_selector.h"

#include "qli_runtime_error.h"
#include "ui_band_pass_settings_selector.h"
void band_pass_settings_selector::set_band_pass_toggle(const bool enable) const
{
	ui_->band_filter_min_dx->setEnabled(enable);
	ui_->band_filter_max_dx->setEnabled(enable);
	ui_->band_filter_remove_dc->setEnabled(enable);
	ui_->btn_band_pass_settings_enabled->setChecked(enable);
}

void band_pass_settings_selector::band_pass_settings_update()
{
	const auto item = get_band_pass_settings();
	emit band_pass_settings_changed(item);
}

band_pass_settings_selector::band_pass_settings_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::band_pass_settings>();
	ui_->setupUi(this);
	for (auto qsb : { ui_->band_filter_min_dx,ui_->band_filter_max_dx, })
	{
		connect(qsb, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &band_pass_settings_selector::band_pass_settings_update);
	}
	connect(ui_->band_filter_remove_dc, &QCheckBox::toggled, this, &band_pass_settings_selector::band_pass_settings_update);
	connect(ui_->btn_band_pass_settings_enabled, &QPushButton::toggled,this,&band_pass_settings_selector::set_band_pass_toggle);
	connect(ui_->btn_band_pass_settings_enabled, &QCheckBox::toggled, this, &band_pass_settings_selector::band_pass_settings_update);
}

band_pass_settings band_pass_settings_selector::get_band_pass_settings() const
{
	const auto min_dx = ui_->band_filter_min_dx->value();
	const auto max_dx = ui_->band_filter_max_dx->value();
	const auto remove_dc = ui_->band_filter_remove_dc->isChecked();
	const auto do_band_pass = ui_->btn_band_pass_settings_enabled->isChecked();
	return band_pass_settings(min_dx, max_dx, remove_dc, do_band_pass);
}

void band_pass_settings_selector::set_band_pass_settings(const band_pass_settings& band_pass_settings) const
{
	ui_->band_filter_min_dx->setValue(band_pass_settings.min_dx);
	ui_->band_filter_max_dx->setValue(band_pass_settings.max_dx);
	ui_->band_filter_remove_dc->setChecked(band_pass_settings.remove_dc);
	set_band_pass_toggle(band_pass_settings.do_band_pass);
#if _DEBUG
	{
		const auto what_we_got = get_band_pass_settings();
		if (!what_we_got.item_approx_equals(band_pass_settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}
