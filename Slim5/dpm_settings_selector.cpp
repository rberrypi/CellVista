#include "stdafx.h"
#include "dpm_settings_selector.h"
#include "ui_dpm_settings_selector.h"
#include "qli_runtime_error.h"
dpm_settings_selector::dpm_settings_selector(QWidget *parent) : QWidget(parent)
{
	ui_=std::make_unique<Ui::dpm_settings_selector>();
	ui_->setupUi(this);
	for (auto* widgets : { ui_->dpm_phase_left_column ,ui_->dpm_phase_top_row,static_cast<QSpinBox*>(ui_->dpm_phase_width) ,ui_->dpm_amp_left_column,ui_->dpm_amp_top_row ,static_cast<QSpinBox*>(ui_->dpm_amp_width) })
	{
		QObject::connect(widgets, qOverload<int>(&QSpinBox::valueChanged), this, &dpm_settings_selector::update_dpm_settings_selector);
	}
	//QObject::connect(ui_->snap_bg, &QCheckBox::stateChanged, [&] {this->update_dpm_settings_selector(); });
}

void dpm_settings_selector::update_dpm_settings_selector()
{
	const auto value = get_dpm_settings();
	emit dpm_settings_changed(value);
}

dpm_settings dpm_settings_selector::get_dpm_settings() const
{
	const auto dpm_phase_left_column = ui_->dpm_phase_left_column->value();
	const auto dpm_phase_top_row = ui_->dpm_phase_top_row->value();
	const auto dpm_phase_width = ui_->dpm_phase_width->value();
	const auto dpm_amp_left_column = ui_->dpm_amp_left_column->value();
	const auto dpm_amp_top_row = ui_->dpm_amp_top_row->value();
	const auto dpm_amp_width = ui_->dpm_amp_width->value();
	const dpm_settings dpm_settings(dpm_phase_left_column, dpm_phase_top_row, dpm_phase_width, dpm_amp_left_column, dpm_amp_top_row, dpm_amp_width);
	//dpm_settings.dpm_snap_bg = ui_->snap_bg->isChecked();
	return dpm_settings;
}

void dpm_settings_selector::set_dpm_settings_silent(const dpm_settings& settings)
{
	QSignalBlocker sb(*this);
	set_dpm_settings(settings);
}

void dpm_settings_selector::set_dpm_settings(const dpm_settings& settings) const
{
	ui_->dpm_phase_left_column->setValue(settings.dpm_phase_left_column);
	ui_->dpm_phase_top_row->setValue(settings.dpm_phase_top_row);
	ui_->dpm_phase_width->setValue(settings.dpm_phase_width);
	ui_->dpm_amp_left_column->setValue(settings.dpm_amp_left_column);
	ui_->dpm_amp_top_row->setValue(settings.dpm_amp_top_row);
	ui_->dpm_amp_width->setValue(settings.dpm_amp_width);
	//ui_->snap_bg->setChecked(settings.dpm_snap_bg);
#if _DEBUG
	{
		const auto what_we_got = get_dpm_settings();
		if (what_we_got!=settings)
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void dpm_settings_selector::enable_buttons(bool enable) const
{
	qli_not_implemented();
}
