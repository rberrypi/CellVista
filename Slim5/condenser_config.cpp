#include "stdafx.h"
#include "condenser_config.h"
#include "device_factory.h"
#include "scope.h"
#include "qli_runtime_error.h"
#include "ui_condenser_config.h"

void condenser_config::update_condenser_position()
{
	const auto nac_position = ui_->cmb_condenser_position_names->currentData();
	ui_->qsb_condenser_nac_setting->setEnabled(nac_position != condenser_position::invalid_nac_position());
	const auto current_state = get_condenser_position();
	emit condenser_settings_changed(current_state);
}

condenser_config::condenser_config(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::condenser_config>();
	ui_->setupUi(this);
	const auto channels = D->scope->get_condenser_settings_names();
	ui_->cmb_condenser_position_names->addItem("Unspecified", condenser_position::invalid_nac_position());
	QObject::connect(ui_->cmb_condenser_position_names, qOverload<int>(&QComboBox::currentIndexChanged), this, &condenser_config::update_condenser_position);
	for (auto chan_idx = 0; chan_idx < channels.size(); ++chan_idx)
	{
		ui_->cmb_condenser_position_names->addItem(channels.at(chan_idx), chan_idx);
	}
	{
		const auto range = D->scope->chan_drive->get_condenser_na_limit();
		ui_->qsb_condenser_nac_setting->setMinimum(range.nac_min);
		ui_->qsb_condenser_nac_setting->setMaximum(range.nac_max);
		const auto steps = 8;
		ui_->qsb_condenser_nac_setting->setSingleStep((range.nac_max - range.nac_min) / steps);
		connect(ui_->qsb_condenser_nac_setting, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &condenser_config::update_condenser_position);  ///qOverload<double>(&QDoubleSpinBox::valueChanged)
	}
	//set_condenser_position(settings);
}


condenser_position condenser_config::get_condenser_position() const
{
	const auto nac = ui_->qsb_condenser_nac_setting->value();
	const auto position = ui_->cmb_condenser_position_names->currentData().toInt();
	return condenser_position(nac, position);
}

void  condenser_config::set_condenser_position(const condenser_position& settings) const
{
	const auto position_idx = ui_->cmb_condenser_position_names->findData(settings.nac_position);
	ui_->cmb_condenser_position_names->setCurrentIndex(position_idx);
	ui_->qsb_condenser_nac_setting->setValue(settings.nac);
	ui_->qsb_condenser_nac_setting->setEnabled(settings.nac_position != condenser_position::invalid_nac_position());
#if _DEBUG
	{
		const auto  what_we_got = get_condenser_position();
		if (!(what_we_got.item_approx_equals(settings)))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void condenser_config::enable_buttons(const bool enable) const
{
	//can this functionality be applied directly on the widget?
	ui_->cmb_condenser_position_names->setEnabled(enable);
	ui_->qsb_condenser_nac_setting->setEnabled(enable);
}