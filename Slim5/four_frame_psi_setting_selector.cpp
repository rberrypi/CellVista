#include "stdafx.h"
#include "four_frame_psi_setting_selector.h"
#include "ui_four_frame_psi_setting_selector.h"
#include "device_factory.h"
#include "qli_runtime_error.h"
void four_frame_psi_setting_selector::update_four_frame_psi_setting()
{
	const auto value = this->get_four_frame_psi_setting();
	emit four_frame_psi_setting_changed(value);
}

four_frame_psi_setting_selector::four_frame_psi_setting_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::four_frame_psi_setting_selector>();
	ui_->setupUi(this);
	for (auto* box : { ui_->main_value,ui_->bg_value })
	{
		QObject::connect(box, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &four_frame_psi_setting_selector::update_four_frame_psi_setting);
	}
	QObject::connect(ui_->psi_function_pairs, &psi_function_pairs_selector::psi_function_pairs_changed, this, &four_frame_psi_setting_selector::update_four_frame_psi_setting);
	#if _DEBUG
	{
		if (!get_four_frame_psi_setting().is_valid())
		{
			qli_runtime_error();
		}
	}
#endif
}

four_frame_psi_setting four_frame_psi_setting_selector::get_four_frame_psi_setting() const
{
	const auto bg = ui_->bg_value->value();
	const auto value = ui_->main_value->value();
	const auto levels = slm_levels(value, bg);
	const auto weights = ui_->psi_function_pairs->get_psi_function_pairs();
	auto settings =  four_frame_psi_setting(levels, weights);
#if _DEBUG
	if (!settings.is_valid())
	{
		qli_runtime_error();
	}
#endif
	return settings;
}


void four_frame_psi_setting_selector::set_four_frame_psi_setting(const four_frame_psi_setting& modulator_settings)
{
#if _DEBUG
	if (!modulator_settings.is_valid())
	{
		qli_invalid_arguments();
	}
#endif
	ui_->bg_value->setValue(modulator_settings.slm_background);
	ui_->main_value->setValue(modulator_settings.slm_value);
	ui_->psi_function_pairs->set_psi_function_pairs(modulator_settings.weights);
#if _DEBUG
	{
		const auto value = this->get_four_frame_psi_setting();
		if (!value.item_approx_equals(modulator_settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void four_frame_psi_setting_selector::set_slm_mode(const slm_mode& slm_mode)
{
	const auto kill  = slm_mode==slm_mode::slim || slm_mode ==slm_mode::qdic;
	ui_->bg_value->setEnabled(!kill);
}

