#include "stdafx.h"
#include "ui_four_frame_psi_settings_selector.h"
#include "four_frame_psi_settings_selector.h"
#include "qli_runtime_error.h"

void four_frame_psi_settings_selector::update_four_frame_psi_settings()
{
	const auto values = this->get_four_frame_psi_settings();
	emit four_frame_psi_settings_changed(values);
}

four_frame_psi_settings_selector::four_frame_psi_settings_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::four_frame_psi_settings_selector>();
	ui_->setupUi(this);
	selectors = { ui_->psi_one,ui_->psi_two,ui_->psi_three,ui_->psi_four };
	for (auto* pattern : selectors)
	{
		QObject::connect(pattern, &four_frame_psi_setting_selector::four_frame_psi_setting_changed, this, &four_frame_psi_settings_selector::update_four_frame_psi_settings);
	}
	set_four_layout_direction(true);
}

modulator_configuration::four_frame_psi_settings four_frame_psi_settings_selector::get_four_frame_psi_settings() const
{
	modulator_configuration::four_frame_psi_settings settings = {
		ui_->psi_one->get_four_frame_psi_setting(),
		ui_->psi_two->get_four_frame_psi_setting(),
		ui_->psi_three->get_four_frame_psi_setting(),
		ui_->psi_four->get_four_frame_psi_setting(),
	};
	return settings;
}

void four_frame_psi_settings_selector::set_slm_mode(const slm_mode slm_mode)
{
	const auto& patterns = slm_mode_setting::settings.at(slm_mode).patterns;
	for (auto i = 0; i < selectors.size(); ++i)
	{
		const auto enable = i < patterns;
		selectors.at(i)->setEnabled(enable);
		selectors.at(i)->set_slm_mode(slm_mode);
	}
}


void four_frame_psi_settings_selector::set_four_layout_direction(const bool horizontal)
{
	const auto direction  = horizontal ? QBoxLayout::LeftToRight: QBoxLayout::TopToBottom;
	ui_->verticalLayout->setDirection(direction);
}

void four_frame_psi_settings_selector::set_four_frame_psi_settings_silent(const modulator_configuration::four_frame_psi_settings& modulator_settings)
{
	QSignalBlocker blk(*this);
	set_four_frame_psi_settings(modulator_settings);
}

void four_frame_psi_settings_selector::set_four_frame_psi_settings(const modulator_configuration::four_frame_psi_settings& modulator_settings)
{
	const auto original_settings = get_four_frame_psi_settings();
	const auto no_change = modulator_configuration::four_frame_psi_setting_holder_approx_equals(modulator_settings, original_settings);
	if (no_change)
	{
		return;
	}
	const auto is_single_channel = modulator_settings.front().is_single_channel();
	set_four_layout_direction(is_single_channel);
	for (auto i = 0; i < modulator_settings.size(); ++i)
	{
		auto* selector = selectors.at(i);
		QSignalBlocker blocker(selector);
		const auto& setting = modulator_settings.at(i);
		selector->set_four_frame_psi_setting(setting);
	}
	update_four_frame_psi_settings();
#if _DEBUG
	{
		const auto what_we_got = get_four_frame_psi_settings();
		if (!modulator_configuration::four_frame_psi_setting_holder_approx_equals(what_we_got, modulator_settings))
		{
			qli_runtime_error();
		}
	}
#endif
}