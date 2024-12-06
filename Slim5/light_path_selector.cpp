#include "stdafx.h"
#include "light_path_selector.h"
#include "device_factory.h"
#include "scope.h"
#include "camera_device.h"
#include "qli_runtime_error.h"
#include "ui_light_path_selector.h"

void light_path_selector::update_light_path()
{
	const auto value_to_pass = get_light_path();
	emit light_path_selector_changed(value_to_pass);
}

light_path_selector::light_path_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::light_path_selector>();
	ui_->setupUi(this);

	{
		const auto channels = D->scope->get_channel_settings_names();
		ui_->cmb_scope_channel->addItems(channels);
		connect(ui_->cmb_scope_channel, qOverload<int>(&QComboBox::currentIndexChanged), this, &light_path_selector::update_light_path);
	}
	{
		const auto light_paths = D->scope->get_light_path_names();
		ui_->cmb_light_path->addItems(light_paths);
		connect(ui_->cmb_light_path, qOverload<int>(&QComboBox::currentIndexChanged), this, &light_path_selector::update_light_path);
		const auto hide_light_path = !D->scope->chan_drive->has_light_path;
		if (hide_light_path)
		{
			ui_->cmb_light_path->setHidden(hide_light_path);
		}
	}
	{
		const auto hide_nac_control = !D->scope->chan_drive->has_nac;
		connect(ui_->wdg_condenser_config, &condenser_config::condenser_settings_changed, this, &light_path_selector::update_light_path);
		ui_->wdg_condenser_config->setHidden(hide_nac_control);
	}
}

void light_path_selector::enable_buttons(const bool enable) const
{
	ui_->cmb_scope_channel->setEnabled(enable);
	ui_->cmb_light_path->setEnabled(enable);
	ui_->wdg_condenser_config->setEnabled(enable);
}

void light_path_selector::fixup_light_path(const int camera_idx)
{
	const auto default_light_path_idx = D ? D->cameras.at(camera_idx)->default_light_path_index : 0;
	ui_->cmb_light_path->setCurrentIndex(default_light_path_idx);
}

void light_path_selector::set_light_path_selector(const microscope_light_path& settings) const
{
	ui_->cmb_scope_channel->setCurrentIndex(settings.scope_channel);
	ui_->cmb_light_path->setCurrentIndex(settings.light_path);
	ui_->wdg_condenser_config->set_condenser_position(settings);
#if _DEBUG
	{
		const auto  what_we_got = get_light_path();
		if (!(what_we_got.item_approx_equals(settings)))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void light_path_selector::set_hidden(const bool hidden) const
{
	ui_->cmb_light_path->setHidden(hidden);
	ui_->wdg_condenser_config->setHidden(hidden);
}

microscope_light_path light_path_selector::get_light_path() const
{
	const auto scope_channel_idx = ui_->cmb_scope_channel->currentIndex();
	const auto light_path_idx = ui_->cmb_light_path->currentIndex();
	const auto condenser = ui_->wdg_condenser_config->get_condenser_position();
	return microscope_light_path(scope_channel_idx, light_path_idx, condenser);
}
