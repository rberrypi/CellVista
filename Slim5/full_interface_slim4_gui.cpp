#include "stdafx.h"
#include "full_interface_gui.h"
#include "slim_four.h"
#include "ui_full_interface_gui.h"
[[nodiscard]] QString full_interface_gui::get_dir() const
{
	return slim_four_handle->get_dir();
}

void full_interface_gui::add_common_channel(const int button_idx) const
{
	const auto channel = slim_four_handle->get_common_channels_list().at(button_idx);
	compact_light_path light_path(channel,channel,channel,channel,0.0f,channel.exposures_and_delays,channel,channel.label_suffix);
	ui_->wdg_light_path_holder->add_channel(light_path);
}
