#include "stdafx.h"
#include "slim_four.h"
#include "device_factory.h"
#include "camera_device.h"
#include "qli_runtime_error.h"

live_channels slim_four::get_current_live_channels()
{
	const auto button = get_contrast_idx();
	contrast_button_toggle(false, button);
	for (auto& channel : current_contrast_settings_)
	{
		if (!channel.is_valid())
		{
			qli_not_implemented();
		}
	}
	return current_contrast_settings_;
}

live_channels slim_four::get_common_channels_list() const
{
	return current_contrast_settings_;
}
