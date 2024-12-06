#pragma once
#ifndef LIVE_SETTINGS_STORE
#define LIVE_SETTINGS_STORE
#include <boost/noncopyable.hpp>
#include "channel_settings.h"
class live_settings_store : boost::noncopyable
{
	static live_channels get_default_contrast_settings();
protected:
	live_channels current_contrast_settings_;
	live_channels default_contrast_settings_;
	live_settings_store() noexcept;
	[[nodiscard]] static std::string get_channel_setting_name( int channel_idx) noexcept;
	virtual ~live_settings_store();
};

#endif
