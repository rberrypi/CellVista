#pragma once
#ifndef CAPTURE_MODE_H
#define CAPTURE_MODE_H
#include <string>
#include <unordered_map>
enum class capture_mode { sync_capture_sync_io, sync_capture_async_io, async_capture_async_io, burst_capture_async_io };
struct capture_mode_settings final
{
	std::string name;
	bool async_capture, async_io, is_burst, is_hardware_trigger;
	typedef std::unordered_map<capture_mode, const capture_mode_settings> capture_mode_settings_map;
	const static capture_mode_settings_map info;
};

#ifdef QT_DLL
#include <QMetaType> 
Q_DECLARE_METATYPE(capture_mode)
#endif

#endif