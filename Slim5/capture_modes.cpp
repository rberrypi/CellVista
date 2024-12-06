#include "stdafx.h"
#include "capture_modes.h"


const capture_mode_settings::capture_mode_settings_map capture_mode_settings::info = {
	{ capture_mode::sync_capture_sync_io,{ "Synchronous",false,false,false,false } },
	{ capture_mode::sync_capture_async_io,{ "Asynchronous IO",false,true,false,false } } ,
	{ capture_mode::async_capture_async_io,{ "Full Asynchronous",true,true,false,false } },
	{ capture_mode::burst_capture_async_io,{ "Burst Capture",false,true,true,false } } ,
};