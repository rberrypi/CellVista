#include "stdafx.h"
#include "camera_config.h"

camera_config camera_config::invalid_cam_config()
{
	constexpr auto invalid_gain = std::numeric_limits<decltype(gain_index)>::max();
	constexpr auto invalid_bin = std::numeric_limits<decltype(bin_index)>::max();
	constexpr auto invalid_roi = std::numeric_limits<decltype(aoi_index)>::max();
	constexpr auto invalid_idx = std::numeric_limits<decltype(camera_idx)>::max();
	constexpr auto enable_cooling = true;
	return camera_config(invalid_gain, invalid_bin, invalid_roi, camera_mode::software, enable_cooling, invalid_idx);
}