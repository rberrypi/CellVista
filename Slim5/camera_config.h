#pragma once
#ifndef CAMERA_CONFIG_H
#define CAMERA_CONFIG_H

enum class camera_mode { software, burst, hardware_trigger };
struct camera_config_aoi_camera_pair
{
	int aoi_index, camera_idx;
	camera_config_aoi_camera_pair(const int aoi_index, const int camera_idx) noexcept :aoi_index(aoi_index), camera_idx(camera_idx)
	{
	}
	camera_config_aoi_camera_pair() noexcept : camera_config_aoi_camera_pair(0, 0) {}

};
struct camera_config : camera_config_aoi_camera_pair
{
	//ideally exposure will move into here
	int gain_index, bin_index;//maybe just fuck it and short or char?
	camera_mode mode;//this is a weird property because its typically edited in the acquisition list functions
	bool enable_cooling;
	bool operator== (const camera_config& b) const noexcept
	{
		return gain_index == b.gain_index && bin_index == b.bin_index && aoi_index == b.aoi_index && camera_idx == b.camera_idx && mode == b.mode && enable_cooling == b.enable_cooling;
	}
	bool operator!= (const camera_config& b) const noexcept
	{
		return !(*this == b);
	}
	explicit camera_config(const int gain_index, const int bin_index, const int aoi_index, const camera_mode mode, const bool enable_cooling, const int camera_idx) noexcept : camera_config_aoi_camera_pair(aoi_index, camera_idx), gain_index(gain_index), bin_index(bin_index), mode(mode), enable_cooling(enable_cooling)
	{
	}
	camera_config() noexcept : camera_config(0, 0, 0, camera_mode::software, true, 0) {}

	static camera_config invalid_cam_config();
};

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(camera_config)
Q_DECLARE_METATYPE(camera_config_aoi_camera_pair)
#endif
#endif