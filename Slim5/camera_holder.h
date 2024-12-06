#pragma once
#ifndef CAMERA_HOLDER_H
#define CAMERA_HOLDER_H
#include <boost/core/noncopyable.hpp>
#include <vector>

#include "virtual_camera_shared.h"
#include "display_settings.h"
#include "frame_size.h"
#include "phase_processing.h"
class camera_device;
class camera_holder : boost::noncopyable
{
	const static std::string camera_mounts_filename;
public:
	explicit camera_holder(virtual_camera_type camera_type, int slms);
	//todo make unique pointer
	[[nodiscard]] display_range max_raw_display_range() const noexcept;
	typedef std::vector<demosaic_mode> supported_demosaic_modes;
	[[nodiscard]] const supported_demosaic_modes& system_demosaic_modes() const;
	[[nodiscard]] frame_size max_camera_frame_size() const;
	[[nodiscard]] bool camera_with_cooling() const;
	[[nodiscard]] bool has_forced_color_camera() const;// AKA Zeiss where demosaicing is performed by the driver
	[[nodiscard]] bool has_a_color_camera() const;
	[[nodiscard]] int max_samples_per_pixels() const;
	
	//Use this kind of protection for now, eventually move all functions over?

	//camera management up to the call site
	std::vector<camera_device*> cameras;
};
#endif