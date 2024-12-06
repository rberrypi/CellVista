#include "stdafx.h"
#include "camera_holder.h"

#include <iostream>

#include "virtual_camera_device.h"
#include "virtual_camera_settings.h"
#if CAMERA_PRESENT_ANDOR == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "andor_device.h"
#endif
#if CAMERA_PRESENT_HAMAMATSU == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "hamamatsu_camera.h"
#endif
#if CAMERA_PRESENT_ZEISSMR == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "zeiss_camera.h"
#endif
#if CAMERA_PRESENT_FLYCAPTURE == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "fly_capture_device.h"
#endif
#if CAMERA_PRESENT_SPINRAKER == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "spinnaker_camera.h"
#endif
#if CAMERA_PRESENT_PCO_PANDA == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include <pco_panda.h>
#endif
#if CAMERA_PRESENT_PCO_EDGE == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include <pco_edge.h>
#endif


camera_holder::camera_holder(const virtual_camera_type camera_type, const int slms)
{
	const auto setup_virtual_cameras = virtual_camera_settings::settings.empty();
	if (setup_virtual_cameras)
	{
		LOGGER_INFO("setting up virtual cameras");
		virtual_camera_settings::transfer_settings_to_test_cameras(slms);
		LOGGER_INFO("Done setting up virtual cameras");
	}
	try
	{
#if CAMERA_PRESENT_VIRTUAL_PSI == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
		{
			cameras.push_back(new virtual_camera_device(camera_type, cameras.size(), NULL));
		}
#endif
#ifdef EXTRA_VIRTUAL_CAMERA
		{
			const auto extra_camera_type = virtual_camera_type::dpm_regular;
			cameras.push_back(new virtual_camera_device(extra_camera_type, cameras.size(), NULL));
		}
#endif
#if  CAMERA_PRESENT_ANDOR == CAMERA_PRESENT|| BUILD_ALL_DEVICES_TARGETS
		{
			cameras.push_back(new andor_device(cameras.size(), nullptr));
		}
#endif
#if  CAMERA_PRESENT_HAMAMATSU == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
		{
			cameras.push_back(new hamamatsu_device(cameras.size()));
		}
#endif
#if  CAMERA_PRESENT_ZEISSMR == CAMERA_PRESENT  || BUILD_ALL_DEVICES_TARGETS
		{
			cameras.push_back(new zeiss_camera(cameras.size(), NULL));
		}
#endif
#if  CAMERA_PRESENT_FLYCAPTURE == CAMERA_PRESENT  || BUILD_ALL_DEVICES_TARGETS
		{
			cameras.push_back(new fly_capture_device(cameras.size()));
		}
#endif
#if  CAMERA_PRESENT_SPINRAKER == CAMERA_PRESENT  || BUILD_ALL_DEVICES_TARGETS
		{
			cameras.push_back(new spinnaker_camera(cameras.size()));
		}
#endif
#if CAMERA_PRESENT_PCO_PANDA == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
		{
			cameras.push_back(new pco_panda(cameras.size()));
		}
#endif
#if CAMERA_PRESENT_PCO_EDGE == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS
		{
			cameras.push_back(new pco_edge(cameras.size()));
		}
#endif
	}
	catch (...)
	{
		if (cameras.empty())
		{
			std::cout << "Failed to connect to physical camera, connecting to virtual" << std::endl;
			auto* camera = new virtual_camera_device(camera_type, cameras.size(), nullptr);
			cameras.emplace_back(camera);
		}
	}
	//this only works once, but oh well
	{
		const auto max_values = max_raw_display_range();
		for (auto& item : phase_processing_setting::settings)
		{
			auto& nuke = item.second.display_range;
			if (nuke == camera_intensity_placeholder)
			{
				nuke = max_values;
			}
		}
	}
}

bool camera_holder::camera_with_cooling() const
{
	const auto cooling_check = [](camera_device* camera) { return camera->has_cooling; };
	const auto has_cooling = std::all_of(cameras.begin(), cameras.end(), cooling_check);
	return has_cooling;
}

frame_size camera_holder::max_camera_frame_size() const
{
	static auto aoi_size = [&]
	{
		const auto max_lambda = [](const camera_device* dev_a, const camera_device* dev_b)
		{
			//might get screwed up
			return dev_a->max_aoi_size().n() < dev_b->max_aoi_size().n();
		};
		const auto* val = *std::max_element(cameras.begin(), cameras.end(), max_lambda);
		return val->max_aoi_size();
	}();
	return aoi_size;
}

display_range camera_holder::max_raw_display_range() const noexcept
{
	display_range range = { std::numeric_limits<float>::max(),-std::numeric_limits<float>::max() };
	for (const auto& camera : cameras)
	{
		const auto camera_range = camera->raw_pixel_range;
		range.max = std::max(camera_range.max, range.max);
		range.min = std::min(camera_range.min, range.min);
	}
	return range;
}

bool camera_holder::has_a_color_camera() const
{
	static auto could_possibly_make_a_color_frame = [&] {
		auto modes = camera_holder::system_demosaic_modes();
		const auto has_a_color_demosaic_mode = std::find(modes.begin(), modes.end(), demosaic_mode::rggb_14_native) != modes.end();
		const auto test = has_forced_color_camera() || has_a_color_demosaic_mode;
		return test;
	}();
	return could_possibly_make_a_color_frame;
}

bool camera_holder::has_forced_color_camera() const
{
	static auto has_forced_color_camera = [&]()
	{
		for (const auto& camera : this->cameras)
		{
			if (camera->is_forced_color())
			{
				return true;
			}
		}
		return false;
	}();
	return has_forced_color_camera;
}

const camera_holder::supported_demosaic_modes& camera_holder::system_demosaic_modes() const
{
	static auto return_me = [&]()
	{
		static std::set<demosaic_mode> modes;
		modes.insert(demosaic_mode::no_processing);
		for (const auto& camera : cameras)
		{
			const auto demosaic_modes = camera->demosaic_modes;
			for (auto demosaic : demosaic_modes)
			{
				modes.insert(demosaic);
			}
		}
		supported_demosaic_modes as_small_vector;
		for (const auto& mode : modes)
		{
			as_small_vector.push_back(mode);
		}
		return as_small_vector;
	}();
	return return_me;
}

[[nodiscard]] int camera_holder::max_samples_per_pixels() const
{
	return has_a_color_camera() ? 3 : 1;
}
// ReSharper disable once CppInconsistentNaming

const std::string camera_holder::camera_mounts_filename = "camera_mount_points.json";
