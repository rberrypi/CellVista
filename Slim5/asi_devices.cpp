#include "stdafx.h"
#if (STAGE_TYPE==STAGE_TYPE_ASI) ||  (BODY_TYPE==BODY_TYPE_ASI) || BUILD_ALL_DEVICES_TARGETS
#include "asi_devices.h"
#include "com_persistent_device.h"
#include <iostream>

com_persistent_device asi_com_link("ASI_Stage", com_persistent_device::default_buad_rate, com_persistent_device::com_number_unspecified, "\r\n", "");

//note XYZ motion isn't very efficient because we might wait for XY to clear before Z

std::mutex asi_stream_sync;

const float default_conversion = 10;

void poll_move_done()
{
	auto acknowledgment = asi_com_link.pop_com_message();
	const auto done = "N";
	const auto max_checks = 2000;//convert to human time
	for (auto i = 0; i < max_checks; ++i)
	{
		asi_com_link.com_send("/");
		const auto msg = asi_com_link.pop_com_message();
		if (msg == done)
		{
			return;
		}
	}
	std::cout << "Warning waiting for motion, timed out" << std::endl;
}

void microscope_z_drive_asi::move_to_z_internal(const float z)
{
	std::unique_lock<std::mutex> lk(asi_stream_sync);
	const auto size_z = z * micron_to_internal_z_;
	const auto move_command = "M Z=" + std::to_string(size_z);
	asi_com_link.com_send(move_command);
	poll_move_done();
}

float microscope_z_drive_asi::get_position_z_internal()
{
	std::unique_lock<std::mutex> lk(asi_stream_sync);
	float z_pos;
	asi_com_link.com_send("W Z");
	auto data = asi_com_link.pop_com_message();
	const auto back = sscanf_s(data.c_str(), ":A %f", &z_pos, data.size());
	if (back != 1)
	{
		THROW_COM_ERROR();
	}
	return z_pos / micron_to_internal_z_;
}

scope_limit_z microscope_z_drive_asi::get_z_drive_limits_internal()
{
	return {};
}

microscope_z_drive_asi::microscope_z_drive_asi()
{
	{
		std::unique_lock<std::mutex> lk(asi_stream_sync);
		{
			float setme;
			asi_com_link.com_send("UM X ?");
			auto settings = asi_com_link.pop_com_message();
			const auto back = sscanf_s(settings.c_str(), ":X=%f A", &setme, settings.size());
			if (back != 1)
			{
				std::cout << "Warning units on Z drive not supported " << std::endl;
				micron_to_internal_z_ = default_conversion;
			}
			else
			{
				micron_to_internal_z_ = setme / 1000;
			}
		}
	}
	common_post_constructor();
}

void microscope_xy_drive_asi::move_to_xy_internal(const scope_location_xy& xy)
{
	//std::cout << "MOVE XY " << std::endl;
	std::unique_lock<std::mutex> lk(asi_stream_sync);
	// 10000 units  is 1000 microns
	const auto size_x = xy.x * micron_to_internal_x_;
	const auto size_y = xy.y * micron_to_internal_y_;
	const auto move_command = "M X=" + std::to_string(size_x) + " Y=" + std::to_string(size_y);
	asi_com_link.com_send(move_command);
	poll_move_done();
}

scope_location_xy microscope_xy_drive_asi::get_position_xy_internal()
{
	//std::cout << "GET XY " << std::endl;
	std::unique_lock<std::mutex> lk(asi_stream_sync);
	float x_pos, y_pos;
	asi_com_link.com_send("W X Y");
	auto data = asi_com_link.pop_com_message();
	const auto back = sscanf_s(data.c_str(), ":A %f %f", &x_pos, &y_pos, data.size());
	if (back != 2)
	{
		std::cout << "Got back, instead " << data << std::endl;
		THROW_COM_ERROR();
	}
	return { x_pos / micron_to_internal_x_, y_pos / micron_to_internal_y_ };
}

microscope_xy_drive_asi::microscope_xy_drive_asi()
{
	{
		std::unique_lock<std::mutex> lk(asi_stream_sync);
		{
			float setme;
			asi_com_link.com_send("UM X ?");
			auto settings = asi_com_link.pop_com_message();
			const auto back = sscanf_s(settings.c_str(), ":X=%f A", &setme, settings.size());
			if (back != 1)
			{
				std::cout << "Warning units not supported " << std::endl;
				micron_to_internal_x_ = default_conversion;
			}
			else
			{
				micron_to_internal_x_ = setme / 1000;
			}
		}
		{
			float setme;
			asi_com_link.com_send("UM Y ?");
			auto settings = asi_com_link.pop_com_message();
			const auto back = sscanf_s(settings.c_str(), ":Y=%f A", &setme, settings.size());
			if (back != 1)
			{
				std::cout << "Warning units not supported " << std::endl;
				micron_to_internal_y_ = default_conversion;
			}
			else
			{
				micron_to_internal_y_ = setme / 1000;
			}
		}
	}
	common_post_constructor();
}

scope_limit_xy microscope_xy_drive_asi::get_stage_xy_limits_internal()
{
	return {};
}

#endif
