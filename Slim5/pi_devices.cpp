#include "stdafx.h"
#include "pi_devices.h"
#if (BODY_TYPE_PI_Z ==BODY_TYPE) || (BUILD_ALL_DEVICES_TARGETS==1)
#pragma comment(lib, "PI_GCS2_DLL_x64.lib")
#include "PI_GCS2_DLL.h"
#include "qli_runtime_error.h"
#include <iostream>
//not thread safe so add a bunch of mutexes?
//Choose to use a DLL instead of RS232 because it has support other connection modalities like TCP/IP
auto sz_axes = "1";
std::vector < std::string> tokenize_a_c_str(char* ptr)
{
	std::vector < std::string> items;
	const auto* tokens = "\n";
	auto* pch = std::strtok(ptr, tokens);
	while (pch != nullptr)
	{
		const std::string phrase(pch);
		items.push_back(phrase);
		std::cout << phrase << std::endl;

		pch = strtok(nullptr, tokens);
	}
	return items;
}

microscope_z_drive_pi::microscope_z_drive_pi() : usb_id_(-1)
{
	//Lets connect via USB?
	std::vector<char> buffers(1024, 0);
	const auto controllers = PI_EnumerateUSB(buffers.data(), buffers.size(), "Piezo");
	if (controllers != 1)
	{
		std::cout << "PI Error: exactly one PI controller (device called 'Piezo') can be plugged in, instead found " << controllers << ", aborting" << std::endl;
		auto device_names = tokenize_a_c_str(buffers.data());
		for (auto& items : device_names)
		{
			std::cout << items << std::endl;
		}
		qli_not_implemented();
	}
	usb_id_ = PI_ConnectUSB(buffers.data());
	if (usb_id_ == (-1))
	{
		std::cout << "PI Error: controller not responding, or already connected" << std::endl;
		qli_not_implemented();
	}
	{
		BOOL pb_value_array = true;
		const auto error = PI_SVO(usb_id_, sz_axes, &pb_value_array);
		if (error == FALSE)
		{
			std::cout << "PI Error: can't set servo" << std::endl;
			qli_not_implemented();
		}
	}
	common_post_constructor();
}

void microscope_z_drive_pi::move_to_z_internal(const float z)
{
	{
		double position = z;
		const auto error_check = PI_MOV(usb_id_, sz_axes, &position);
		if (error_check == FALSE)
		{
			std::cout << "PI Error: can't move to position " << z << std::endl;
		}
	}
	const auto is_on_target = [&]()
	{
		//loop until on-target?
		BOOL status;
		const auto error_check = PI_qONT(usb_id_, sz_axes, &status);
		if (error_check == FALSE)
		{
			std::cout << "PI Error: can't figure out if position is on-target " << std::endl;
		}
		return static_cast<bool>(status);
	};
	const auto attempts = 1000;
	for (auto attempt = 0; attempt < attempts; ++attempt)
	{
		windows_sleep(ms_to_chrono(1));
		const auto on_target = is_on_target();
		if (on_target)
		{
			return;
		}
	}
	std::cout << "PI Error: Position not stabilized" << std::endl;
}

float microscope_z_drive_pi::get_position_z_internal()
{
	auto pos = std::numeric_limits<double>::quiet_NaN();
	const auto error_check = PI_qPOS(usb_id_, sz_axes, &pos);
	if (error_check == FALSE)
	{
		std::cout << "PI Error: can't get current position " << std::endl;
	}
	return pos;
}

scope_limit_z microscope_z_drive_pi::get_z_drive_limits_internal()
{
	const auto get_min = [&]()
	{
		auto min_limit = std::numeric_limits<double>::quiet_NaN();
		const auto error_check = PI_qTMN(usb_id_, sz_axes, &min_limit);
		if (error_check == FALSE)
		{
			std::cout << "PI Error: can't get axis min. limit" << std::endl;
		}
		return min_limit;
	};
	//
	const auto get_max = [&]()
	{
		auto min_limit = std::numeric_limits<double>::quiet_NaN();
		const auto error_check = PI_qTMX(usb_id_, sz_axes, &min_limit);
		if (error_check == FALSE)
		{
			std::cout << "PI Error: can't get axis max. limit" << std::endl;
		}
		return min_limit;
	};
	const auto limits = scope_limit_z(get_min(), get_max(), true);
	return limits;
}

scope_z_drive::focus_system_status microscope_z_drive_pi::get_focus_system_internal()
{
	return scope_z_drive::focus_system_status::off;
}

microscope_z_drive_pi::~microscope_z_drive_pi()
{
	{
		BOOL pb_value_array = false;
		const auto error = PI_SVO(usb_id_, sz_axes, &pb_value_array);
		if (error == FALSE)
		{
			std::cout << "PI Error: can't set servo" << std::endl;
		}
	}
	//how to disconnect? 
}

void microscope_z_drive_pi::print_settings(std::ostream&) noexcept
{

}


#endif