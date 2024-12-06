#include "stdafx.h"
#if SLM_PRESENT_THORLABSCOM == SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "com_retarder_device.h"

com_retarder_device::com_retarder_device() :slm_device(1, 1, true), com_persistent_device("LCC25", CBR_115200, com_number_unspecified, "\r", "")
{
	const std::string mode = "mode=1";
	const std::string fast_modulation = "freq=150";
	const std::string output_enable = "enable=1";
	const std::string remote_on = "remote=1";
	for (const auto& command : { mode, fast_modulation, output_enable, remote_on })
	{
		com_send(command);
	}
}

com_retarder_device::~com_retarder_device()
{
	const std::string output_disable = "enable=0";
	const std::string remote_off = "remote=0";
	for (const auto& command : { output_disable, remote_off })
	{
		com_send(command);
	}
}

void com_retarder_device::set_frame_internal(const int num)
{
	const auto stored_char = frame_data_.at(num).data.front();
	const auto val = char_to_voltage(stored_char, voltage_max);
	const auto command = "volt1=" + std::to_string(val);
	com_send(command);
}

float com_retarder_device::char_to_voltage(const unsigned char input, const float voltage_limit)
{
	const auto min = 0;
	const auto max = 255;
	const auto a = 0.0;
	const auto b = voltage_limit;//volts
	const auto return_me = (b - a) * (input - min) / (max - min) + a;
	return return_me;
}

#endif