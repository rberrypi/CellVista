#include "stdafx.h"
#include "thorlabs_eom.h"
#if SLM_PRESENT_THORLABS_EOM == SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "com_retarder_device.h"
#include "qli_runtime_error.h"
thorlabs_eom::thorlabs_eom() :slm_device(1, 1, true), com_persistent_device("HVA200", CBR_115200, com_number_unspecified, "\r", "")
{
	const std::string mode = "enable=1";
	for (const auto& command : { mode })
	{
		com_send(command);
	}
}

thorlabs_eom::~thorlabs_eom()
{
	const std::string output_disable = "enable=0";
	for (const auto& command : { output_disable })
	{
		com_send(command);
	}
}

void thorlabs_eom::hardware_trigger_sequence_internal(size_t capture_items, const channel_settings& channel_settings) 
{
	qli_not_implemented();
}

void thorlabs_eom::set_frame_internal(const int num)
{
	const auto stored_char = frame_data_.at(num).data.front();
	//rescale to the unit 0,65535
	const auto scale_value = [](const float min_in, const float max_in, const float min_out, const float max_out, const float value)
	{
		return min_out + (max_out - min_out) * (value - min_in) / (max_in - min_in);
	};
	const auto val_in_dc = scale_value(0, 255, -voltage_max, voltage_max, stored_char);
	const auto val_in_sixteen_bit_internal = static_cast<unsigned short>(std::round(scale_value(-200, 200, 65535, 0, val_in_dc)));
	const auto command = "value=" + std::to_string(val_in_sixteen_bit_internal);
	com_send(command);
	auto volatile test = 0;
}


#endif