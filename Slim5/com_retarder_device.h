#pragma once
#ifndef COM_RETARDER_DEVICE_H
#define COM_RETARDER_DEVICE_H

#include "slm_device.h"
#include "com_persistent_device.h"

class com_retarder_device  final : public slm_device, public com_persistent_device
{
	[[nodiscard]] static float char_to_voltage(unsigned char input, float voltage_limit);
public:
	explicit com_retarder_device();
	virtual ~com_retarder_device();

	[[nodiscard]] std::chrono::microseconds vendor_stability_time() const override
	{
		return std::chrono::microseconds(1000);
	}
protected:
	void load_frame_internal(const int) override
	{

	}

	void set_frame_internal(int num) override;
};

#endif