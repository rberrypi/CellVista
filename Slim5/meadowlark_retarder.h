#pragma once
#ifndef MEADOWLARK_RETARDER_H
#define MEADOWLARK_RETARDER_H

#include "slm_device.h"
#include "com_persistent_device.h"

class meadowlark_retarder  final : public slm_device
{
	HANDLE dev1, pipe0, pipe1;
	std::array<BYTE, 128> conversion_buffer;
	std::array<BYTE, 128> status_buffer;
	void issue_command(const std::string& command, bool echo_result, int check_size);
	float char_to_voltage(unsigned char input, float voltage_limit);
	mutable std::recursive_mutex command_issuing_mutex;
public:
	explicit meadowlark_retarder();
	virtual ~meadowlark_retarder();

	std::chrono::microseconds vendor_stability_time() const override
	{
		return std::chrono::microseconds(1000 * 70);
	}

	void hardware_trigger_sequence_internal(size_t capture_items, const channel_settings& channel_settings) override;

	bool has_high_speed_mode() const noexcept override 
	{
		return false;
	}
	void toggle_mode_internal(slm_trigger_mode) override
	{
	}

protected:
	void load_frame_internal(int num) override;
	void set_frame_internal(int num) override;
private:
	static const int internal_slm_patterns = 4;
};

#endif