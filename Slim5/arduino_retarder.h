#pragma once
#ifndef ARDUINO_RETARDER_H
#define ARDUINO_RETARDER_H

#include "slm_device.h"
#include "com_persistent_device.h"

struct arduino_retarder_settings
{
	float coeff_m;
	float coeff_b;

	static float invalid_setting()
	{
		return std::numeric_limits<float>::max();
	}
	arduino_retarder_settings() : coeff_m(invalid_setting()), coeff_b(invalid_setting())
	{

	}

	[[nodiscard]] bool are_retarder_settings_set() const
	{
		return coeff_m != invalid_setting() && coeff_b != invalid_setting();
	}
	static std::string extra_settings_filename()
	{
		return "arduino_retarder_auxiliary_settings.json";
	}
};

class arduino_retarder  final : public slm_device, public com_persistent_device, protected arduino_retarder_settings
{
public:
	explicit arduino_retarder();
	virtual ~arduino_retarder();

	[[nodiscard]] std::chrono::microseconds vendor_stability_time() const override
	{
		return std::chrono::microseconds(1000);
	}

	void hardware_trigger_sequence_internal(size_t capture_items, const channel_settings& channel_settings) override;

	[[nodiscard]] bool has_high_speed_mode() const noexcept override
	{
		return true;
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