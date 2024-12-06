#pragma once
#ifndef THORLABS_EOM_H
#define THORLABS_EOM_H

#include "slm_device.h"
#include "com_persistent_device.h"

class thorlabs_eom  final : public slm_device, public com_persistent_device
{
public:
	explicit thorlabs_eom();
	virtual ~thorlabs_eom();

	[[nodiscard]] std::chrono::microseconds vendor_stability_time() const override
	{
		return std::chrono::microseconds(1000 * 70);
	}

	void hardware_trigger_sequence_internal(size_t capture_items, const channel_settings& channel_settings) override;

	[[nodiscard]] bool has_high_speed_mode() const noexcept override
	{
		return false;
	}
	void toggle_mode_internal(slm_trigger_mode) override
	{
	}

protected:
	void set_frame_internal(int num) override;
	void load_frame_internal(const int) override
	{

	}
private:
	static const int internal_slm_patterns = 4;
};

#endif