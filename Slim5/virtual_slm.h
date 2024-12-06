#pragma once
#ifndef VIRTUAL_SLM_H
#define VIRTUAL_SLM_H

#include "slm_device.h"
class virtual_slm_device  final : public slm_device
{
public:
	explicit virtual_slm_device(virtual_slm_type slm_type);
	virtual ~virtual_slm_device();

	[[nodiscard]] std::chrono::microseconds vendor_stability_time() const override
	{
		return ms_to_chrono(35);
	}

	[[nodiscard]] bool has_high_speed_mode() const noexcept override
	{
		return true;
	}
protected:
	void load_frame_internal(int num) override;
	void set_frame_internal(int num) override;

	void toggle_mode_internal(slm_trigger_mode) override
	{

	}

	void hardware_trigger_sequence_internal(size_t capture_items, const channel_settings& channel_settings) override;
};

#endif