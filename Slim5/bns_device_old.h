#pragma once
#ifndef BNSDEVICE_OLD_H
#define BNSDEVICE_OLD_H

#include "slm_device.h"

class bns_device_old  final : public slm_device
{
public:
	bns_device_old();
	virtual ~bns_device_old();

	[[nodiscard]] std::chrono::microseconds vendor_stability_time() const override
	{
		return ms_to_chrono(30);
	}
protected:
	void load_frame_internal(int num) override;
	void set_frame_internal(int frame_number) override;
};

#endif