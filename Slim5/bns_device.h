#pragma once
#ifndef BNSDEVICE
#define BNSDEVICE

#include "slm_device.h"
#include <memory>
// ReSharper disable CppInconsistentNaming
class CBNSWrapper_QLI;
// ReSharper restore CppInconsistentNaming
class bns_device  final : public slm_device
{
public:
	bns_device();
	virtual ~bns_device();

	[[nodiscard]] std::chrono::microseconds vendor_stability_time() const noexcept override 
	{
		//default to the slow SLMs
		return std::chrono::microseconds(30 * 1000);
	}
protected:
	void load_frame_internal(int num) override;
	void set_frame_internal(int num)  override;
private:
	std::unique_ptr<CBNSWrapper_QLI> hack_;
};

#endif
