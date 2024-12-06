#include "stdafx.h"
#if SLM_PRESENT_BNS == SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "bns_device.h"
#include <BNSWrapper_QLI.h>
#pragma comment(lib, "BNSPCIeBoard.lib")
bns_device::bns_device() : slm_device(512, 512, false), hack_(std::make_unique<CBNSWrapper_QLI>())
{
}

void bns_device::set_frame_internal(const int num)
{
	auto* ptr = frame_data_.at(num).data.data();
	hack_->set_pattern(ptr);
}

bns_device::~bns_device() = default;

void bns_device::load_frame_internal(const int)
{
}
#endif