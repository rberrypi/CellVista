#include "stdafx.h"
#include "virtual_slm.h"
#include "channel_settings.h"
#include "qli_runtime_error.h"

virtual_slm_device::~virtual_slm_device()
{
	windows_sleep(ms_to_chrono(1));
}
struct virtual_slm_configs : frame_size
{
	bool is_retarder;
	virtual_slm_configs(const int width, const int height, const bool is_retarder) : frame_size{ width,height }, is_retarder(is_retarder){}
};
std::unordered_map<virtual_slm_type, const virtual_slm_configs> virtual_slm_device_settings = {
	{virtual_slm_type::point_retarder,{1,1,true}},
	{virtual_slm_type::medium,{512,512,false}},
	{virtual_slm_type::large,{1920,1080,false}}
};
virtual_slm_device::virtual_slm_device(const virtual_slm_type slm_type) : slm_device(0, 0, false)
{
	auto& settings = virtual_slm_device_settings.at(slm_type);
	static_cast<frame_size&>(*this) = settings;
	this->is_retarder = settings.is_retarder;
}

void virtual_slm_device::load_frame_internal(const int num)
{
	if (num >= frame_data_.size())
	{
		qli_invalid_arguments();
	}
	windows_sleep(ms_to_chrono(10));
}

void virtual_slm_device::set_frame_internal(const  int num)
{
	if (num >= frame_data_.size())
	{
		qli_invalid_arguments();
	}
	windows_sleep(ms_to_chrono(1));
}

void virtual_slm_device::hardware_trigger_sequence_internal(const size_t capture_items, const channel_settings& channel_settings)
{
	const auto frames_to_grab = capture_items * channel_settings.iterator().frame_count();
	const auto wait = frames_to_grab * ms_to_chrono(1);
	windows_sleep(wait);
}