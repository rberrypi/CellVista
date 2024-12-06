#include "stdafx.h"
#include "slm_holder.h"
#include "virtual_slm.h"
#include <algorithm>
#include <iostream>
#include "qli_runtime_error.h"

#if SLM_PRESENT_BNS==SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "bns_device.h"
#endif
#if SLM_PRESENT_BNS_ANCIENT==SLM_PRESENT|| BUILD_ALL_DEVICES_TARGETS
#include "bns_device_old.h"
#endif
#if SLM_PRESENT_MONITOR==SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "dvi_slm_device.h"
#endif
#if SLM_PRESENT_THORLABSCOM==SLM_PRESENT|| BUILD_ALL_DEVICES_TARGETS
#include "com_retarder_device.h"
#endif
#if SLM_PRESENT_ARDUINOCOM==SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "arduino_retarder.h"
#endif
#if SLM_PRESENT_MEADOWLARK_RETARDER==SLM_PRESENT || SLM_PRESENT_MEADOWLARK_HS_RETARDER==SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "meadowlark_retarder.h"
#endif
#if SLM_PRESENT_THORLABS_EOM==SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
#include "thorlabs_eom.h"
#endif
slm_holder::slm_holder()
{
	try
	{
#if SLM_PRESENT_VIRTUAL==SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
		slms.push_back(std::make_unique<virtual_slm_device>(virtual_slm_type::medium));
#endif
#if SLM_PRESENT_BNS==SLM_PRESENT || BUILD_ALL_DEVICES_TARGETS
		slms.push_back(std::make_unique<bns_device>());
#endif
#if SLM_PRESENT_BNS_ANCIENT==SLM_PRESENT  || BUILD_ALL_DEVICES_TARGETS
		slms.push_back(std::make_unique<bns_device_old>());
#endif
#if SLM_PRESENT_THORLABSCOM==SLM_PRESENT  || BUILD_ALL_DEVICES_TARGETS
		slms.push_back(std::make_unique<com_retarder_device>());
#endif
#if SLM_PRESENT_MONITOR==SLM_PRESENT  || BUILD_ALL_DEVICES_TARGETS
		slms.push_back(std::make_unique<dvi_slm_device>());
#endif
#if SLM_PRESENT_ARDUINOCOM==SLM_PRESENT  || BUILD_ALL_DEVICES_TARGETS
		slms.push_back(std::make_unique<arduino_retarder>());
#endif
#if SLM_PRESENT_MEADOWLARK_RETARDER==SLM_PRESENT || SLM_PRESENT_MEADOWLARK_HS_RETARDER==SLM_PRESENT
		slms.push_back(std::make_unique<meadowlark_retarder>());
#endif
#if SLM_PRESENT_THORLABS_EOM==SLM_PRESENT  || BUILD_ALL_DEVICES_TARGETS
		slms.push_back(std::make_unique<thorlabs_eom>());
#endif
#ifdef EXTRA_VIRTUAL_SLM
		slms.push_back(std::make_unique<virtual_slm_device>(virtual_slm_type::large));
#endif
	}
	catch (...)
	{
		const auto* const msg = "Couldn't get SLM, defaulting to virtual device";
		std::cout << msg << std::endl;
		slms.push_back(std::make_unique<virtual_slm_device>(virtual_slm_type::medium));
	}
	if (slms.empty())
	{
		qli_runtime_error();
	}
}

slm_holder::~slm_holder() = default;

boost::container::small_vector<bool, 2> slm_holder::has_retarders() const
{
	const auto static cache = [&] {
		boost::container::small_vector<bool, 2>	info;
		for (const auto& slm : slms)
		{
			info.push_back(slm->is_retarder);
		}
		return info;
	}();
	return cache;
}

void slm_holder::slm_consistency_check() const
{
#if _DEBUG		
	const auto print_slm_info = [&]
	{
		for (auto& slm : slms)
		{
			const auto mode = slm->get_modulator_state();
		}
	};
	//all SLMs have same number of frames
	{
		const auto frames_loaded = slms.front()->get_frame_number_total();
		for (auto slm_idx = 1; slm_idx < slms.size(); ++slm_idx)
		{
			const auto frames_expected = slms.at(slm_idx)->get_frame_number_total();
			if ((frames_loaded != frames_expected))
			{
				print_slm_info();
				qli_runtime_error();
			}
		}
	}
	//SLMs are showing the same frame
	for (auto slm_idx = 1; slm_idx < slms.size(); ++slm_idx)
	{
		const auto frames_expected = slms.front()->get_frame_number();
		const auto frames_loaded = slms.at(slm_idx)->get_frame_number();
		const auto is_uninitialized = frames_expected == slm_state::uninitialized_position || frames_loaded == slm_state::uninitialized_position;
		if ((!is_uninitialized) && (frames_loaded != frames_expected))
		{
			print_slm_info();
			qli_runtime_error("Should have the same patterns loaded");
		}
	}
#endif
}

bool slm_holder::has_retarder() const
{
	static auto has_it = [&]
	{
		for (auto& slm : slms)
		{
			if (slm->is_retarder)
			{
				return true;
			}
		}
		return false;
	}();
	return has_it;
}

void slm_holder::set_slm_frame_await(const int frame_number, const std::chrono::microseconds & slm_delay_ms, const bool wait_on)
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	for (auto& slm : slms)
	{
		slm->set_frame_await(frame_number, slm_delay_ms, wait_on);
	}
}

bool slm_holder::set_slm_frame(const int frame_number)
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	slm_consistency_check();
	auto all_successful = true;
	for (auto& slm : slms)
	{
		const auto success = slm->set_frame(frame_number);
		if (success)
		{
			all_successful = false;
		}
	}
	slm_consistency_check();
	return all_successful;
}

void slm_holder::set_slm_mode(const slm_trigger_mode mode)
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	for (auto& slm : slms)
	{
		slm->toggle_mode(mode);
	}
	slm_consistency_check();
}

int slm_holder::get_slm_frames() const
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	slm_consistency_check();
#if _DEBUG
	for (auto slm_idx = 1; slm_idx < slms.size(); ++slm_idx)
	{
		if (slms.at(slm_idx)->get_frame_number_total() != slms.front()->get_frame_number_total())
		{
			qli_runtime_error("Should have the same patterns loaded");
		}
	}
#endif
	return slms.front()->get_frame_number_total();
}

int slm_holder::get_slm_frame_idx() const
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	slm_consistency_check();
	const auto frames_expected = slms.front()->get_frame_number();
	return frames_expected;
}

int slm_holder::get_slm_count() const
{
	return slms.size();
}

slm_dimensions slm_holder::get_slm_dimensions() const
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	slm_consistency_check();
	slm_dimensions items;
	for (const auto& slm : slms)
	{
		const auto frame = static_cast<frame_size>(*slm);
		items.push_back(frame);
	}
	return items;
}

slm_frame_pointers slm_holder::get_slm_frames(const int frame_idx) const
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	slm_frame_pointers pointers;
	for (const auto& slm : slms)
	{
		const auto* const frame = slm->get_frame(frame_idx);
		const slm_frame_pointer pointer(frame, *slm);
		pointers.push_back(pointer);
	}
	return pointers;
}

void slm_holder::toggle_slm_mode(const slm_trigger_mode mode)
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	for (auto& slm : slms)
	{
		slm->toggle_mode(mode);
	}
	slm_consistency_check();
}

void slm_holder::trigger_slm_hardware_sequence(const size_t capture_items, const channel_settings & channel_settings)
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	slm_consistency_check();
	if (slms.size() > 1)
	{
		qli_runtime_error("Functionality not supported");
	}
	slms.front()->hardware_trigger_sequence(capture_items, channel_settings);
}

std::chrono::microseconds slm_holder::max_vendor_stability_time() const
{
	static auto internal = [&] {
		auto max_time = slms.front()->vendor_stability_time();
		for (const auto& slm : slms)
		{
			const auto stability_time = slm->vendor_stability_time();;
			max_time = std::max(max_time, stability_time);
		}
		return max_time;
	}();
	return internal;
}

void slm_holder::reload_settings()
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	for (auto slm_idx = 0; slm_idx < slms.size(); ++slm_idx)
	{
		auto& slm = slms.at(slm_idx);
		auto settings = slm->get_modulator_state();
		slm->load_patterns(slm_idx, settings);
	}
}

slm_states slm_holder::get_modulator_states() const
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	slm_consistency_check();
	slm_states states;
	for (const auto& slm : slms)
	{
		states.push_back(slm->get_modulator_state());
	}
	return states;
}

void slm_holder::load_modulator_states(const slm_states & slm_states)
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	slm_consistency_check();
	for (auto slm_idx = 0; slm_idx < slm_states.size(); ++slm_idx)
	{
		const auto& state = slm_states.at(slm_idx);
		slms.at(slm_idx)->load_modulator_state(state);
	}
	slm_consistency_check();
}

void slm_holder::load_slm_settings(const fixed_modulator_settings & settings, const bool bake_all_patterns)
{
	std::unique_lock<std::mutex> lk(slm_consistency);
	slm_consistency_check();
	if (slms.size() != settings.size())
	{
		qli_invalid_arguments();
	}
	for (auto setting_idx = 0; setting_idx < settings.size(); ++setting_idx)
	{
		auto& slm = slms.at(setting_idx);
		const auto& item = settings.at(setting_idx);
		slm->load_patterns(setting_idx, item);
		if (bake_all_patterns)
		{
			const auto pattern_count = slm->get_frame_number();
			for (auto pattern_idx = 0; pattern_idx < pattern_count; ++pattern_idx)
			{
				slm->load_pattern(pattern_idx);
			}
		}
	}
	slm_consistency_check();
}


