#include "stdafx.h"
#include "acquisition_framework.h"
#include "device_factory.h"
#include "thread_priority.h"
#include <iostream>
#include "frame_meta_data.h"

void acquisition_framework::async_trigger_thread(const size_t start_event_idx, const cycle_position& start_position, const std::function<void()>& sync_io)
{
	thread_priority tp;
	{
		std::lock_guard<std::mutex> lk(async_trigger_started_m_);
		async_trigger_thread_started_ = true;
	}
	async_trigger_started_cv_.notify_one();
	std::cout << "Starting triggering from frame #" << start_event_idx << " at [" << start_position.pattern_idx << "," << start_position.denoise_idx << "]" << std::endl;
	//
	auto& route = D->route;
	for (auto event_idx = start_event_idx; event_idx < route.number_of_events(); ++event_idx)
	{
		const auto& scope_event = route.cap.at(event_idx);
		const auto& channel = route.ch.at(scope_event.channel_route_index);
		const auto settings_changed = route.settings_have_changed_for_this_event(event_idx);
		const auto move_action = acquisition::get_microscope_move_action(channel, scope_event);
		const auto cycle_settings = channel.iterator();
		{
			auto position = event_idx == start_event_idx ? start_position : cycle_position();
			for (; position.denoise_idx < cycle_settings.cycle_limit.denoise_idx; ++position.denoise_idx)
			{
				for (; position.pattern_idx < cycle_settings.cycle_limit.pattern_idx; ++position.pattern_idx)
				{
					if (async_trigger_has_stop_signal())
					{
						return;
					}
					const auto phase_shift_exposure_and_delay = channel.exposures_and_delays.at(position.pattern_idx);
					const frame_meta_data_before_acquire frame_meta_data_before_acquire(channel, position, phase_shift_exposure_and_delay, channel, move_action, channel.processing, channel.scope_channel,scope_event.action);
					trigger_part(frame_meta_data_before_acquire, channel, true);
					if (settings_changed)
					{
						readout_waits_for_.event_happened(event_idx);
					}
				}
				position.pattern_idx = 0;
			}
			roi_delay_part(scope_event, route, event_idx, sync_io);
		}
		////////Wait for AF to catch up, because were going to nuke values
		const auto af_transition = route.is_af_transition(event_idx);
		if (af_transition)
		{
			trigger_waits_for_.wait_for(event_idx);
		}
	}

}