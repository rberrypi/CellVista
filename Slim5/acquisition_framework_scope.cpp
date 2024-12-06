#include "stdafx.h"
#include "acquisition_framework.h"
#include "device_factory.h"
#include "scope.h"
#include <iostream>
#include "time_slice.h"
#include <future>

void acquisition_framework::long_timeout(const capture_item& current_position, const microscope_move_action& next_position, const std::string& message, const std::function<void()>& sync_io)
{
	const auto actually_moved = current_position != next_position;
	const auto move_delay = actually_moved ? next_position.stage_move_delay : ms_to_chrono(0);
	const auto roi_delay_actual = std::max(current_position.roi_move_delay, move_delay);
	const auto wait_for_sync = sync_io != nullptr ? std::async(std::launch::async, sync_io) : std::future<void>();
	if (roi_delay_actual.count())
	{
		std::lock_guard<std::mutex> lk(console_output_mutex_);
		std::cout << "Pausing acquisition " << message << std::endl;
		if (roi_delay_actual < microscope::shutter_time)
		{
			//can also move xyz here?
			windows_sleep(roi_delay_actual);
		}
		else
		{
			const auto start = timestamp();
			{
				auto next_off = next_position;
				next_off.scope_channel = 0;
				D->scope->move_to(next_off, false);
				//note the next move_to will ignore the stage position, and thus hopefully don't have an extra timeout!
			}
			const auto shutter_time = timestamp() - start;
			//Assume opening shutter takes same amount of time
			const auto left = roi_delay_actual - 2 * shutter_time;
			if (left.count() > 0)
			{
				// This means you shouldn't be closing and opening the shutter!!!
				{
					const auto later = std::chrono::system_clock::now() + left;
					const auto now_c = std::chrono::system_clock::to_time_t(later);
					std::cout << "Sleeping for ";
					display_time(std::cout, left);
					// ReSharper disable once CppDeprecatedEntity
					std::cout << " back on " << std::put_time(std::localtime(&now_c), "%c %Z") << std::endl;

				}
				const auto interrupt = ms_to_chrono(100);
				const auto steps = left.count() / interrupt.count();//some rounding errors here, maybe, oh well
				const auto remaining = left - steps * interrupt;
				windows_sleep(remaining);
				const auto stop_time = start + current_position.roi_move_delay;
				for (auto i = 0; i < steps; i++)
				{
					if (async_trigger_has_stop_signal())//also applies to the sync trigger?
					{
						return;
					}
					if (timestamp() >= stop_time)
					{
						break;
					}
					windows_sleep(interrupt);
				}
			}
			D->scope->move_light_path(next_position, false);//On some Zeiss systems this takes over 4000 ms!
		}
	}
	if (wait_for_sync.valid())
	{
		wait_for_sync.wait();
	}
}
