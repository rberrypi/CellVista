#include "stdafx.h"
#include <fstream>
#include "slim_four.h"
#include "save_device_state.h"
#include "io_worker.h"
#include "start_stop.h"
#include "device_factory.h"
#include "time_guarantee.h"
#include "compute_engine.h"
#include "camera_device.h"
#include "scope.h"
#include "slm_device.h"
#include <iostream>

acquisition_framework::acquisition_framework(const std::shared_ptr<compute_engine>& compute_engine) : async_trigger_kill_(false), async_trigger_thread_started_(false), abort_capture(false), compute(compute_engine)
{
}

acquisition_framework::~acquisition_framework()
{
	compute->kill_queues();
}

void acquisition_framework::async_trigger_start(size_t start_event_idx, const cycle_position& start_position, const std::function<void()>& sync_io)
{
	async_trigger_kill_ = false;
	async_trigger_thread_started_ = false;
	async_trigger_thread_ = std::thread(&acquisition_framework::async_trigger_thread, this, start_event_idx, start_position, sync_io);
	std::unique_lock<std::mutex> lk(async_trigger_started_m_);
	async_trigger_started_cv_.wait(lk, [&]
	{
		return async_trigger_thread_started_ || async_trigger_has_stop_signal();
	});
}

void acquisition_framework::async_trigger_stop()
{
	async_trigger_kill_ = true;
	if (async_trigger_thread_.joinable())
	{
		async_trigger_thread_.join();
	}
}

bool acquisition_framework::async_trigger_has_stop_signal() const noexcept
{
	return async_trigger_kill_ || abort_capture;
}

void acquisition_framework::trigger_part(const frame_meta_data_before_acquire& meta_data, const channel_settings& channel_setting, bool is_async)
{
	channel_setting.assert_validity();
	time_guarantee cycle_time(is_async ? meta_data.duration() : ms_to_chrono(0));
	auto& camera = D->cameras.at(channel_setting.camera_idx);
	const auto pattern = meta_data.pattern_idx;
	////////Set SLM & Move
	{
		const auto do_move = meta_data.is_first();
		const auto slm_stability_time = meta_data.slm_stability;
		if (do_move)
		{
			D->load_slm_settings(channel_setting.modulator_settings, true);
		}
		is_async ? D->set_slm_frame(pattern) : D->set_slm_frame_await(pattern, slm_stability_time, true);
		time_guarantee slm_t(is_async ? slm_stability_time : ms_to_chrono(0));
		if (do_move)
		{
			D->scope->move_to(meta_data, false);
			camera->apply_settings(channel_setting);
		}
	}
	////////Trigger	
	camera->trigger(meta_data);
}

void acquisition_framework::roi_delay_part(const capture_item& current_position, const acquisition& route, const size_t event_idx, const std::function<void()>& sync_io)
{
	////////ROI Delay
	const auto next_index = event_idx + 1;
	if (next_index < route.number_of_events())
	{
		const auto action = route.get_microscope_move_action(next_index);
		long_timeout(current_position, action, "ROI Delay", sync_io);
	}
}

acquisition_meta_data acquisition_framework::capture(bool async_capture, bool async_io, bool write_to_console, const int channel_saved, std::fstream& log, render_engine* engine)
{
	//////Setup
	async_trigger_kill_ = false;//dirty hack, sorry Bjarne
	const auto start_cameras = [&] {for (auto&& camera : D->cameras)
	{
		camera->start_software_capture();
	}};
	const auto stop_cameras = [&] {for (auto&& camera : D->cameras)
	{
		camera->stop_software_capture();
	}};
	start_stop start_stop(start_cameras, stop_cameras);
	std::vector<auto_focus_info> autofocus_pairs;
	auto& route = D->route;
	io_worker io_worker(compute, engine);
	QObject::connect(&io_worker, &io_worker::set_io_progress, [&](const size_t left) {set_io_progress(left); });
	QObject::connect(&io_worker, &io_worker::set_io_buffer_progress, [&](const size_t left) {set_io_buffer_progress(left); });
	const auto io_synchronization_functor = async_io ? [&]
	{
		io_worker.flush_io_queue(true);
	} : std::function<void()>();
	//
	acquisition_meta_data meta(timestamp());
	if (async_capture)
	{
		async_trigger_start(0, cycle_position(), io_synchronization_functor);
	}
	//Apply all camera settings
	auto progress_id = 0;
	const auto number_of_events = route.number_of_events();
	for (size_t event_idx = 0; event_idx < number_of_events; ++event_idx)
	{
		const auto& scope_event = route.cap.at(event_idx);
		const auto& channel = route.ch.at(scope_event.channel_route_index);
		const auto move_action = acquisition::get_microscope_move_action(channel, scope_event);
		const auto cycle_settings = channel.iterator();
		auto position = cycle_position();
		auto& camera = D->cameras.at(channel.camera_idx);
		{
			for (; position.denoise_idx < cycle_settings.cycle_limit.denoise_idx; ++position.denoise_idx)
			{
				for (position.pattern_idx = 0; position.pattern_idx < cycle_settings.cycle_limit.pattern_idx; )
				{
					//time_slice ts("Capture");
					if (abort_capture)
					{
						goto escape;
					}
					const compute_engine::work_function process_af = [&](const camera_frame<float>& img)
					{
						const auto value = compute->compute_fusion_focus(img);
						const auto_focus_info info(scope_event.z, value);
						autofocus_pairs.push_back(info);
						const auto af_transition = route.is_af_transition(event_idx);
						if (af_transition)
						{
							const auto best_focus = process_focus_list(autofocus_pairs);
							autofocus_pairs.resize(0);
							route.cap.at(event_idx + 1).z = best_focus;
							trigger_waits_for_.event_happened(event_idx);
						}
					};
					const camera_device::camera_frame_processing_function process_frame = [&](const camera_frame<unsigned short>& frame)
					{
						/////In Sync Mode, Modulate Next Pattern?
						if (!async_capture)
						{
							//wow this is some wizard level stuff, wtf did I write?
							const auto next_pattern_and_stability = route.get_next_pattern_and_stability(event_idx, frame.pattern_idx);
							if (next_pattern_and_stability.first >= 0)
							{
								D->set_slm_frame_await(next_pattern_and_stability.first, next_pattern_and_stability.second, false);
							}
						}
						//////Process Autofocus
						if (scope_event.action == scope_action::focus)
						{
							const live_compute_options no_updates;
							const auto made_work = compute->push_work(frame, channel, no_updates);
							if (made_work)
							{
								compute->get_work_gpu(process_af);
							}
						}
						//////Save Files
						else
						{
							const auto force_sixteen = channel.is_native_sixteen_bit();
							const raw_io_work_meta_data meta_data(scope_event, progress_id, { cycle_settings.retrieval, cycle_settings.processing }, force_sixteen, channel.label_suffix, route.filename_grouping);
							const raw_io_work<unsigned short> work(frame, meta_data, gui_message_kind::none);
							write_capture_log_line_header(log, work, scope_event.roi_move_delay);
							if (write_to_console)
							{
								std::lock_guard<std::mutex> lk(console_output_mutex_);
								write_capture_log_line_header(std::cout, work, scope_event.roi_move_delay);
							}
							if (async_io)
							{
								while (!io_worker.push_work_deep_copy(work))
								{
									std::cout << "IO buffer overflowing, will try to push work, again in 100 ms, if this message persists you will need to slow down the acquisition" << std::endl;
									const auto buffer_timeout = ms_to_chrono(100);
									windows_sleep(buffer_timeout);// and stall forever
									//LOL does this actually work?
								}
							}
							else
							{
								io_worker::do_work(compute, channel, engine, work, D->route.output_dir);
							}
						}
					};
					////////Wait for Settings Change
					if (async_capture)
					{
						const auto settings_changed = route.settings_have_changed_for_this_event(event_idx);
						if (settings_changed)
						{
							readout_waits_for_.wait_for(event_idx);
						}
					}
					if (!async_capture)
					{
						const frame_meta_data_before_acquire meta_data(channel, position, channel.exposures_and_delays.at(position.pattern_idx), channel, move_action, channel.processing, scope_event.channel_route_index, scope_event.action);
						trigger_part(meta_data, channel, false);
					}
					////////Capture
					const auto time_hint = 4*camera->get_transfer_time() + ms_to_chrono(2);//maybe need to revise this function. (large stage translations will fail?), seriously wtf is up with this command?
					const auto status = camera->capture(process_frame, time_hint);
					if (status == camera_device::capture_result::good)
					{
						progress_id = progress_id + 1;
						position.pattern_idx = position.pattern_idx + 1;
						set_capture_progress(progress_id);
					}
					else
					{
						std::cout << "Acquisition error in event #" << event_idx << " at [" << position.pattern_idx << "," << position.denoise_idx << "]" << ", if this is a reoccurring error you need to slow down the acquisition" << std::endl;
						meta.register_failed_frame(event_idx);
						async_trigger_stop();
						camera->stop_software_capture();
						camera->fix_camera();
						camera->start_software_capture();
						async_trigger_start(event_idx, position, io_synchronization_functor);
					}
				}
			}
			if (!async_capture)
			{
				roi_delay_part(scope_event, route, event_idx, io_synchronization_functor);
			}
		}
	}
escape:
	meta.stop = timestamp();
	std::cout <<"Writing Buffers" << std::endl;
	//keep the light off as we're waiting for the file to be written to the disk
	D->scope->chan_drive->move_to_channel(channel_saved);
	async_trigger_stop();
	return meta;
}

void wait_for_event::wait_for(const size_t point)
{
	std::unique_lock<std::mutex> lk(async_barrier_cleared_m_);
	const auto predicate = [&]
	{
		return point == async_barrier_wait_for_;
	};
	async_barrier_cleared_cv_.wait(lk, predicate);
}

void wait_for_event::event_happened(const size_t point)
{
	{
		std::unique_lock<std::mutex> lk(async_barrier_cleared_m_);
		async_barrier_wait_for_ = point;
	}
	async_barrier_cleared_cv_.notify_one();
}
