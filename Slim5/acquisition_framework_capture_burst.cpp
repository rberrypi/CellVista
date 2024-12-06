#include "stdafx.h"
#include "acquisition_framework.h"
#include <fstream>
#include <iostream>
#include "qli_runtime_error.h"
#include "time_slice.h"
#include "device_factory.h"
#include "io_worker.h"
#include "break_into_ranges.h"
#include "time_guarantee.h"
#include "camera_device.h"
#include "scope.h"

acquisition_meta_data acquisition_framework::capture_burst_mode(const int channel_saved, std::fstream& log)
{
	time_slice t("Burst acquisition Took:");
	acquisition_meta_data meta(timestamp());
	auto& route = D->route;
	const auto burst_valid = route.is_valid_for_burst();
	if (!burst_valid)
	{
		qli_runtime_error("Route not valid for burst acquisition!");
	}
	io_worker io_worker(compute, nullptr);
	QObject::connect(&io_worker, &io_worker::set_io_progress, [&](const size_t left) {set_io_progress(left); });
	QObject::connect(&io_worker, &io_worker::set_io_buffer_progress, [&](const size_t left) {set_io_buffer_progress(left); });
	const auto sync_io = std::function<void()>();
	//
	auto comparison = [](const capture_item& a, const capture_item& b)
	{
		return a.action == b.action&& a.channel_route_index == b.channel_route_index && static_cast<const scope_location_xyz&>(a) == b;
	};
	//this probably messed up because should also check the scope_delays
	auto ranges = break_into_ranges(route.cap, comparison);
	size_t total_progress = 0;
	for (auto& range : ranges)
	{
		auto first_event = *range.first;
		auto channel = route.ch.at(first_event.channel_route_index);
		auto& camera = D->cameras.at(channel.camera_idx);
		channel.mode = camera_mode::burst;
		auto proto_type_capture_item = range.first;
		auto last_item_ptr = std::prev(range.second);
		const auto move_action = microscope_move_action(microscope_state(first_event, channel), first_event.stage_move_delay);
		auto cycle_settings = channel.iterator();
		auto local_progress = 0;
		{
			for (auto d = 0; d < cycle_settings.cycle_limit.denoise_idx; ++d)
			{
				for (auto p = 0; p < cycle_settings.cycle_limit.pattern_idx;)
				{
					if (abort_capture)
					{
						io_worker.flush_io_queue(false);
						goto escape;
					}
					//////Move microscope + Set SLM
					{
						const auto slm_stability_time = channel.exposures_and_delays.at(p).slm_stability;
						time_guarantee slm_t(slm_stability_time);
						D->set_slm_frame(p);
						camera->apply_settings(channel);
						if (p == 0)
						{
							D->scope->move_to(move_action, false);
						}
					}
					//////Process Frame
					const camera_device::camera_frame_processing_function process_frame = [&](const camera_frame<unsigned short> frame)
					{
						if (!abort_capture)
						{
							if (first_event.action == scope_action::capture)
							{
								const auto frames_per_capture_item = cycle_settings.frame_count() - 1;
								auto is_last =  d == cycle_settings.cycle_limit.denoise_idx - 1 && p == cycle_settings.cycle_limit.pattern_idx - 1;
								is_last = local_progress == frames_per_capture_item && is_last;
								//auto meta_data_ptr = is_last ? last_item_ptr : proto_type_capture_item;
								const auto meta_data_ptr = proto_type_capture_item + local_progress;
								auto what = *meta_data_ptr;
								const auto roi_delay = is_last ? last_item_ptr->roi_move_delay : ms_to_chrono(0);
								const auto force_sixteen = channel.is_native_sixteen_bit();
								const raw_io_work_meta_data io_meta_data(*meta_data_ptr, total_progress + local_progress, { phase_retrieval::camera, phase_processing::raw_frames }, force_sixteen,  channel.label_suffix,route.filename_grouping);
								const raw_io_work<unsigned short> work(frame, io_meta_data, gui_message_kind::none);

								write_capture_log_line_header(log, work, roi_delay);
								while (!io_worker.push_work_deep_copy(work))
								{
									std::cout << "IO buffer overflowing retrying in 100 ms" << std::endl;
									windows_sleep(ms_to_chrono(100));
								}
								local_progress = local_progress + 1;
							}
							else
							{
								qli_runtime_error("Not implemented");
							}
						}
					};
					{
						//////Start Capture
						const auto blank_render_settings = render_settings();
						const cycle_position position = { d,p };
						const auto& pattern = channel.exposures_and_delays.at(position.pattern_idx);
						const auto time_hint = 4*pattern.duration();//maybe need to revise this function. (large stage translations will fail?), seriously wtf is up with this command?
						const frame_meta_data_before_acquire meta_data(channel, position, pattern, channel, move_action, channel.processing, first_event.channel_route_index, first_event.action);
						const auto valid = camera->capture_burst(range, meta_data, time_hint, process_frame);//capture one time should be better set
						if (!valid)
						{
							std::cout << "Acquisition Error" << std::endl;
							camera->fix_camera();
						}
						else
						{
							p = p + 1;
							total_progress = total_progress + local_progress;//this lets us "resume";
						}
					}
				}
			}
		}
		//Timeout
		if (range.second < route.cap.end())
		{
			const auto next_event = *range.second;
			auto& next_channel = route.ch.at(next_event.channel_route_index);
			const auto next_microscope_state = microscope_state(next_event, next_channel);
			const auto next_microscope_move_action = microscope_move_action(next_microscope_state, next_event.stage_move_delay);
			long_timeout(*proto_type_capture_item, next_microscope_move_action, " ROI Delay", sync_io);
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
