#include "stdafx.h"
#include "live_capture_engine.h"
#include <program_config.h>
#include <future>
#include <timeapi.h>

#include "compute_engine.h"
#include "device_factory.h"
#include "camera_device.h"
#include "render_widget.h"
#include "scope.h"
#include "time_guarantee.h"
#include "time_slice.h"
#include "thread_priority.h"

live_capture_engine::live_capture_engine(render_widget* render_surface, const std::shared_ptr<compute_engine>& compute_engine, QObject* parent) : QObject(parent), acquisition_framework(compute_engine), show_mode(live_compute_options::background_show_mode::regular), old_show_mode(live_compute_options::background_show_mode::regular), render_surface_(render_surface), capture_thread_status(thread_state::ready), is_acquiring(false)
{
	const dpm_bg_update_functor dpm_functor;
	const slim_update_functor slim_functor = [&](const slim_bg_settings& slim_background, int)
	{
		static_cast<slim_bg_settings&>(settings) = slim_background;
	};
	const phase_update_functor phase_functor = [&](const camera_frame_internal& new_background)
	{
		settings.load_background(new_background, true);
		{
			std::unique_lock<std::mutex> bg_lk(background_taking_m);
			show_mode = old_show_mode;
		}
		background_taking_cv.notify_one();
	};
	const background_update_functors functors = {
	dpm_functor,slim_functor,phase_functor
	};
	compute_engine->set_background_update_functors(functors);
}

camera_device* live_capture_engine::current_camera()
{
	return D->cameras.at(settings.camera_idx);
}

void live_capture_engine::stop_camera_and_trigger_thread()
{
	auto* camera = current_camera();
	trigger_thread_status = thread_state::terminate;
	trigger_thread_.join();
	camera->stop_software_capture();
}

void live_capture_engine::capture_thread_start()
{
	capture_thread_ = std::thread(&live_capture_engine::capture_thread, this);
	std::unique_lock<std::mutex> lk(capture_thread_start_m_);
	capture_thread_start_cv_.wait(lk, [&]
		{
			return capture_thread_status == thread_state::running;
		});
	std::cout << "Capture Thread Started" << std::endl;
}

void live_capture_engine::capture_thread_stop()
{
	compute->kill_queues();//hack hack hack
	capture_thread_status = thread_state::terminate;
	if (capture_thread_.joinable())
	{
		capture_thread_.join();
	}
}

void live_capture_engine::take_background()
{
	{
		std::unique_lock<std::mutex> lk(background_taking_m);
		old_show_mode = show_mode;
		show_mode = live_compute_options::background_show_mode::set_bg;
		//minor problem if you kill the thread while taking the background :-/
		const auto magic_deadlock_prevention = ms_to_chrono(2000);
		const auto success = background_taking_cv.wait_for(lk, magic_deadlock_prevention, [&]
			{
				return show_mode != live_compute_options::background_show_mode::set_bg;
			});
		std::cout << "Background Loaded" << std::endl;
	}
	emit background_enabled(true);
}

void live_capture_engine::set_compute_options(const live_compute_options& options)
{
	std::unique_lock<std::mutex> lk(background_taking_m);
	this->show_mode = options.show_mode;
}

void live_capture_engine::clear_background()
{
	const auto settings_changed_functor = [&]
	{
		settings.clear_background();
		show_mode = live_compute_options::background_show_mode::regular;
		background_enabled(false);
	};
	exclusive_access_to_settings(settings_changed_functor);
}

void live_capture_engine::exclusive_access_to_settings(const std::function<void()>& operation)
{
	std::unique_lock lk(exclusive_access_mutex);
	capture_thread_status = thread_state::pause;
	trigger_thread_status = thread_state::pause;
	{
		std::unique_lock lk2(trigger_thread_quanta);
		auto* camera = current_camera();
		camera->trigger_release_capture();
		std::unique_lock lk3(capture_thread_quanta);
		//
		operation();
		//sometimes we own the thread quanta anyways, so the we didn't actually need to push the shutdown trigger.
		camera->undo_release_capture_trigger();
		//
		trigger_thread_status = thread_state::running;
		capture_thread_status = thread_state::running;
	}
	capture_thread_quanta_cv.notify_one();
	trigger_thread_quanta_cv.notify_one();
}

void live_capture_engine::set_channel_settings(const channel_settings& new_settings)
{
	//stop acquisition + change settings
	const auto settings_changed_functor = [&]
	{
		const auto requires_a_reload = settings.difference_requires_camera_reload(new_settings);
		const auto clears_background = settings.difference_clears_background(new_settings);
		const auto old_background = clears_background ? compute_and_scope_settings::background_frame_ptr() : settings.background_;
		if (requires_a_reload)
		{
			//restart live capture (?)
			current_camera()->stop_software_capture();
			settings = new_settings;
			current_camera()->apply_settings(settings);
			current_camera()->start_software_capture();
		}
		else
		{
			settings = new_settings;
		}
		settings.background_ = old_background;
		emit background_enabled(settings.has_valid_background());
	};
	exclusive_access_to_settings(settings_changed_functor);
}

void live_capture_engine::live_trigger_part(camera_device* camera, const channel_settings& contrast, const cycle_position& position)
{
	//todo merge this function
	const auto phase_shift_exposure_and_delay = contrast.exposures_and_delays.at(position.pattern_idx);
	constexpr auto dont_refresh = false;
	const auto location = D->scope->get_state(dont_refresh);
	const auto zero_move_delay = ms_to_chrono(0);
	const auto move_action = microscope_move_action(location, zero_move_delay);//don't refresh, this costs 30ms on some devices, like the Nikon microscope
	constexpr auto channel_index = 1;
	const auto meta_data = frame_meta_data_before_acquire(contrast, position, phase_shift_exposure_and_delay, contrast, move_action, contrast.processing, channel_index, scope_action::capture);
	constexpr auto producer_consumer_mismatch = 5;// one second of stalling out ?
	//In live mode its "okay" to set an invalid pattern
	D->set_slm_frame(position.pattern_idx);
	if (!contrast.is_raw_frame())
	{
		windows_sleep(phase_shift_exposure_and_delay.slm_stability);
	}
	const auto checked = camera->trigger(meta_data, producer_consumer_mismatch);
	//Okay problem here, so next frame can be switched, so we need at least exposure time + readout time delay here, is that happening?
	if (checked > 2)
	{
		// std::cout << "Throughput exceeded, slow down the acquisition by adding more SLM stability time " << checked << std::endl;
	}
}

template<typename T>
bool windows_precise_timer_crap(T time_to_sleep)
{
	//https://stackoverflow.com/questions/13397571/precise-thread-sleep-needed-max-1ms-error

	auto as_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_to_sleep);
	const auto ns = as_ns.count();
		/* Declarations */
		HANDLE timer;   /* Timer handle */
		LARGE_INTEGER li;   /* Time defintion */
		/* Create timer */
		if (!(timer = CreateWaitableTimer(NULL, TRUE, NULL)))
			return FALSE;
		/* Set timer properties */
		li.QuadPart = -ns;
		if (!SetWaitableTimer(timer, &li, 0, NULL, NULL, FALSE)) {
			CloseHandle(timer);
			return FALSE;
		}
		/* Start & wait for timer */
		WaitForSingleObject(timer, INFINITE);
		/* Clean resources */
		CloseHandle(timer);
		/* Slept without problems */
		return TRUE;
}

void live_capture_engine::live_trigger_thread()
{
	//https://developercommunityapi.westus.cloudapp.azure.com/content/problem/1093078/timebeginperiod-function-dont-change-anymore-the-r.html
	//Except it does?
	timeBeginPeriod(1);
	//
	{
		{
			std::unique_lock<std::mutex> lk(trigger_thread_start_m_);
			trigger_thread_status = thread_state::running;
		}
		trigger_thread_start_cv_.notify_one();
	}
	try
	{
		auto position = cycle_position::start_position();
		while (true)
		{
			{
				std::unique_lock lk(trigger_thread_quanta);
				//time_slice ts("Triggering");
				trigger_thread_quanta_cv.wait(lk, [&] {
					return trigger_thread_status != thread_state::pause;
					});
				if (trigger_thread_status == thread_state::terminate)
				{
					goto escape;
				}
				const auto cycle_limit = settings.iterator().cycle_limit;
				position.advance(cycle_limit);
				auto* camera = current_camera();
				const auto pattern_index = settings.slm_pattern_for_live_mode(position.pattern_idx);
				const auto& slm_pattern = settings.exposures_and_delays.at(pattern_index);
				const auto transfer_time = camera->get_transfer_time();
				const auto single_software_trigger_time = camera->get_readout_time() + slm_pattern.exposure_time + ms_to_chrono(1);
				const auto minimal_frame_time = std::max(transfer_time, single_software_trigger_time);
				time_guarantee tg(minimal_frame_time);
				//
				const cycle_position cycle_position(position.pattern_idx, pattern_index);
				{
					// time_slice ts("Trigger Part");
					live_trigger_part(camera, settings, cycle_position);//sleeps SLM time
				}
				{
					// time_slice ts("Exposure Time");
					// windows_sleep(slm_pattern.exposure_time + ms_to_chrono(1));
					windows_sleep(single_software_trigger_time);
				}
			}
		}
	escape:
		std::cout << "Stopping Trigger Thread" << std::endl;
	}
	catch (...)
	{
	}
}

void live_capture_engine::start_camera_and_trigger_thread()
{
	auto* camera = current_camera();
	trigger_thread_status = thread_state::ready;
	camera->apply_settings(settings);
	camera->start_software_capture();
	trigger_thread_ = std::thread(&live_capture_engine::live_trigger_thread, this);
	std::unique_lock<std::mutex> lk(trigger_thread_start_m_);
	trigger_thread_start_cv_.wait(lk, [&]
		{
			return trigger_thread_status == thread_state::running;
		});
	std::cout << "Trigger Thread Started" << std::endl;
}

void live_capture_engine::capture_thread()
{
	{
		{
			std::unique_lock<std::mutex> lk(capture_thread_start_m_);
			capture_thread_status = thread_state::running;
		}
		capture_thread_start_cv_.notify_one();
	}
	try
	{
		start_camera_and_trigger_thread();
		auto position = cycle_position::start_position();
		while (true)
		{

			std::unique_lock lk(capture_thread_quanta);
			//time_slice ts("Capturing");
			capture_thread_quanta_cv.wait(lk, [&] {
				return capture_thread_status != thread_state::pause;
				});
			if (capture_thread_status == thread_state::terminate)
			{
				goto escape;
			}
			const auto cycle_limit = settings.iterator().cycle_limit;
			position.advance(cycle_limit);

			auto* camera = current_camera();
			const camera_device::camera_frame_processing_function process_frame = [&](const camera_frame<unsigned short>& input)
			{
				//time_slice ts2("Compute");
				const auto compute_options = live_compute_options(true, show_mode);
				const auto captured_frames = compute->push_work(input, settings, compute_options);
				for (auto frame = 0; frame < captured_frames; ++frame)
				{
					const compute_engine::work_function render_me = [&](const camera_frame<float>& input_d)
					{
						const auto* dpm_ptr = [&] {
							const auto show_dpm = settings.do_ft && settings.processing == phase_processing::raw_frames && settings.retrieval == phase_retrieval::diffraction_phase;
							return show_dpm ? &settings : nullptr;
						}();
						const auto msg = pop_live_message();
						render_surface_->paint_surface(true, input_d, msg, dpm_ptr);

						
						auto current_z = D->scope->z_drive->get_position_z(true);
						LOGGER_INFO("Current Z: " << current_z);
						//D->scope->z_drive->move_to_z(current_z - 0.1);
						
					};
					compute->get_work_gpu(render_me);
				}
			};
			const auto capture_body = [&] {
				const auto no_microscope_move = ms_to_chrono(0);
				const auto status = camera->capture(process_frame, no_microscope_move);
				if (status != camera_device::capture_result::good)
				{
					auto what = 0;
				}
				if (status == camera_device::capture_result::failure)
				{
					//need to get exclusive access to the settings
					capture_thread_quanta.unlock();
					fix_capture();
					capture_thread_quanta.lock();
				}
			};
			capture_body();

		}
	escape:
		stop_camera_and_trigger_thread();
	}
	catch (const thrust::system::system_error& some_error)
	{
		qli_runtime_error(some_error.what());
	}
	catch (const std::exception& some_error)
	{
		qli_runtime_error(some_error.what());
	}
	catch (...)
	{
		//What?
		std::cout << "Camera shut down failure?" << std::endl;
	}
}

void live_capture_engine::dirty_acquisition_wrapper(const capture_mode capture_mode)
{
	const std::function<void()> function = [&]
	{
		current_camera()->stop_software_capture();
		{
			try
			{
				capture_wrapper(capture_mode, render_surface_);
			}
			catch (const std::exception& exception)
			{
				qli_runtime_error(exception.what());
			}
			catch (...)
			{
				qli_runtime_error();
			}
		}
		current_camera()->apply_settings(settings);
		current_camera()->start_software_capture();
	};
	emit gui_enable(false);
	exclusive_access_to_settings(function);
	emit gui_enable(true);
	play_done_sound();
	is_acquiring = false;
}


void live_capture_engine::start_acquisition(capture_mode capture_mode)
{
	if (is_acquiring)
	{
		return;
	}
	is_acquiring = true;
	std::thread t(&live_capture_engine::dirty_acquisition_wrapper, this, capture_mode);
	t.detach();
}

void live_capture_engine::stop_acquisition() noexcept
{
	abort_capture = true;
}

void live_capture_engine::terminate_live_capture()
{
	capture_thread_stop();
}

void live_capture_engine::begin_live_capture(const channel_settings& channel_settings, const live_compute_options& live_compute_options)
{
	channel_settings.assert_validity();
	this->settings = channel_settings;
	this->show_mode = live_compute_options.show_mode;
	capture_thread_start();
}

void live_capture_engine::fix_capture()
{
	const std::function<void()> function = [&]
	{
		current_camera()->stop_software_capture();
		current_camera()->fix_camera();
		current_camera()->start_software_capture();
	};
	exclusive_access_to_settings(function);
}

