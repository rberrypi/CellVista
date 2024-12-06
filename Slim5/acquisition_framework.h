#pragma once
#ifndef ACQUISITION_FRAMEWORK_H
#define ACQUISITION_FRAMEWORK_H
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include "capture_modes.h"
#include "acquisition.h"
#include <boost/noncopyable.hpp>
class render_engine;
struct cycle_position;
struct camera_config;
struct frame_meta_data_before_acquire;

struct acquisition_meta_data final
{
	std::chrono::microseconds cycle_time, start, stop;
	std::unordered_map<size_t, size_t> failed_frames;
	explicit acquisition_meta_data(const std::chrono::microseconds& start) noexcept: cycle_time(0), start(start), stop(0) {}
	acquisition_meta_data() noexcept : acquisition_meta_data(std::chrono::microseconds(0)) {}
	void register_failed_frame(const size_t frame_idx)
	{
		failed_frames[frame_idx]++;
	}

	[[nodiscard]] std::chrono::microseconds duration() const noexcept
	{
		return stop - start;
	}
};

struct auto_focus_info final
{
	float z, metric;
	auto_focus_info(const float z, const float metric) noexcept:z(z), metric(metric) {}
	auto_focus_info() noexcept :auto_focus_info(0, 0) {}
};

class compute_engine;

struct wait_for_event
{
private:
	size_t async_barrier_wait_for_;
	std::mutex async_barrier_cleared_m_;
	std::condition_variable async_barrier_cleared_cv_;
public:
	wait_for_event() noexcept: async_barrier_wait_for_(std::numeric_limits<size_t>::max()) {}
	//Trigger waits for this point
	void wait_for(size_t point);
	//Readout point
	void event_happened(size_t point);
	/* Example scenario
	 * push(5) <- waiting for readout to reach this point autofocus to be filled
	 * push(7) <- error?
	 *
	 */
};
class acquisition_framework : private boost::noncopyable
{
	std::mutex console_output_mutex_;
	wait_for_event trigger_waits_for_;
	wait_for_event readout_waits_for_;
	//
	void roi_delay_part(const capture_item& current_position, const acquisition& route, size_t event_idx, const std::function<void()>& sync_io);

	//todo arguments to this monster need to be a structure!!!
	static void trigger_part(const frame_meta_data_before_acquire& meta_data, const channel_settings& channel_setting, bool is_async);
	void async_trigger_thread(size_t start_event_idx, const cycle_position& start_position, const std::function<void()>& sync_io);
	void async_trigger_start(size_t start_event_idx, const cycle_position& start_position, const std::function<void()>& sync_io);
	void async_trigger_stop();
	[[nodiscard]] bool async_trigger_has_stop_signal() const noexcept;
	volatile bool async_trigger_kill_;
	std::thread async_trigger_thread_;
	bool async_trigger_thread_started_;
	std::condition_variable async_trigger_started_cv_;
	std::mutex async_trigger_started_m_;
	//
	//Utility 
	//static bool is_last_kind(std::vector<capture_item>& route, size_t idx);
	static float process_focus_list(const std::vector<auto_focus_info>& values);
	static float parabolic_peak(float i_a, float v_a, float i_b, float v_b, float i_c, float v_c);

	void long_timeout(const capture_item& current_position, const microscope_move_action& next_position, const std::string& message, const std::function<void()>& sync_io);

public:
	typedef std::function<bool(const capture_item&, const capture_item&)> transition_predicate;
	//acquisitionFramework()
	explicit acquisition_framework(const std::shared_ptr<compute_engine>& compute_engine);
	virtual ~acquisition_framework();
	std::atomic<bool> abort_capture;
	// ReSharper disable CppParameterNeverUsed
	virtual void set_capture_progress(size_t current)  {}
	virtual void set_capture_total(size_t total)  {}
	virtual void set_io_progress(size_t left)  {}
	virtual void set_io_progress_total(size_t left)  {}
	virtual void set_io_buffer_progress(size_t total)  {}
	// ReSharper restore CppParameterNeverUsed
	acquisition_meta_data capture_burst_mode(int channel_saved, std::fstream& log);
	acquisition_meta_data capture(bool async_capture, bool async_io, bool write_to_console, int channel_saved, std::fstream& log, render_engine* engine);
	std::shared_ptr<compute_engine> compute;
	acquisition_meta_data capture_wrapper(capture_mode capture_mode, render_engine* engine);
	static void merge_meta_data_cs_vs(const std::string& output_directory);
};
#endif