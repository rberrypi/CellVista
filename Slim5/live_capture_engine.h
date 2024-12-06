#pragma once
#ifndef LIVE_CAPTURE_ENGINE_H
#define LIVE_CAPTURE_ENGINE_H
#include <QObject>
#include <mutex>
#include "acquisition_framework.h"
#include "channel_settings.h"
#include "gui_message.h"
#include "capture_modes.h"
#include <condition_variable>
class compute_engine;
class camera_device;
class render_widget;

class live_capture_engine final : public QObject, public acquisition_framework, public gui_messages
{
	Q_OBJECT

	//
	std::mutex background_taking_m;
	std::condition_variable background_taking_cv;
	live_compute_options::background_show_mode show_mode,old_show_mode;
	//
	channel_settings settings;
	render_widget* render_surface_;
	enum class thread_state { ready, pause, running, terminate };
	//
	void capture_thread();
	void capture_thread_start();
	void capture_thread_stop();
	std::thread capture_thread_;	
	std::atomic<thread_state> capture_thread_status;
	std::condition_variable capture_thread_start_cv_;
	std::mutex capture_thread_start_m_;
	std::recursive_mutex capture_thread_quanta;
	std::condition_variable_any capture_thread_quanta_cv;
	//
	std::thread trigger_thread_;	
	std::atomic<thread_state> trigger_thread_status;
	std::condition_variable trigger_thread_start_cv_;
	std::mutex trigger_thread_start_m_;
	std::mutex trigger_thread_quanta;
	std::condition_variable trigger_thread_quanta_cv;
	
	void live_trigger_thread();
	void start_camera_and_trigger_thread();
	void stop_camera_and_trigger_thread();
	camera_device* current_camera();
	static void live_trigger_part(camera_device* camera, const channel_settings& contrast, const cycle_position& position);
	void exclusive_access_to_settings(const std::function<void()>& operation);
	std::mutex exclusive_access_mutex;
	static void play_done_sound();
	void dirty_acquisition_wrapper(capture_mode capture_mode);
	std::atomic<bool> is_acquiring;
public:
	explicit live_capture_engine(render_widget* render_surface, const std::shared_ptr<compute_engine>& compute_engine, QObject* parent);

public slots:
	void start_acquisition(capture_mode capture_mode);
	void stop_acquisition() noexcept;
	//
	void terminate_live_capture();
	void begin_live_capture(const channel_settings& channel_settings, const live_compute_options& live_compute_options);
	void set_channel_settings(const channel_settings& new_settings);
	void set_compute_options(const live_compute_options& options);
	void take_background();
	void clear_background();
	void fix_capture();
signals:
	void gui_enable(bool enable );
	void set_capture_progress(size_t current)  override;
	void set_capture_total(size_t total)  override;
	void set_io_progress(size_t left)  override;
	void set_io_progress_total(size_t total)  override;
	void set_io_buffer_progress(size_t total)  override;
	void background_enabled(bool enabled);
};

#endif