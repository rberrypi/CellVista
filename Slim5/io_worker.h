#pragma once
#ifndef IO_WORKER_H
#define IO_WORKER_H
#include <thread>
#include <queue>
#include "pre_allocated_pool.h"
#include <QObject>
#include "io_work.h"
struct channel_settings;
class compute_engine;
class render_engine;
struct gui_message;
class io_worker final : public QObject
{
	Q_OBJECT
		Q_DISABLE_COPY(io_worker)
		bool kill_thread_;

	static pre_allocated_pool pool_;
	//needs a GPU
	std::mutex io_queue_m_;
	std::condition_variable io_queue_cv_;
	std::queue<raw_io_work<unsigned short>> io_queue_;
	//
	bool clear_queue_;//should be converted to special work items?
	//
	bool kill_io_;
	void work_thread(std::shared_ptr<compute_engine>& cp, render_engine* render_engine);
	void kill_join();
	// recall C++ is constructed last, destroyed first
	// THIS IS IMPORTANT BECAUSE WE CAN'T HAVE THE CONDITION VARIABLES DIE ON YOU
	// (AKA MSVC DOESN'T INSERT THOSE INSTRUCTIONS BECAUSE ITS UB)
	std::condition_variable cv_;
	std::thread worker_;
	static void simulate_delay();

public:
	static void do_work(std::shared_ptr<compute_engine>& cp, const channel_settings& channel_settings, render_engine* render_engine, raw_io_work<unsigned short> io_work, const std::experimental::filesystem::path& out_dir);
	void flush_io_queue(bool wait);
	bool push_work_deep_copy(const raw_io_work<unsigned short>& work_unit);// is deep-copied
	virtual ~io_worker();
	explicit io_worker(std::shared_ptr<compute_engine>& cp, render_engine* render_engine);
signals:
	void set_io_progress(size_t index);//should be a queued connection!
	void set_io_buffer_progress(size_t index);//should be a queued connection!
};

#endif