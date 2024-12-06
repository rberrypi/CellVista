#include "stdafx.h"
#include "io_worker.h"
#include "time_slice.h"
#include "device_factory.h"
#include "pre_allocated_pool.h"
#include "compute_engine.h"
#include "render_engine.h"
#include "qli_runtime_error.h"
pre_allocated_pool io_worker::pool_;// should be default constructed, which will allocate the memory!

io_worker::io_worker(std::shared_ptr<compute_engine>& cp, render_engine* render_engine) : kill_thread_(false), clear_queue_(false), kill_io_(false), worker_(&io_worker::work_thread, this, std::ref(cp), render_engine) {}

void io_worker::simulate_delay()
{
	static auto yes_maybe_no = true;
	yes_maybe_no = !yes_maybe_no;
	const auto timeout = yes_maybe_no ? ms_to_chrono(500) : ms_to_chrono(10);
	windows_sleep(timeout);
}

bool io_worker::push_work_deep_copy(const raw_io_work<unsigned short>& work_unit)
{
	{
		std::unique_lock<std::mutex> lk(io_queue_m_);
		const auto bytes = sizeof(unsigned short) * work_unit.n() * work_unit.samples_per_pixel;
		const auto buffer = reinterpret_cast<unsigned short*>(pool_.get(bytes));
		if (buffer == nullptr)
		{
			return false;
		}
		std::memcpy(buffer, work_unit.img, bytes);
		auto push_work = work_unit;
		push_work.img = buffer;
		io_queue_.push(push_work);
	}
	io_queue_cv_.notify_all();
	return true;
}

void io_worker::flush_io_queue(const bool wait)
{
	clear_queue_ = !wait;
	std::unique_lock<std::mutex> lk(io_queue_m_);
	io_queue_cv_.wait(lk, [&] {return io_queue_.empty() || kill_thread_; });
}

void io_worker::work_thread(std::shared_ptr<compute_engine>& cp, render_engine* render_engine)
{
	time_slice t("File IO Took:");
	raw_io_work<unsigned short> work;
	while (true)
	{
		{
			std::unique_lock<std::mutex> lk(io_queue_m_);
			io_queue_cv_.wait(lk, [&] {return !io_queue_.empty() || clear_queue_ || kill_thread_; });
			if (clear_queue_)
			{
				pool_.reset_queue();
				clear_queue_ = false;
			}
			if (kill_thread_ && io_queue_.empty())
			{
				goto escape;
			}
			work = io_queue_.front();
			io_queue_.pop();
			set_io_progress(work.progress_id + 1);
			set_io_buffer_progress(io_queue_.size());
			if (io_queue_.empty())
			{
				lk.unlock();
				io_queue_cv_.notify_all();
			}
		}
		if (work.img != nullptr)
		{
			auto channel = D->route.ch.at(work.channel_route_index);
			do_work(cp, channel, render_engine, work, D->route.output_dir);
			pool_.put_back(sizeof(unsigned short) * work.n());
		}
	}
escape:
	std::cout << "Closed IO Thread" << std::endl;
}

void io_worker::kill_join()
{
	kill_thread_ = true;
	io_queue_cv_.notify_all();
	worker_.join();
}

void io_worker::do_work(std::shared_ptr<compute_engine>& cp, const channel_settings& channel_settings, render_engine* render_engine, raw_io_work<unsigned short> io_work, const std::experimental::filesystem::path& out_dir)
{
	//https://futurism.com/the-byte/oral-sex-toy-ai-deep-learning
	//WORK - ALSO, FUCK GABRIEL POPESCU, the guy is an embarassment as a human being
	const auto has_display = render_engine != nullptr;
	const auto has_direct = channel_settings.is_direct_write();
	const auto sets_bg = io_work.action == scope_action::set_bg_for_this_channel;
	const auto has_write = (!sets_bg);
	const auto background_show_mode = [&]
	{
		if (sets_bg)
		{
			return live_compute_options::background_show_mode::set_bg;
		}
		const auto valid_bg = channel_settings.background_ ? io_work.info_matches_except_complexity(channel_settings.background_->info()) : false;
		if (valid_bg)
		{
			return live_compute_options::background_show_mode::show_bg_subtracted;
		}
		return live_compute_options::background_show_mode::regular;
	}();
	const live_compute_options processing(false, background_show_mode);
	const compute_engine::work_function render_function = [&](camera_frame<float>& frame)
	{
		const gui_message no_message;
		render_engine->paint_surface(false, frame, no_message);
	};
	const compute_engine::work_function write_function = [&](camera_frame<float>& img_host)
	{
		using std::cout, std::endl;
		LOGGER_INFO("writing the fucking shit. go ask gabriel popescu to fuck himself with a rock");
		raw_io_work<float> write_me(img_host, io_work, gui_message_kind::none);
		compute_engine::write_image(write_me, out_dir);
	};
	if (!has_write && !has_display && !has_direct)
	{
		qli_runtime_error("Shouldn't happen");
	}
	if (!has_write && !has_display && has_direct)
	{
		qli_runtime_error("Shouldn't happen");
	}
	if (!has_write && has_display && !has_direct)
	{
		//push
		//render+pop
		const auto frames_made = cp->push_work(static_cast<camera_frame<unsigned short>>(io_work), channel_settings, processing);
		for (auto i = 0; i < frames_made; ++i)
		{
			cp->get_work_gpu(render_function, true);
		}
	}
	else if (!has_write && has_display && has_direct)
	{
		//push
		//render+pop
		const auto frames_made = cp->push_work(static_cast<camera_frame<unsigned short>>(io_work), channel_settings, processing);
		for (auto i = 0; i < frames_made; ++i)
		{
			cp->get_work_gpu(render_function, true);
		}
	}
	else if (has_write && !has_display && !has_direct)
	{
		//push
		//write gpu
		const auto frames_made = cp->push_work(static_cast<camera_frame<unsigned short>>(io_work), channel_settings, processing);
		for (auto i = 0; i < frames_made; ++i)
		{
			cp->get_work_host(write_function, true);
		}
	}
	else if (has_write && !has_display && has_direct)
	{
		//write direct
		compute_engine::write_image(io_work, out_dir);
	}
	else if (has_write && has_display && !has_direct)
	{
		//push
		//render+save
		//write gpu
		const auto frames_made = cp->push_work(static_cast<camera_frame<unsigned short>>(io_work), channel_settings, processing);
		for (auto i = 0; i < frames_made; ++i)
		{
			cp->get_work_gpu(render_function, false);
			cp->get_work_host(write_function, true);
		}
	}
	else if (has_write && has_display && has_direct)
	{
		//push
		//render+pop
		//write direct
		const auto frames_made = cp->push_work(static_cast<camera_frame<unsigned short>>(io_work), channel_settings, processing);
		for (auto i = 0; i < frames_made; ++i)
		{
			cp->get_work_gpu(render_function, true);
		}
		compute_engine::write_image(io_work, out_dir);
	}
}

io_worker::~io_worker()
{
	const time_slice t("Waiting for IO to finish");
	kill_join();
}
