#pragma once
#ifndef ZEISS_DEVICE
#define ZEISS_DEVICE
#include "camera_device.h"
#include <condition_variable>
#include <vector>
#include <queue>
#include <boost/align/aligned_allocator.hpp>
#include <iostream>


#include "qli_runtime_error.h"
struct zeiss_camera_imp;
class zeiss_camera final : public camera_device // todo make this an abstract class
{
	Q_OBJECT

		std::unique_ptr<zeiss_camera_imp> f_;
	std::vector<std::vector<unsigned short, boost::alignment::aligned_allocator<unsigned short, 32> >> camera_buffer_;
	std::queue<unsigned short*> free_buffer_, inside_camera_;
	std::mutex free_buffer_m_, inside_camera_m_;
	std::condition_variable inside_camera_cv_, camera_buffer_queue_cv_;
	bool camera_buffer_queue_m_kill_;
	const static int internal_buffers = 20;
public:
	explicit zeiss_camera(int camera_idx, QObject* parent = nullptr);
	virtual ~zeiss_camera();
	void trigger_internal() override;
	bool capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& timeout) override;

	bool  capture_burst_internal(const std::pair<std::vector<capture_item>::const_iterator, std::vector<capture_item>::const_iterator>& /*frames*/, const frame_meta_data_before_acquire& /*meta_data*/, const std::chrono::microseconds& /*exposure*/, const std::chrono::microseconds& /*frame_time_out*/, const camera_frame_processing_function& /*process_function*/) override
	{
		std::cout << "capture burst internal in header file" << std::endl;
		qli_not_implemented();
	}

	virtual size_t capture_hardware_sequence_internal(const camera_frame_processing_function& fill_me, size_t frames, const frame_meta_data_before_acquire& prototype, const  channel_settings& channel_settings) override
	{
		std::cout << "capture hardware sequence internal in header file" << std::endl;
		qli_not_implemented();
	}

	void flush_camera_internal_buffer() override
	{
		//unimplemented
	}

	[[nodiscard]] QStringList get_gain_names_internal() const override
	{
		QStringList names;
		names.append(tr("Not Implemented"));
		return names;
	}

	void fix_camera_internal() override;
	void apply_settings_internal(const camera_config& new_config)  override;
	void set_cooling_internal(bool enable) override;
	void set_exposure_internal(const std::chrono::microseconds& exposure)  override;
	void print_debug(std::ostream& input)  override;
	void start_capture_internal()  override;
	void stop_capture_internal()  override;
	[[nodiscard]] std::chrono::microseconds get_min_exposure_internal() override;
	[[nodiscard]] std::chrono::microseconds get_readout_time_internal() override;
	[[nodiscard]] std::chrono::microseconds get_transfer_time_internal() override
	{
		std::cout << "get transfer time internal in header file" << std::endl;
		return std::chrono::microseconds(0);
		// qli_not_implemented();
	}
	// [[nodiscard]] std::chrono::microseconds get_min_cycle_internal() override;

	[[nodiscard]] int get_internal_buffer_count() const override
	{
		//so this is kinda bullshit right?
		return 30;
	}
};

#endif