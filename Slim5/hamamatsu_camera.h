#pragma once
#ifndef HAMAMATSU_DEVICE_H
#define HAMAMATSU_DEVICE_H
#include "camera_device.h"
class pre_allocated_pool;

class hamamatsu_device final : public camera_device
{

	Q_OBJECT

		static const long internal_frame_count = 300;
public:
	explicit hamamatsu_device(int camera_idx, QObject* parent = nullptr);
	virtual ~hamamatsu_device();

	[[nodiscard]] QStringList get_gain_names_internal() const override
	{
		QStringList in_modes;
		in_modes << "none";
		return in_modes;
	}
	void flush_camera_internal_buffer() override;

	void set_exposure_internal(const std::chrono::microseconds& exposure)  override;

	void apply_settings_internal(const camera_config& new_config)  override;

	void set_cooling_internal(bool enable)override;

	void fix_camera_internal() override;

	void trigger_internal() override;

	bool capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& timeout) override;
	
	bool capture_burst_internal(const std::pair<std::vector<capture_item>::const_iterator, std::vector<capture_item>::const_iterator>& frames, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& exposure, const std::chrono::microseconds& frame_time_out, const camera_frame_processing_function& process_function)  override;
	
	void start_capture_internal()  override;

	void stop_capture_internal()  override;

	[[nodiscard]] std::chrono::microseconds get_min_exposure_internal() override;
	[[nodiscard]] std::chrono::microseconds get_readout_time_internal() override;
	[[nodiscard]] std::chrono::microseconds get_transfer_time_internal() override;

	void print_debug(std::ostream& input)  override;

	[[nodiscard]] int get_internal_buffer_count() const override
	{
		return internal_frame_count;
	}
};
#endif