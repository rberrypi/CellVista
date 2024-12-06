#pragma once
#ifndef FLY_CAPTURE_DEVICE_H
#define FLY_CAPTURE_DEVICE_H
#include "camera_device.h"

struct pre_allocated_pool;

// ReSharper disable CppInconsistentNaming
namespace FlyCapture2
{
	class Camera;
}
// ReSharper restore CppInconsistentNaming

class fly_capture_device final : public camera_device // todo make this an abstract class
{
	Q_OBJECT

		FlyCapture2::Camera* cam_;
	void fire_software_trigger() const;
	void wait_for_power_on() const;
	void poll_for_trigger_ready() const;
public:
	explicit fly_capture_device(int camera_idx, QObject* parent = Q_NULLPTR);

	[[nodiscard]] QStringList get_gain_names_internal() const override
	{
		QStringList in_modes;
		in_modes << "none";
		return in_modes;
	}
	virtual ~fly_capture_device();
	void trigger_internal() override;
	bool capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& timeout) override;
	void fix_camera_internal() override;
	void apply_settings_internal(const camera_config& new_config)  override;
	void set_cooling_internal(bool enable) override;

	void flush_camera_internal_buffer() override
	{
		//probably should implement!!!
	}

	void set_exposure_internal(const std::chrono::microseconds& exposure)  override;
	void print_debug(std::ostream& input)  override;
	void start_capture_internal()  override;
	void stop_capture_internal()  override;
	[[nodiscard]] std::chrono::microseconds get_min_exposure_internal() override;
	[[nodiscard]] std::chrono::microseconds get_readout_time_internal() override;
	[[nodiscard]] std::chrono::microseconds get_transfer_time_internal() override;

	[[nodiscard]] int get_internal_buffer_count() const override
	{
		return 30;
	}
};
#endif