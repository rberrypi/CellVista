#pragma once
#ifndef SPINNAKER_CAMERA_H
#define SPINNAKER_CAMERA_H
#include "camera_device.h"
struct spinnaker_camera_impl;
class spinnaker_camera final : public camera_device // todo make this an abstract class
{
	Q_OBJECT

		void switch_to_software_trigger() const;
	std::unique_ptr<spinnaker_camera_impl> impl;
	void debug_fps(const char* file,  int line) const;
public:
	explicit spinnaker_camera(int camera_idx, QObject* parent = nullptr);
	virtual ~spinnaker_camera();
	void trigger_internal() override;
	bool capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& timeout) override;
	void flush_camera_internal_buffer() override;

	[[nodiscard]] QStringList get_gain_names_internal() const override;
	void fix_camera_internal() override;
	void apply_settings_internal(const camera_config& new_config)  override;
	void set_cooling_internal(bool) override;
	void set_exposure_internal(const std::chrono::microseconds& exposure)  override;
	void print_debug(std::ostream&)  override;
	void start_capture_internal()  override;
	void stop_capture_internal()  override;
	[[nodiscard]] std::chrono::microseconds get_min_exposure_internal() override;
	[[nodiscard]] std::chrono::microseconds get_readout_time_internal() override;
	[[nodiscard]] std::chrono::microseconds get_transfer_time_internal() override;

	[[nodiscard]] int get_internal_buffer_count() const override;
};

#endif