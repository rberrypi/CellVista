#pragma once
#ifndef BSI_CAMERA_H
#define BSI_CAMERA_H

#include "camera_device.h"
#include <vector>
#include "page_locked_allocator.h"

//


//todo refactor the fuck out of this
class bsi_device final : public camera_device // todo make this an abstract class
{
	Q_OBJECT
		std::vector<unsigned short, page_locked_allocator<unsigned short>> data;
	unsigned int last_exposure;
public:
	explicit bsi_device(int camera_idx, QObject* parent = nullptr);
	virtual ~bsi_device();
	void trigger_internal() override;
	bool capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& timeout) override;
	void fix_camera_internal() override;
	void apply_settings_internal(const camera_config& new_config)  override;
	void set_cooling_internal(bool enable) override;
	void set_exposure_internal(const std::chrono::microseconds& exposure)  override;
	void print_debug(std::ostream& input)  override;
	void start_capture_internal()  override;
	void stop_capture_internal()  override;
	[[nodiscard]] std::chrono::microseconds get_min_exposure_internal() override;
	[[nodiscard]] std::chrono::microseconds get_readout_time_internal() override;
	[[nodiscard]] std::chrono::microseconds get_transfer_time_internal() override;
	[[nodiscard]] int get_internal_buffer_count() const override;
	[[nodiscard]] QStringList get_gain_names_internal() const override;
	void flush_camera_internal_buffer() override;
};
#endif