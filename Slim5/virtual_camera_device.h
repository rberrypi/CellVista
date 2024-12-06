#pragma once
#ifndef VIRTUAL_CAMERA_DEVICE_H
#define VIRTUAL_CAMERA_DEVICE_H
#include <condition_variable>
#include <queue>
#include <thrust/device_vector.h>
#include "camera_device.h"
#include "virtual_camera_shared.h"
struct acquisition_simulation final
{
	std::chrono::microseconds time;
	int pattern;
};

struct virtual_camera_image final : image_info
{
	std::vector<unsigned short> img;
	explicit virtual_camera_image(const image_info& image_info) : image_info(image_info)
	{
		const auto elements = n() * samples_per_pixel;
		const auto blank = static_cast<unsigned short>(0);
		img.resize(elements, blank);
	}
	void resize(const image_info& info)
	{
		static_cast<image_info&>(*this) = info;
		img.resize(samples());
	}
	virtual_camera_image() : virtual_camera_image(image_info()) {}
};

struct virtual_camera_image_prepared
{
	virtual_camera_image img, bg;
};

struct gpu_frame_set
{
	thrust::device_vector<unsigned short> img, background;
};

struct gpu_loaded_frame_set final : image_info
{
	boost::container::static_vector< gpu_frame_set, typical_psi_patterns > frames;
};


class virtual_camera_device final : public camera_device
{

	Q_OBJECT

	std::vector<unsigned short> output_buffer;
	std::vector<virtual_camera_image_prepared> prepared_images_;
	virtual_camera_image& get_prepared_image(int pattern, int aoi, int bin, bool background);
	void make_virtual_images(const gpu_loaded_frame_set& patterns);
	static virtual_camera_type prompt_for_virtual_camera();
	int pattern_count;
	gpu_loaded_frame_set make_aois(const virtual_camera_type& camera_type);
	virtual_camera_image blank_frame;
	std::queue<acquisition_simulation> frame_queue_simulation_;
	std::mutex frame_queue_simulation_m_;
	std::condition_variable frame_queue_simulation_cv_;
	static constexpr const char* resource_file_name = "virtualcamera.rcc";
public:
	static bool register_resource();
	explicit virtual_camera_device(virtual_camera_type camera_type, int camera_idx, QObject* parent = nullptr);
	[[nodiscard]] QStringList get_gain_names_internal() const override;
	void flush_camera_internal_buffer() override;
	void fix_camera_internal() override;
	void trigger_internal() override;
	bool capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& frame_meta_data_before_acquire, const std::chrono::microseconds&
		timeout) override;
	typedef std::vector<capture_item>::const_iterator capture_item_iterator;
	bool capture_burst_internal(const std::pair<capture_item_iterator, capture_item_iterator>& frames, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& exposure, const std::chrono::microseconds& frame_time_out, const camera_frame_processing_function& processing_function) override;
	void apply_settings_internal(const camera_config& config) override;
	void set_exposure_internal(const std::chrono::microseconds& exposure) override;
	void print_debug(std::ostream& input) override;
	void start_capture_internal() override;
	void stop_capture_internal() override;
	[[nodiscard]] std::chrono::microseconds get_min_exposure_internal() override;
	[[nodiscard]] std::chrono::microseconds get_readout_time_internal() override;
	[[nodiscard]] std::chrono::microseconds get_transfer_time_internal() override;
	void set_cooling_internal(bool enable) override;

	[[nodiscard]] int get_internal_buffer_count() const override;

	//
};


#endif