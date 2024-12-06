#pragma once
#ifndef ANDORDEVICE
#define ANDORDEVICE

#include "camera_device.h"
#include <vector>
#include <queue>
#include <boost/align/aligned_allocator.hpp>
//
#define AT_TIMEOUT (1000)

//todo refactor the fuck out of this
class andor_device final : public camera_device // todo make this an abstract class
{
	Q_OBJECT

		const static auto chasing_the_avx_512_dream = 16;//actually AVX 512 isn't supported in the kernel :-(
	std::vector<std::vector<unsigned char, boost::alignment::aligned_allocator<unsigned char, chasing_the_avx_512_dream> >> camera_buffer_;
	std::vector<unsigned char, boost::alignment::aligned_allocator<unsigned char, chasing_the_avx_512_dream> > bit_convert_buffer_;
	std::queue<unsigned char*> inside_camera_;
	const static int andor_internal_pool_count = 50;
	void flush_camera_internal_buffer() override;
	[[nodiscard]] QStringList get_gain_names_internal() const override;
	void allocate_memory_pool();
	void align_and_convert_andor_buffer(const camera_frame_processing_function& fill_me, const camera_frame<unsigned short>& ptr_raw);
	//
	std::chrono::microseconds andor_transfer_rate_;
	[[nodiscard]] static std::chrono::microseconds timestamp_delta(unsigned char* p_buf, size_t buffer_size, long long clock_rate) noexcept;

public:
	//Properties
	long long clock_rate;
	[[nodiscard]] std::chrono::microseconds get_min_copy_back_in_mili() const;
	[[nodiscard]] std::chrono::microseconds get_min_fps_in_mili() const;
	explicit andor_device(int camera_idx, QObject* parent = nullptr);
	virtual ~andor_device();
	void trigger_internal() override;
	bool capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& timeout) override;
	bool capture_burst_internal(const std::pair<std::vector<capture_item>::const_iterator, std::vector<capture_item>::const_iterator>& frames, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& exposure, const std::chrono::microseconds& frame_time_out, const camera_frame_processing_function& process_function)  override;
	[[nodiscard]] size_t capture_hardware_sequence_internal(const camera_frame_processing_function& process_function, size_t capture_items, const frame_meta_data_before_acquire& prototype, const channel_settings& channel_settings) override;
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

	[[nodiscard]] int get_internal_buffer_count() const noexcept override 
	{
		return andor_internal_pool_count;
	}


};
/*Check on this*/
#endif