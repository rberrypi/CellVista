#pragma once
#ifndef SLM_HOLDER_H
#define SLM_HOLDER_H
#include <boost/core/noncopyable.hpp>
#include <boost/container/static_vector.hpp>
#include "slm_state.h"
#include "frame_size.h"
#include "modulator_configuration.h"
#include <mutex>
#include <memory>

struct channel_settings;
class slm_device;
class QProgressDialog;
struct slm_frame_pointer : frame_size
{
	const uint8_t* frame_pointer;
	slm_frame_pointer() noexcept: slm_frame_pointer(nullptr, frame_size()) {}
	slm_frame_pointer(const uint8_t* pointer, const frame_size& frame) noexcept:frame_size(frame), frame_pointer(pointer)
	{
	}

	[[nodiscard]] bool is_valid() const noexcept
	{
		return frame_pointer != nullptr && n() > 0;
	}
};
typedef boost::container::small_vector<slm_frame_pointer, 2> slm_frame_pointers;

class slm_holder : boost::noncopyable
{
	mutable std::mutex slm_consistency;
	boost::container::static_vector<std::unique_ptr<slm_device>, max_slms> slms;
	void slm_consistency_check() const;
public:
	slm_holder();
	~slm_holder();
	boost::container::small_vector<bool, 2> has_retarders() const;
	[[nodiscard]] bool has_retarder() const;
	void reload_settings();
	void load_slm_settings(const fixed_modulator_settings& settings, bool bake_all_patterns);
	void set_slm_frame_await(int frame_number, const std::chrono::microseconds& slm_delay_ms, bool wait_on);
	[[nodiscard]] bool set_slm_frame(int frame_number);//always asynchronous
	void set_slm_mode(slm_trigger_mode mode);
	[[nodiscard]] int get_slm_frames() const;
	[[nodiscard]] int get_slm_frame_idx() const;
	[[nodiscard]] slm_dimensions get_slm_dimensions() const;
	[[nodiscard]] int get_slm_count() const;
	[[nodiscard]] slm_frame_pointers get_slm_frames(int frame_idx) const;
	void toggle_slm_mode(slm_trigger_mode mode);
	void trigger_slm_hardware_sequence(size_t capture_items, const channel_settings& channel_settings);
	[[nodiscard]] std::chrono::microseconds max_vendor_stability_time() const;
	[[nodiscard]] slm_states get_modulator_states() const;
	void load_modulator_states(const slm_states& slm_states);
	//
	[[nodiscard]] fixed_modulator_settings get_settings() const;
	void write_slm_directory(const std::string& directory, QProgressDialog& progress_dialog);
};
#endif