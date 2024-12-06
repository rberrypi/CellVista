#pragma once
#ifndef SLM_DEVICE_H
#define SLM_DEVICE_H

#include <vector>
#include <mutex>
#include <boost/core/noncopyable.hpp>
#include "slm_state.h"
#include "frame_size.h"
#include <filesystem>
#include <QImage>
struct channel_settings;

struct slm_frame final
{
	std::vector<unsigned char> data;
	bool dirty;// if dirty then re-render, only do this on demand
};

enum class virtual_slm_type { point_retarder, medium, large };
class slm_device : boost::noncopyable, public frame_size, protected slm_state
{
	QImage alignment_patterns;
	std::recursive_mutex protect_patterns_;
	void fill_symbol(unsigned char* buff, unsigned char outer, unsigned char inner,const frame_size& slm_size);
	static void fill_custom_pattern(unsigned char* buff, const std::filesystem::path& full_path, const frame_size& slm_size);
public:
	[[nodiscard]] slm_state get_modulator_state() const;
	void load_modulator_state(const slm_state& slm_state);
	//note channel settings can change with custom channels
	bool is_retarder;
	void load_pattern(int pattern_idx);
	void load_patterns(int slm_index, const per_modulator_saveable_settings& settings);//and FYI dump everything
	slm_device(int width, int height, bool is_retarder);
	virtual ~slm_device() = default;
	//custom settings can actually resize the settings!!!
	bool set_frame(int frame_number);//always asynchronous
	void set_frame_await(int frame_number, const std::chrono::microseconds& slm_delay_ms, bool wait_on);
	int get_frame_number();
	int get_frame_number_total();

	[[nodiscard]] virtual bool has_high_speed_mode() const noexcept
	{
		return false;
	}
	void toggle_mode(slm_trigger_mode mode);
	virtual void fix_hardware_trigger();
	void hardware_trigger_sequence(size_t capture_items, const channel_settings& channel_settings);
	const unsigned char* get_frame(int frame);
	[[nodiscard]] virtual std::chrono::microseconds vendor_stability_time() const = 0;
protected:
	//
	virtual void hardware_trigger_sequence_internal(size_t capture_items, const channel_settings& channel_settings);
	virtual void toggle_mode_internal(slm_trigger_mode);
	virtual void load_frame_internal(int num) = 0;//<-Copies frame number into internal SLM buffer space
	virtual void set_frame_internal(int num) = 0;//<-Sets the frame, on some devices same as copying
	//
	std::vector<slm_frame> frame_data_;
};

#endif
