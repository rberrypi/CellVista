#pragma once
#ifndef CAMERA_DEVICE_H
#define CAMERA_DEVICE_H

#include <QObject>
#include <QStringList>
#include <mutex>
#include <deque>
#include <boost/core/noncopyable.hpp>
#include "capture_item.h"
#include "camera_frame.h"
#include "camera_config.h"
#include "channel_settings.h"

typedef  long long long_width;
struct camera_aoi final
{
	long_width width, height, top, left;
	camera_aoi(const long_width width, const long_width height, const long_width top, const long_width left) noexcept : width(width), height(height), top(top), left(left)
	{
	}
	camera_aoi(const long_width width, const long_width height) noexcept : camera_aoi(width, height, 0, 0)
	{

	}
	void re_center_and_fixup(long_width width_total, long_width height_total, long_width sensor_block_requirement = 1);

	[[nodiscard]] size_t n() const noexcept
	{
		return width * height;
	}

	[[nodiscard]] QString to_string() const
	{
		return QString("%1x%2").arg(width).arg(height);
	}

	[[nodiscard]] frame_size to_frame_size() const
	{
		return frame_size(width, height);
	}
};

struct camera_bin final
{
	long_width s;//scale factor where img = original/s
	explicit camera_bin(const long_width scale) noexcept
	{
		s = static_cast<long_width>(scale);
	}

	[[nodiscard]] QString to_string() const
	{
		return QString("%1x%1").arg(s);
	}
};

enum class camera_chroma { monochrome, forced_color, optional_color, optional_polarization };
struct camera_chroma_setting
{
	std::string name;
	demosaic_mode preferred_demosaic_mode;
	typedef std::map<camera_chroma, camera_chroma_setting> camera_chroma_settings_map;
	const static camera_chroma_settings_map settings;
	camera_chroma_setting(const std::string& name, const demosaic_mode demosaic_mode) : name(name), preferred_demosaic_mode(demosaic_mode)
	{

	}
	camera_chroma_setting() :camera_chroma_setting("", demosaic_mode::no_processing) {};
};

struct camera_contrast_features
{
	camera_chroma chroma;
	typedef std::set<demosaic_mode> demosaic_modes_set;
	demosaic_modes_set demosaic_modes;
	[[nodiscard]] bool no_bayer_modes() const
	{
		return demosaic_modes.size() == 1 && demosaic_modes.count(demosaic_mode::no_processing) == 1;
	}
	display_range raw_pixel_range;
	[[nodiscard]] bool is_forced_color() const noexcept
	{
		return chroma == camera_chroma::forced_color;
	}
	static int samples_per_pixel(camera_chroma chroma, demosaic_mode demosaic);
	camera_contrast_features(const camera_chroma chroma, const demosaic_modes_set& demosaic, const display_range& native_pixel_range) : chroma(chroma), demosaic_modes(demosaic), raw_pixel_range(native_pixel_range) {}

	camera_contrast_features(const camera_chroma chroma, const demosaic_mode& demosaic, const display_range& native_pixel_range) : chroma(chroma), demosaic_modes({ demosaic }), raw_pixel_range(native_pixel_range)
	{

	}

	camera_contrast_features() : camera_contrast_features(camera_chroma::monochrome, { demosaic_mode::no_processing }, display_range()) {}
};

struct camera_device_features : camera_contrast_features
{
	const bool is_virtual_camera;
	const bool has_burst_mode;//NOTE the frame rate is the minimal exposure!
	const bool has_async_mode;
	const bool has_cooling;
	camera_device_features(const bool has_burst_mode, const bool has_async_mode, const bool has_cooling, const bool is_virtual_camera, const camera_contrast_features& contrast) :camera_contrast_features(contrast), is_virtual_camera(is_virtual_camera), has_burst_mode(has_burst_mode), has_async_mode(has_async_mode), has_cooling(has_cooling) {}
	camera_device_features() :camera_device_features(false, false, false, false, camera_contrast_features()) {}
};


enum class camera_device_state { camera_ready_software, camera_capturing_software, camera_ready_burst, camera_capturing_burst, camera_ready_hardware_trigger, camera_ready_hardware_trigger_modulator_synced, camera_capturing_hardware_triggering };

const std::unordered_map<camera_device_state, camera_mode> camera_device_state_camera_mode = { { camera_device_state::camera_ready_software ,camera_mode::software },{ camera_device_state::camera_capturing_software ,camera_mode::software },{ camera_device_state::camera_ready_burst ,camera_mode::burst },{ camera_device_state::camera_capturing_burst ,camera_mode::burst },{ camera_device_state::camera_ready_hardware_trigger ,camera_mode::hardware_trigger },{ camera_device_state::camera_capturing_hardware_triggering ,camera_mode::hardware_trigger } };

const std::unordered_map<camera_mode, camera_device_state> camera_mode_idle_camera_device_state = { { camera_mode::software, camera_device_state::camera_ready_software } ,{ camera_mode::hardware_trigger, camera_device_state::camera_ready_hardware_trigger },{ camera_mode::burst, camera_device_state::camera_ready_burst } };


//todo there is a bug where the minimal exposure isn't checked for burst mode
class camera_device : public QObject, public camera_device_features, boost::noncopyable// todo make this an abstract class
{
	//todo move stuff into the protect, private spaces

	Q_OBJECT

		std::mutex queue_pat_m_;
	std::condition_variable queue_pat_cv_;
	std::deque<frame_meta_data_before_acquire> queue_pat_;
	std::recursive_mutex capture_mutex_;//Just in case?

	mutable QStringList gain_names, bin_names, aoi_names;
public:
	explicit camera_device(const camera_device_features& features, int camera_idx, QObject* parent);
	virtual ~camera_device();
	int default_light_path_index;
	[[nodiscard]] void assert_valid_queue_for_sync()  noexcept;
	const static std::chrono::microseconds undefined_exposure;

	[[nodiscard]] int trigger(const frame_meta_data_before_acquire& meta_data, int max_queue_length = std::numeric_limits<int>::max());
	typedef  std::function<void(camera_frame<unsigned short>)>	camera_frame_processing_function;

	[[nodiscard]] bool capture_burst(const std::pair<std::vector<capture_item>::const_iterator, std::vector<capture_item>::const_iterator>& frames, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& frame_time_out, const camera_frame_processing_function& process_function);
	size_t capture_hardware_sequence(const camera_frame_processing_function& fill_me, size_t capture_items, const frame_meta_data_before_acquire& prototype, const channel_settings& channel_settings);
	struct capture_delays
	{
		std::chrono::microseconds slm_delay;
		std::chrono::microseconds exposure_time;
		std::chrono::microseconds move_delay;
		capture_delays(const std::chrono::microseconds slm_delay, const std::chrono::microseconds exposure_time, const std::chrono::microseconds move_delay) noexcept : slm_delay(slm_delay), exposure_time(exposure_time), move_delay(move_delay)
		{

		}
		[[nodiscard]] std::chrono::microseconds get_delays(const std::chrono::microseconds cycle_delay) const noexcept
		{
			return std::max(slm_delay + exposure_time, cycle_delay) + move_delay;
		}
	};
	enum class capture_result { good, failure, stop_capture };
	capture_result capture(const camera_frame_processing_function& fill_me, const std::chrono::microseconds& move_delay);

	void fix_camera();
	[[nodiscard]] static bool is_a_idle_mode(camera_device_state state);
	void trigger_release_capture();
	void undo_release_capture_trigger();
	[[nodiscard]] frame_size get_sensor_size(const camera_config& query) const;
	[[nodiscard]] size_t get_sensor_bytes(const camera_config& query) const;
	[[nodiscard]] frame_size max_aoi_size() const;
	[[nodiscard]] camera_config get_camera_config() const noexcept;
	//Properties
	[[nodiscard]] QStringList get_aoi_names() const;
	[[nodiscard]] QString get_aoi_name(int idx_aoi) const;
	[[nodiscard]] QStringList get_bin_names() const;
	[[nodiscard]] QStringList get_gain_names() const;

	[[nodiscard]] std::chrono::microseconds get_min_exposure() const noexcept;
	[[nodiscard]] std::chrono::microseconds get_readout_time() const noexcept;
	[[nodiscard]] std::chrono::microseconds get_transfer_time() const noexcept;

	[[nodiscard]] std::chrono::microseconds get_exposure() const noexcept;
	void apply_settings(const camera_config& new_config);

	[[nodiscard]] bool is_new_config(const camera_config& new_config) const;
	void set_exposure(const std::chrono::microseconds& exposure);
	void start_software_capture();
	void stop_software_capture();
	//void stop_burst_capture();
	[[nodiscard]] bool get_cooling() const;
	virtual void print_debug(std::ostream& input) = 0;
	//
	std::vector<camera_aoi> aois;
	std::vector<camera_bin> bin_modes;
	//
	bool first_capture;//protected by first mutex
	//
	[[nodiscard]] virtual int get_internal_buffer_count() const = 0;
	//
protected:
	std::recursive_mutex trigger_mutex_;
	std::chrono::microseconds exposure_, min_exposure_, readout_time_, transfer_time_;
	camera_config camera_configuration_;
	bool cooling_enabled_;
	virtual void flush_camera_internal_buffer() = 0;
	virtual void set_exposure_internal(const std::chrono::microseconds& exposure) = 0;
	virtual void apply_settings_internal(const camera_config& new_config) = 0;
	virtual void set_cooling_internal(bool enable) = 0;
	virtual void fix_camera_internal() = 0;
	virtual void trigger_internal() = 0;
	virtual bool capture_internal(const camera_frame_processing_function& fill_me, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& timeout) = 0;
	virtual bool capture_burst_internal(const std::pair<std::vector<capture_item>::const_iterator, std::vector<capture_item>::const_iterator>& /*frames*/, const frame_meta_data_before_acquire& /*meta_data*/, const std::chrono::microseconds& /*exposure*/, const std::chrono::microseconds& /*frame_time_out*/, const camera_frame_processing_function& /*process_function*/);
	virtual size_t capture_hardware_sequence_internal(const camera_frame_processing_function& fill_me, size_t frames, const frame_meta_data_before_acquire& prototype, const  channel_settings& channel_settings);

	[[nodiscard]] virtual QStringList get_gain_names_internal() const = 0;
	virtual void start_capture_internal() = 0;
	virtual void stop_capture_internal() = 0;
	[[nodiscard]] virtual std::chrono::microseconds get_min_exposure_internal() = 0;
	[[nodiscard]] virtual std::chrono::microseconds get_readout_time_internal() = 0;
	[[nodiscard]] virtual std::chrono::microseconds get_transfer_time_internal() = 0;
	void common_post_constructor();
	camera_device_state state_;//maybe atomic? actually this needs to go because its already in mode?
	void set_cooling(bool enable);
	int camera_device_idx_;
signals:
	//TS version
	void min_exp_changed(std::chrono::microseconds time);
	void min_exp_changed_ts(std::chrono::microseconds time); // For example when the frame size changes
};

#endif