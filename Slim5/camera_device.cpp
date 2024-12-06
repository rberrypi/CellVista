#include "stdafx.h"
#include "camera_device.h"
#include "qli_runtime_error.h"
#include "time_slice.h"
#include "ml_shared.h"

const auto logic_error_str = "Camera State Machine Logic Error";

const std::chrono::microseconds  camera_device::undefined_exposure = std::chrono::microseconds::max();

void twenty_checkpoint()
{
#if 0
	{
		const auto fity = 50;
		static std::array<std::chrono::microseconds, fity> test_list;
		static auto idx = 0;
		test_list[idx] = timestamp();
		if (idx == (test_list.size() - 1))
		{
			std::array<std::chrono::microseconds, fity> result;
			std::adjacent_difference(test_list.begin(), test_list.end(), result.begin());
			for (auto what : result)
			{
				std::cout << "TEST " << to_mili(what) << std::endl;
			}
		}
		idx = (idx + 1) % test_list.size();
	}
#endif
}

int camera_contrast_features::samples_per_pixel(const camera_chroma chroma, const demosaic_mode demosaic)
{
	constexpr auto color = 3;
	constexpr auto bw = 1;
	const auto is_color_processing = demosaic_setting::info.at(demosaic).will_be_color;
	const auto demosaic_selected = is_color_processing ? color : bw;
	//not some of these shouldn't really happen
	switch (chroma)
	{
	case camera_chroma::monochrome:
	case camera_chroma::optional_color:
	case camera_chroma::optional_polarization:
		return demosaic_selected;
	case camera_chroma::forced_color:
		return color;
	default:
		qli_not_implemented();
	}
}

camera_device::camera_device(const camera_device_features& features, const int camera_idx, QObject* parent) : QObject(parent), camera_device_features(features), default_light_path_index(0), first_capture(true), exposure_(std::chrono::microseconds::max()), min_exposure_(std::chrono::microseconds::max()), readout_time_(std::chrono::microseconds::max()), transfer_time_(std::chrono::microseconds::max()), camera_configuration_(camera_config::invalid_cam_config()), cooling_enabled_(false), state_(camera_device_state::camera_ready_software), camera_device_idx_(camera_idx)
{
	qRegisterMetaType<frame_size>();
	connect(this, &camera_device::min_exp_changed, this, &camera_device::min_exp_changed_ts, Qt::QueuedConnection);
}

camera_device::~camera_device()  // NOLINT(bugprone-exception-escape)
{
	if (state_ != camera_device_state::camera_ready_software)
	{
		//this shuts up the static analysis
		//The behavior is unambiguous: throwing an uncaught exception in the destructor calls std::terminate
		//I KNOW WHAT I'M DOING!!!
#pragma warning( push )
#pragma warning( disable : 4297)
		qli_runtime_error(logic_error_str);
#pragma warning( pop ) 
	}
}

void camera_device::trigger_release_capture()
{
	{
		std::unique_lock<std::mutex> lk(queue_pat_m_);
		auto blank = frame_meta_data_before_acquire();
		blank.action = scope_action::shutdown_live;
		queue_pat_.push_back(blank);
	}
	queue_pat_cv_.notify_one();
}

void camera_device::undo_release_capture_trigger()
{
	std::unique_lock<std::mutex> lk(queue_pat_m_);
	if (!queue_pat_.empty())
	{
		const auto& item = queue_pat_.back();
		if (item.is_stop_capture())
		{
			queue_pat_.pop_back();//undo it
		}
	}
}

[[nodiscard]] void camera_device::assert_valid_queue_for_sync()  noexcept
{
	std::unique_lock<std::mutex> lk(queue_pat_m_);
	if (has_async_mode)
	{
		return;//skip this check
	}
	const auto function = [](const frame_meta_data_before_acquire& items)
	{
		return!items.is_stop_capture();
	};
	const auto item_count = std::count_if(queue_pat_.begin(), queue_pat_.end(), function);
	if (item_count > 1)
	{
		qli_runtime_error("Something Wrong");
	}
}

int camera_device::trigger(const frame_meta_data_before_acquire& meta_data, int max_queue_length)
{
	std::unique_lock<std::recursive_mutex> lk_trigger(trigger_mutex_);
	if (state_ != camera_device_state::camera_capturing_software)
	{
		qli_runtime_error(logic_error_str);
	}
	auto checked = 0;
	{
		std::unique_lock<std::mutex> lk(queue_pat_m_);
		const auto magic_dead_lock_prevention_hack_hack_hack = ms_to_chrono(100) + meta_data.exposure_time;
		const auto check = [&]
		{
			checked = checked + 1;
			const auto size = static_cast<int>(queue_pat_.size());
			return size < max_queue_length;
		};
		if (!queue_pat_cv_.wait_for(lk, magic_dead_lock_prevention_hack_hack_hack, check))
		{
			return checked;
		}
		set_exposure(meta_data.exposure_time);
		if (has_async_mode)
		{
			//At least min_cycle + last exposure should have elapsed before triggering on Andor models
			static auto last_exposure = ms_to_chrono(0);
			static auto last_call = ms_to_chrono(0);
			const auto last_frame = get_readout_time() +  last_exposure;
			const auto frame_rate_requirement = get_transfer_time();
			const auto required_time = std::max(last_frame, frame_rate_requirement);
			const auto current_time = timestamp();
			const auto elapsed = current_time - last_call;
			if (elapsed < required_time)
			{
				const auto remaining = required_time - elapsed;
				windows_sleep(remaining);
			}
			trigger_internal();//assumed to be synchronous
			last_exposure = meta_data.exposure_time;
			last_call = timestamp();
		}
		else
		{
			trigger_internal();
		}
		queue_pat_.push_back(meta_data);
	}
	queue_pat_cv_.notify_one();
	return checked;
}

bool camera_device::capture_burst(const std::pair<std::vector<capture_item>::const_iterator, std::vector<capture_item>::const_iterator>& frames, const frame_meta_data_before_acquire& meta_data, const std::chrono::microseconds& frame_time_out, const camera_frame_processing_function& process_function)
{
	if (state_ != camera_device_state::camera_ready_burst)
	{
		qli_runtime_error(logic_error_str);
	}
	const auto exposure = meta_data.exposure_time;
	if (exposure > frame_time_out)
	{
		qli_runtime_error(logic_error_str);

	}
	if (exposure.count() <= 0)
	{
		qli_runtime_error(logic_error_str);

	}
	state_ = camera_device_state::camera_capturing_burst;
	const auto success = capture_burst_internal(frames, meta_data, exposure, frame_time_out, process_function);
	state_ = camera_device_state::camera_ready_burst;
	return success;
}


size_t camera_device::capture_hardware_sequence(const camera_frame_processing_function& fill_me, const size_t capture_items, const frame_meta_data_before_acquire& prototype, const channel_settings& channel_settings)
{
	if (state_ != camera_device_state::camera_ready_hardware_trigger_modulator_synced)
	{
		qli_runtime_error(logic_error_str);

	}
	state_ = camera_device_state::camera_capturing_hardware_triggering;
	const auto captured = capture_hardware_sequence_internal(fill_me, capture_items, prototype, channel_settings);
	state_ = camera_device_state::camera_ready_hardware_trigger;
	return captured;
}

camera_device::capture_result camera_device::capture(const camera_frame_processing_function& fill_me, const std::chrono::microseconds& move_delay)
{
	std::unique_lock<std::recursive_mutex> lk_capture(capture_mutex_);
	if (state_ != camera_device_state::camera_capturing_software)
	{
		qli_runtime_error(logic_error_str);
	}
	frame_meta_data_before_acquire meta_data;
	{
		std::unique_lock<std::mutex> lk(queue_pat_m_);
		const auto predicate = [&]
		{
			return !queue_pat_.empty();
		};
		queue_pat_cv_.wait(lk, predicate);
		meta_data = queue_pat_.front();
		queue_pat_.pop_front();
		if (meta_data.is_stop_capture())
		{
			return capture_result::stop_capture;
		}
	}
	//
	const auto expected_timeout = [&] {
		auto delay = move_delay + meta_data.duration();
		if (first_capture)
		{
			// a lot of this bullshit is because of cameras we can't test (!)
			delay += get_transfer_time();
			first_capture = false;
		}
		return delay;
	}();
	//

	const auto success = capture_internal(fill_me, meta_data, expected_timeout);
	// on failure still lets it trigger, but that probably won't be a problem as subsequent triggers are screwed up 
	queue_pat_cv_.notify_one();
	return success ? camera_device::capture_result::good : camera_device::capture_result::failure;
}

void camera_device::fix_camera()
{
	if (!is_a_idle_mode(state_))
	{
		qli_runtime_error(logic_error_str);
	}
	std::unique_lock<std::mutex> lk(queue_pat_m_);
	fix_camera_internal();
}

size_t camera_device::get_sensor_bytes(const camera_config& query) const
{
	const auto frame_bytes = get_sensor_size(query).n() * sizeof(unsigned short);
	return frame_bytes;
}

frame_size camera_device::get_sensor_size(const camera_config& query) const
{
	const auto& info = aois.at(query.aoi_index);
	const auto binning = bin_modes.at(query.bin_index);
	auto original_frame_size = info.to_frame_size();
	original_frame_size.width = original_frame_size.width / binning.s;
	original_frame_size.height = original_frame_size.height / binning.s;
	return original_frame_size;
}

frame_size camera_device::max_aoi_size() const
{
	return aois.begin()->to_frame_size();
}

camera_config camera_device::get_camera_config() const noexcept
{
	return camera_configuration_;
}

QString camera_device::get_aoi_name(const int idx_aoi) const
{
	auto aoi_item = aois.at(idx_aoi).to_string();
	if (camera_device_idx_ > 0)
	{
		const auto str = QString("Cam%1 ").arg(camera_device_idx_ + 1);
		aoi_item.prepend(str);
	}
	return aoi_item;
}


QStringList camera_device::get_aoi_names() const
{
	if (aoi_names.empty())
	{
		for (size_t i = 0; i < aois.size(); i++)
		{
			aoi_names << get_aoi_name(i);
		}
	}
	return aoi_names;
}

QStringList camera_device::get_bin_names() const
{
	if (bin_names.empty())
	{
		for (const auto& i : bin_modes)
		{
			bin_names << i.to_string();
		}
	}
	return bin_names;
}

QStringList camera_device::get_gain_names() const
{
	if (gain_names.empty())
	{
		gain_names = get_gain_names_internal();
	}
	return gain_names;
}

std::chrono::microseconds camera_device::get_exposure() const noexcept
{
	return exposure_;
}

std::chrono::microseconds camera_device::get_min_exposure() const noexcept
{
	return min_exposure_;
}

std::chrono::microseconds camera_device::get_readout_time() const noexcept
{
	return readout_time_;
}

std::chrono::microseconds camera_device::get_transfer_time() const noexcept
{
	return transfer_time_;
}

void camera_device::apply_settings(const camera_config& new_config)
{
	if (new_config.camera_idx != camera_device_idx_)
	{
		qli_invalid_arguments();
	}
	if (is_new_config(new_config))
	{
		std::unique_lock<std::recursive_mutex> lk_capture(capture_mutex_);// changing settings will "acquiring" might cause certain acquisitions to fail, but I don't think this is much of an issue, because they will be repeated?
		std::unique_lock<std::recursive_mutex> lk_trigger(trigger_mutex_);
		const auto last_state = state_;
		if (last_state == camera_device_state::camera_capturing_software)
		{
			stop_software_capture();
		}
		flush_camera_internal_buffer();
		set_cooling(new_config.enable_cooling);
		apply_settings_internal(new_config);
		state_ = camera_mode_idle_camera_device_state.at(new_config.mode);
		//
		camera_configuration_ = new_config;
		//
		min_exposure_ = std::max(ms_to_chrono(1),get_min_exposure_internal());
		readout_time_ = get_readout_time_internal();
		transfer_time_ = get_transfer_time_internal();
		emit min_exp_changed(min_exposure_);
		{
			static size_t old_size = 0;
			const auto new_size = get_sensor_bytes(camera_configuration_);
			if (old_size != new_size)
			{
				old_size = new_size;
			}
		}
		if (last_state == camera_device_state::camera_capturing_software)
		{
			start_software_capture();
		}
	}
}

void camera_device::set_exposure(const std::chrono::microseconds& exposure)
{
	if (exposure.count() <= 0)
	{
#if _DEBUG
		qli_runtime_error(logic_error_str);
#endif
	}
	if (exposure_ != exposure)
	{
		set_exposure_internal(exposure);
		exposure_ = exposure;
	}
}

void camera_device::start_software_capture()
{
	std::unique_lock<std::recursive_mutex> lk_capture(capture_mutex_);
	std::unique_lock<std::recursive_mutex> lk_trigger(trigger_mutex_);
	if (state_ != camera_device_state::camera_ready_software)
	{
		qli_runtime_error(logic_error_str);
	}
	if (!queue_pat_.empty())
	{
		queue_pat_.clear();
	}
	flush_camera_internal_buffer();
	start_capture_internal();
	first_capture = true;
	state_ = camera_device_state::camera_capturing_software;
}

void camera_device::stop_software_capture()
{
	std::unique_lock<std::recursive_mutex> lk_capture(capture_mutex_);
	std::unique_lock<std::recursive_mutex> lk_trigger(trigger_mutex_);
	if (state_ != camera_device_state::camera_capturing_software)
	{
		qli_runtime_error(logic_error_str);
	}
	stop_capture_internal();
	state_ = camera_device_state::camera_ready_software;
}

void camera_device::common_post_constructor()
{
	std::sort(aois.begin(), aois.end(), [](const camera_aoi& a, const camera_aoi& b) {
		return b.n() < a.n();
		});
	//speeds up the program, because the debug version uses checked iterators which are useful and slow
	const auto initial_size = aois.size();
#if INCLUDE_ML == 1
	{
		const auto cmp = [](const camera_aoi in)
		{
			return !is_divisible_by_sixteen(in.width) || !is_divisible_by_sixteen(in.height);
		};
		aois.erase(std::remove_if(aois.begin(), aois.end(), cmp), aois.end()); //-V530  // NOLINT(bugprone-unused-return-value)
	}
	if (aois.empty())
	{
		qli_runtime_error("Test Vector Broken for AI, todo update");
	}
#endif
#if _DEBUG && 0
	if (aois.empty())
	{
		qli_runtime_error();
		}
	if (aois.size() > 1)
	{
		const auto min_frame_size = frame_size(512, 512).n();
		const auto min_element = *std::min_element(aois.begin(), aois.end(), [](const camera_aoi& a, const  camera_aoi& b)
			{
				return a.n() < b.n();
			});
		auto min_camera_pixel_count = std::max(min_frame_size, min_element.n());
		const auto cmp = [min_camera_pixel_count](const camera_aoi& in)
		{
			const auto pixel_count = in.n();
			const auto valid = pixel_count > min_camera_pixel_count;
			return valid;
		};
		aois.erase(std::remove_if(aois.begin(), aois.end(), cmp), aois.end());
	}
#endif
	if (aois.empty() || bin_modes.empty() || get_gain_names().isEmpty())
	{
		const auto msg = "Missing camera codes, initial size:" + std::to_string(initial_size);
		qli_runtime_error(msg);
	}
	auto starting_config = camera_config();
	starting_config.camera_idx = camera_device_idx_;
	apply_settings(starting_config);
	//
	//exposure_ = undefined_exposure;//next call will set this
	const auto min_exposure = get_min_exposure();
	set_exposure(min_exposure);

}

void camera_device::set_cooling(const bool enable)
{
	//should only be called from apply settings because of weird stuff with Andor cameras
	if (has_cooling)
	{
		set_cooling_internal(enable);
		cooling_enabled_ = enable;
	}
}

bool camera_device::get_cooling() const
{
	return cooling_enabled_;
}

void camera_aoi::re_center_and_fixup(const long_width width_total, const long_width height_total, long_width sensor_block_requirement)
{
	if (width > width_total || height > height_total)
	{
		qli_runtime_error(logic_error_str);
	}
	//andor starts at one, this might fuck over other cameras!!!
	left = (width_total - width) / 2;
	top = (height_total - height) / 2;
	//
	const auto match_to_block = [sensor_block_requirement](const auto value)
	{
		return floor(value / sensor_block_requirement) * sensor_block_requirement;
	};
	left = match_to_block(left);
	top = match_to_block(top);
	height = match_to_block(height);
	width = match_to_block(width);
}


bool camera_device::is_a_idle_mode(const camera_device_state state)
{
	std::set<camera_device_state> camera_device_state_is_idle = { camera_device_state::camera_ready_software , camera_device_state::camera_ready_burst  , camera_device_state::camera_ready_hardware_trigger };
	return camera_device_state_is_idle.find(state) != camera_device_state_is_idle.end();
}

[[nodiscard]] bool camera_device::is_new_config(const camera_config& new_config) const
{
	return new_config != camera_configuration_;
}


size_t camera_device::capture_hardware_sequence_internal(const camera_frame_processing_function&, size_t, const frame_meta_data_before_acquire&, const  channel_settings&)
{
	qli_not_implemented();
}

bool camera_device::capture_burst_internal(const std::pair<std::vector<capture_item>::const_iterator, std::vector<capture_item>::const_iterator>&, const frame_meta_data_before_acquire&, const std::chrono::microseconds&, const std::chrono::microseconds&, const camera_frame_processing_function&)
{
	qli_not_implemented();
}