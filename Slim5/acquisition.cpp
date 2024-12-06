#include "stdafx.h"
#include "camera_device.h"
#include "acquisition.h"
#include <fstream>
#include <cereal/archives/json.hpp>

#include "device_factory.h"
#include "qli_runtime_error.h"

const calibration_study_settings_map calibration_study_settings = {
	{calibration_study_kind::gray_level_matching,"Gray Level Matching"},
	{calibration_study_kind::qdic_shear,"QDIC Shear"},
	{calibration_study_kind::take_one,"Acquire One Image"} };

bool acquisition::is_valid_for_hardware_trigger() const noexcept
{
	//todo actually implement
	return is_valid_for_burst();
}

bool acquisition::is_valid_for_burst() const noexcept
{
	//Add more criteria
	const auto capture_item_functor = [](const capture_item& x)
	{
		return x.action == scope_action::capture;
	};
	const auto all_capture_actions = std::all_of(cap.begin(), cap.end(), capture_item_functor);
	const auto channel_item_functor = [](const channel_settings& x)
	{
		return D->cameras.at(x.camera_idx)->has_burst_mode;
	};
	const auto all_channels = std::all_of(ch.begin(), ch.end(), channel_item_functor);
	return all_capture_actions && all_channels;
}

bool acquisition::is_valid() const noexcept
{
	const auto has_items = !ch.empty() && !cap.empty();
	const auto all_valid = std::all_of(ch.begin(), ch.end(), [](const auto ch)
	{
		return ch.is_valid();
	});
	return has_items && all_valid && cap.size() > start_idx && !output_dir.empty();
}

microscope_move_action acquisition::get_microscope_move_action(const size_t event_idx) const
{
	const auto& current_event = cap.at(event_idx);
	const auto& channel = ch.at(current_event.channel_route_index);
	return get_microscope_move_action(channel, current_event);
}

microscope_move_action acquisition::get_microscope_move_action(const channel_settings& channel, const capture_item& loc) noexcept
{
	const auto state = microscope_state(loc, channel);
	const auto move_action = microscope_move_action(state, loc.stage_move_delay);
	return move_action;
}

size_t acquisition::number_of_events() const noexcept
{
	return cap.size();
}

std::pair<int, std::chrono::microseconds> acquisition::get_next_pattern_and_stability(const size_t current_event_idx, const int current_pattern) const
{

	auto pear = std::make_pair(-1, ms_to_chrono(0));
	const auto channel_idx = cap.at(current_event_idx).channel_route_index;
	const auto& channel = ch.at(channel_idx);
	const auto patterns = channel.iterator().cycle_limit.pattern_idx;
	const auto is_terminal_pattern = current_pattern == patterns - 1;
	if (!is_terminal_pattern)
	{
		auto next_pattern = current_pattern + 1;
		const auto modulo_pattern = next_pattern % channel.exposures_and_delays.size();
		auto stability_time = channel.exposures_and_delays.at(modulo_pattern).slm_stability;
		pear = std::make_pair(next_pattern, stability_time);
	}
	else
	{
		const auto next_idx = current_event_idx + 1;
		if (next_idx < cap.size())
		{
			const auto channel_idx_nxt = cap.at(current_event_idx).channel_route_index;
			const auto& channel_nxt = ch.at(channel_idx_nxt);
			auto first_pattern = 0;
			auto stability_time = channel_nxt.exposures_and_delays.at(first_pattern).slm_stability;
			pear = std::make_pair(first_pattern, stability_time);
		}
	}
	return pear;
}

size_t acquisition::total_patterns(bool include_af) const
{
	// build a look up table to make this go faster?
	std::vector<int> channel_to_pattern(ch.size());
	std::transform(ch.begin(), ch.end(), channel_to_pattern.begin(), [](const channel_settings& chan) {return chan.iterator().frame_count(); });
	const auto sum_function = [channel_to_pattern, include_af](const size_t result, const capture_item& item)
	{
		const auto acquisition_items = include_af || item.action == scope_action::capture ? channel_to_pattern.at(item.channel_route_index) : 0;
		return result + acquisition_items;
	};
	const auto patterns = std::accumulate(cap.begin(), cap.end(), static_cast<size_t>(0), sum_function);
	return patterns;
}

void acquisition::clear() noexcept
{
	ch.clear();
	cap.clear();
	this->filename_grouping = filename_grouping_mode::same_folder;
}

bool acquisition::settings_have_changed_for_this_event(const size_t idx) const
{
	if (idx < 1)
	{
		return true;
	}
	const auto previous_idx = idx - 1;
	const auto& current = cap.at(idx);
	const auto& previous = cap.at(previous_idx);
	//note that any camera change also changes the channel!
	const auto channel_change = current.channel_route_index != previous.channel_route_index;
	return channel_change;
}

bool acquisition::is_af_transition(const size_t idx) const
{
	const auto next_idx = idx + 1;
	if (next_idx >= cap.size())
	{
		return false;
	}
	const auto& current = cap.at(idx);
	const auto& next = cap.at(next_idx);
	//AKA wait for AF to populate, finish in async mode
	const auto current_is_af = current.action == scope_action::focus;
	const auto next_is_capture = next.action == scope_action::capture;
	return current_is_af && next_is_capture;
}

void acquisition::assert_valid() const
{
#if _DEBUG
	if (!is_valid())
	{
		qli_runtime_error("Invalid Acquisition");
	}
#endif
}

template <class Archive>
void serialize(Archive& archive, io_settings& cc)
{
	archive(
		cereal::make_nvp("io_show_files", cc.io_show_files),
		cereal::make_nvp("io_show_cmd_progress", cc.io_show_cmd_progress)
	);
}

bool io_settings::write(const std::string& filename) const
{
	std::ofstream os(filename);
	if (os.is_open())
	{
		cereal::JSONOutputArchive archive(os);
		archive(*this);
		return true;
	}
	return false;
}

io_settings::io_settings(const std::string& filename)
{
	io_show_files = true;
	io_show_cmd_progress = true;
	try
	{
		std::ifstream configuration_file(filename);
		cereal::JSONInputArchive archive(configuration_file);
		archive(*this);
	}
	catch (...)
	{
		*this = io_settings();
	}
}

std::pair< acquisition::used_channels, acquisition> acquisition::generate_take_one_sequence(const channel_settings& current_live_settings, const microscope_state& state)
{
	acquisition new_acquisition;
	capture_item capture_item;
	static_cast<scope_location_xyz&>(capture_item) = state;
	new_acquisition.cap.push_back(capture_item);
	new_acquisition.ch.push_back(current_live_settings);
	new_acquisition.ch.front().processing = phase_processing::raw_frames;
	acquisition::used_channels channels = { 0 };
	return std::make_pair(channels, new_acquisition);
}

std::pair< acquisition::used_channels, acquisition> acquisition::generate_qdic_shear_sequence(const channel_settings& current_live_settings, const microscope_state& state)
{
	acquisition::used_channels channels;
	auto channel_settings = current_live_settings;
	acquisition new_acquisition;
	static_cast<processing_quad&>(channel_settings) = processing_quad(phase_retrieval::glim_demux, phase_processing::phase, current_live_settings.demosaic, denoise_mode::off);
	channel_settings.fixup_channel();
	for (auto i = 0; i < 360; ++i)
	{
		const auto name = roi_name(i, 0, 0, 0, 0, 0);
		const capture_item capture_item(name, scope_delays(), state, i);
		new_acquisition.cap.push_back(capture_item);
		channel_settings.qsb_qdic_shear_angle = i;
		new_acquisition.ch.push_back(channel_settings);
		channels.insert(i);
	}
	return std::make_pair(channels, new_acquisition);
}

std::pair< acquisition::used_channels, acquisition>  acquisition::generate_gray_level_sequence(const channel_settings& current_live_settings, const microscope_state& state)
{
	acquisition new_acquisition;
	capture_item capture_item;
	static_cast<scope_location_xyz&>(capture_item) = state;
	new_acquisition.cap.push_back(capture_item);
	const processing_quad quad(phase_retrieval::custom_patterns, phase_processing::raw_frames, current_live_settings.demosaic, denoise_mode::off);
	auto channel = channel_settings::generate_test_channel(quad);
	channel.exposures_and_delays.front().slm_stability = ms_to_chrono(2);//two seconds for first frame to stabilize
	if (channel.modulator_settings.size() == 2)
	{
		//todo use enum lookup
		const auto& illuminator_settings = current_live_settings.modulator_settings.at(per_modulator_saveable_settings::illumination_idx);
		const auto max_value = illuminator_settings.illumination_power * illuminator_settings.brightfield_scale_factor;
		auto& illuminator_settings_new = channel.modulator_settings.at(per_modulator_saveable_settings::illumination_idx);
		for (auto& pattern : illuminator_settings_new.patterns)
		{
			pattern.slm_background = max_value;
			pattern.slm_value = max_value;
			pattern.pattern_mode = slm_pattern_mode::checkerboard;
		}
	}
	static_cast<microscope_light_path&>(channel) = state;
	const auto first_setting = current_live_settings.exposures_and_delays.front();
	std::fill(channel.exposures_and_delays.begin(), channel.exposures_and_delays.end(), first_setting);
	new_acquisition.ch.push_back(channel);
	acquisition::used_channels channels = { 0 };
	return std::make_pair(channels, new_acquisition);
}