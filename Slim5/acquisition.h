#pragma once
#ifndef ACQUISITION_H
#define ACQUISITION_H
#include <set>
#include "capture_item.h"
#include "channel_settings.h"
#include "compact_light_path.h"
#include "filename_grouping_mode.h"

enum class channel_switching_order { switch_channel_per_roi, switch_channel_per_grid, switch_channel_per_tile, switch_channel_per_row, switch_channel_per_z };
enum class calibration_study_kind { gray_level_matching, qdic_shear, take_one };
typedef std::unordered_map<calibration_study_kind, std::string>  calibration_study_settings_map;
extern const calibration_study_settings_map calibration_study_settings;

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(calibration_study_kind)
Q_DECLARE_METATYPE(channel_switching_order)

#endif

struct acquisition final
{
	typedef std::set<int> used_channels;
	static std::pair< used_channels, acquisition> generate_take_one_sequence(const channel_settings& current_live_settings, const microscope_state& state);
	static std::pair< used_channels, acquisition> generate_qdic_shear_sequence(const channel_settings& current_live_settings, const microscope_state& state);
	static std::pair< used_channels, acquisition> generate_gray_level_sequence(const channel_settings& current_live_settings, const microscope_state& state);

	acquisition() noexcept: start_idx(0), filename_grouping(filename_grouping_mode::same_folder)
	{

	}
	struct preflight_info
	{
		bool pass;
		size_t resume_idx;
		preflight_info() noexcept:preflight_info(false, 0) {}
		preflight_info(const bool pass, const size_t resume_idx) noexcept: pass(pass), resume_idx(resume_idx) {}
	};
	std::string output_dir;//change all paths to actual paths
	std::vector<channel_settings> ch;
	std::vector<capture_item> cap;
	size_t start_idx;
	filename_grouping_mode filename_grouping;
	[[nodiscard]] microscope_move_action get_microscope_move_action(size_t event_idx) const;
	static microscope_move_action get_microscope_move_action(const channel_settings& channel, const capture_item& loc) noexcept;
	[[nodiscard]] size_t number_of_events() const noexcept;
	[[nodiscard]] size_t total_patterns(bool include_af) const;
	[[nodiscard]] std::pair<int, std::chrono::microseconds> get_next_pattern_and_stability(size_t current_event_idx, int current_pattern) const;
	void clear() noexcept;
	[[nodiscard]] bool is_valid_for_burst() const noexcept; //const;
	[[nodiscard]] bool is_valid_for_hardware_trigger() const noexcept;
	typedef std::function < QString()> preflight_function; //empty string on success
	[[nodiscard]] static bool prompt_if_failure(const QString& message);
	[[nodiscard]] preflight_info preflight_checks(const used_channels& channels_used) const;
	[[nodiscard]] bool settings_have_changed_for_this_event(size_t idx) const;
	[[nodiscard]] bool is_af_transition(size_t idx) const;
	[[nodiscard]] bool is_valid() const noexcept;
	void assert_valid() const;
	constexpr static int large_capture_threshold = 50000;
};

struct io_settings
{
	bool io_show_files, io_show_cmd_progress;
	explicit io_settings(const bool show_files, const bool show_cmd_progress) noexcept: io_show_files(show_files), io_show_cmd_progress(show_cmd_progress)
	{

	}
	io_settings() noexcept: io_settings(true, true)
	{

	}

	[[nodiscard]] bool write(const std::string& filename) const;
	explicit io_settings(const std::string& filename);
};

#endif