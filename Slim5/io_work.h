#pragma once
#ifndef IO_WORK_H
#define IO_WORK_H
#include "camera_frame.h"
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
#include "capture_item.h"
#include "filename_grouping_mode.h"
#include "gui_message_kind.h"
struct raw_io_work_meta_data : roi_name
{
	size_t progress_id;
	processing_double processing_info;
	bool force_sixteen_bit;
	std::string custom_label;
	filename_grouping_mode filename_grouping;
	typedef std::unordered_map<file_kind, std::string> file_extension_map;
	const static file_extension_map extensions;
	[[nodiscard]] static std::experimental::filesystem::path get_full_path(filename_grouping_mode grouping, const roi_name& roi_name, const std::string& custom_label, const processing_double& processing_double, const cycle_position& position, file_kind kind, int channel_route_index, const std::experimental::filesystem::path& dir = "");
	static void ensure_directories_exist(const std::experimental::filesystem::path& full_path);

	struct serialized_name
	{
		roi_name name;
		int channel_route_index;
		bool success;
		
	};
	[[nodiscard]] static serialized_name filepath_to_type(const std::string& full_path);
	static void write_file_convention(const std::string& dir);
	[[nodiscard]] static std::string filename_convention_header(char separator);

	raw_io_work_meta_data(const roi_name& name, const size_t progress_id, const processing_double& processing_double, const bool force_sixteen_bit, const std::string& custom_label, const filename_grouping_mode filename_grouping) : roi_name(name), progress_id(progress_id), processing_info(processing_double), force_sixteen_bit(force_sixteen_bit), custom_label(custom_label), filename_grouping(filename_grouping) {}

	raw_io_work_meta_data() noexcept: raw_io_work_meta_data(roi_name(), 0, { phase_retrieval::camera, phase_processing::raw_frames }, false,  std::string(), filename_grouping_mode::same_folder) {}

};
template <typename T>
struct raw_io_work final : camera_frame<T>, raw_io_work_meta_data
{
	raw_io_work(const camera_frame<T>& frame, const raw_io_work_meta_data& metadata, const gui_message_kind gui_message) 
		: camera_frame<T>(frame), 
		  raw_io_work_meta_data(metadata), 
		  gui_message(gui_message) {}
	raw_io_work() : raw_io_work(camera_frame<T>(), raw_io_work_meta_data(), gui_message_kind::none) {}

	[[nodiscard]] std::experimental::filesystem::path get_path(const file_kind kind, const std::experimental::filesystem::path& dir = "") const
	{

		// ReSharper disable All 
		return raw_io_work_meta_data::get_full_path(this->filename_grouping, *this, this->custom_label, this->processing_info, *this, kind, this->channel_route_index, dir);
		// ReSharper restore All
	}

	gui_message_kind gui_message;
};
extern std::ostream& write_capture_log_line_header(std::ostream& os);

template<typename T>
std::ostream& write_capture_log_line_header(std::ostream& os, const raw_io_work<T>& item, const std::chrono::microseconds& roi_delay);

// template specializations for function write_capture_log_line_header
template std::ostream& write_capture_log_line_header(std::ostream& os, const raw_io_work<float>& item, const std::chrono::microseconds& roi_delay);
template std::ostream& write_capture_log_line_header(std::ostream& os, const raw_io_work<unsigned short>& item, const std::chrono::microseconds& roi_delay);
#endif
