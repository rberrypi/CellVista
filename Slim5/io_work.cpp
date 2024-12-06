#include "stdafx.h"
#include "io_work.h"
#include <sstream> 
//#include <QFile>
#include <iostream>
#include <QFileInfo>
#include <QTextStream>
#include <boost/format.hpp>

#include "qli_runtime_error.h"

raw_io_work_meta_data::serialized_name raw_io_work_meta_data::filepath_to_type(const std::string& full_path)
{
	const QFile file(QString::fromStdString(full_path));
	const QFileInfo file_info(file.fileName());
	const auto filename = file_info.fileName();
	const auto filename_as_str = filename.toStdString();
	raw_io_work_meta_data::serialized_name r;

	// ReSharper disable once CppDeprecatedEntity
	const auto matches = sscanf_s(filename_as_str.c_str(), "f%d_t%d_i%d_ch%d_c%d_r%d_z%d", &r.name.roi, &r.name.time, &r.name.repeat, &r.channel_route_index, &r.name.column, &r.name.row, &r.name.page);
	r.success = matches == 7;
	return r;
}

std::string raw_io_work_meta_data::filename_convention_header(const char separator)
{
	// ReSharper disable once StringLiteralTypo
	const auto file_format = "%s,%s,%s,%s,%s,%s,%s,%s";
	auto substitute = QString::asprintf(file_format, "ROI", "TIME", "ITERATION", "CHANNEL", "COLUMN", "ROW", "ZSLICE", "MODULATION");
	if (separator != ',')
	{
		substitute.replace(',', separator);
	}
	return substitute.toStdString();
}

const raw_io_work_meta_data::file_extension_map raw_io_work_meta_data::extensions = { { file_kind::image, ".tif" },{ file_kind::roi, ".zip" },{ file_kind::stats, ".csv" } };

std::experimental::filesystem::path raw_io_work_meta_data::get_full_path(const filename_grouping_mode grouping, const roi_name& roi_name, const std::string& custom_label, const processing_double& processing_double, const cycle_position& position, const file_kind kind, const int channel_route_index, const std::experimental::filesystem::path& dir)
{
	const auto stub = [&]
	{
		const auto roi = roi_name.roi;
		const auto time = roi_name.time;
		const auto repeat = roi_name.repeat;
		const auto column = roi_name.column;
		const auto row = roi_name.row;
		const auto page = roi_name.page;

#if _DEBUG
		{
			if (custom_label.empty())
			{
				qli_invalid_arguments();
			}
		}
#endif
		std::stringstream ss;
		ss << "f" << roi;
		ss << "_t" << time;
		ss << "_i" << repeat;
		ss << "_ch" << channel_route_index;
		ss << "_c" << column;
		ss << "_r" << row;
		ss << "_z" << page;
		ss << "_m" << custom_label;
		const auto is_raw_frames = processing_double.is_raw_frame();
		if (is_raw_frames)
		{
			ss << std::to_string(position.pattern_idx);
		}
		ss << extensions.at(kind);
		auto str = ss.str();
		return str;
	}();

	switch (grouping)
	{
	case filename_grouping_mode::same_folder:
	{
		return dir / stub;
	}
	case filename_grouping_mode::fov_channel:
	{
		const auto fov = std::experimental::filesystem::path("f" + std::to_string(roi_name.roi));
		const auto channel = std::experimental::filesystem::path("ch" + std::to_string(channel_route_index));
		return dir / fov / channel / stub;
	}
	default:
		qli_not_implemented();
	}
}

void raw_io_work_meta_data::ensure_directories_exist(const std::experimental::filesystem::path& full_path)
{
	const auto directory = full_path.parent_path();
	if (std::experimental::filesystem::exists(directory))
	{
		return;
	}
	const auto directory_create_attempts = 5;
	auto directory_create_success = false;
	for (auto i = 0; i < directory_create_attempts; ++i)
	{
		directory_create_success = std::experimental::filesystem::create_directories(directory);
		if (directory_create_success)
		{
			return;
		}
	}
	if (!directory_create_success)
	{
		std::cout << "Warning failed to create directory " << directory << std::endl;
	}
}

std::ostream& write_capture_log_line_header(std::ostream& os)
{
	//maybe also size?
	os
		<< "Event ID (#)," << "Filename," << "Timestamp (ms),"
		<< "Stage X (um)," << "Stage Y (um)," << "Stage Z (um)," << "Microscope Channel (#)," << "NA Position (#)," << "NAc,"
		<< "Stage Delay (ms)," << "ROI Delay (ms),"
		<< "Denoise (#)," << "Pattern (#),"
		<< "SLM Delay (us)," << "Exposure (ms),"
		<< "Event,"
		<< "Action";
	return os;
}

template<typename T>
std::ostream& write_capture_log_line_header(std::ostream& os, const raw_io_work<T>& item, const std::chrono::microseconds& roi_delay)
{
	os << boost::format("%010d,%s,%014.2f,") % item.progress_id % item.get_path(file_kind::image) % to_mili(item.timestamp);
	os << boost::format("%010.4f,%010.4f,%010.4f,%02d,%02d,%010.4f,") % item.x % item.y % item.z % item.scope_channel % item.nac_position % item.nac;
	os << boost::format("%014.2f,%014.2f,") % to_mili(item.stage_move_delay) % to_mili(roi_delay);
	os << boost::format("%03d,%03d,") % item.denoise_idx % item.pattern_idx;
	os << boost::format("%014.2f,%014.2f,") % to_mili(item.slm_stability) % to_mili(item.exposure_time);
	os << boost::format("%d,") % static_cast<int>(item.gui_message);
	os << boost::format("%d") % static_cast<int>(item.action) << std::endl;
	return os;
}