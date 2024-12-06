#pragma once
#ifndef TRAKEM2_STITCHING_STRUCTS_H
#define TRAKEM2_STITCHING_STRUCTS_H
#include "capture_item.h"
#include "frame_size.h"
#include "display_settings.h"
#include <boost/container/small_vector.hpp>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
struct acquisition;

struct trakem2_fixed_attribute
{
	std::string name, value;
};

typedef boost::container::small_vector<trakem2_fixed_attribute, 10> trakem2_fixed_attributes;

struct trakem2_oid
{
	size_t oid;

	void set_oid()
	{
		oid = get_oid();
	}
	static size_t get_oid() noexcept
	{
		return ++oid_counter;
	}
	static void reset_oid() noexcept
	{
		oid_counter = 0;
	}
	trakem2_oid() noexcept : oid(0)
	{

	}
private:
	static size_t oid_counter;
};

struct t2_calibration_settings
{
	/*
		<t2_calibration
			pixelWidth="1.0"
			pixelHeight="1.0"
			pixelDepth="1.0"
			xOrigin="0.0"
			yOrigin="0.0"
			zOrigin="0.0"
			info="null"
			valueUnit="Gray Value"
			timeUnit="sec"
			unit="pixel"
		/>
	 */
	const static trakem2_fixed_attributes t2_calibration_settings_attributes;
};


struct trakem2_xy
{
	float x, y;

	trakem2_xy operator+(const trakem2_xy& pair) const
	{
		return trakem2_xy(x + pair.x, y + pair.y);
	}
	trakem2_xy operator-(const trakem2_xy& pair) const
	{
		return trakem2_xy(x - pair.x, y - pair.y);
	}
	trakem2_xy operator*(const float s) const
	{
		return trakem2_xy(x * s, y * s);
	}
	trakem2_xy operator/(const float s) const
	{
		return trakem2_xy(x / s, y / s);
	}
	bool operator==(const trakem2_xy& pair) const noexcept
	{
		return pair.x == x && pair.y == y;
	}

	[[nodiscard]] bool approx_equal(const trakem2_xy& pair) const noexcept
	{
		return approx_equals(pair.x, x) && approx_equals(pair.y, y);
	}
	trakem2_xy(const float x, const float y) noexcept : x(x), y(y) {};
	trakem2_xy() noexcept : trakem2_xy(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()) {}

	[[nodiscard]] bool is_valid() const noexcept
	{
		return std::isfinite(x) && std::isfinite(y);
	}
};

struct t2_patch_settings : trakem2_oid, trakem2_xy, frame_size
{
	/*
	<t2_patch
		oid = "8"
		width = "528.0"
		height = "512.0"
		transform = "matrix(1.0,0.0,0.0,1.0,0.0,0.0)"
		title = "f0_t0_i0_ch0_c0_r0_z0_mPhase.tif"
		links = ""
		type = "1"
		file_path = "f0_t0_i0_ch0_c0_r0_z0_mPhase.tif"
		style = "fill-opacity:1.0;stroke:#ffff00;"
		o_width = "528"
		o_height = "512"
		min = "1341.0"
		max = "24407.0"
		mres = "32"
	>
	*/
	t2_patch_settings(const frame_size& frame_size, const trakem2_xy& trakem2_xy, const std::experimental::filesystem::path& file_path, const display_range& range) : trakem2_oid(), trakem2_xy(trakem2_xy), frame_size(frame_size), file_path(file_path), range(range)
	{

	}
	std::experimental::filesystem::path file_path;
	display_range range;
	[[nodiscard]] trakem2_fixed_attributes t2_patch_settings_attributes(bool is_alignment) const;
	[[nodiscard]] bool is_valid() const;
};

struct t2_layer : trakem2_oid
{
	/*
		<t2_layer oid="5"
			 thickness="1.0"
			 z="0.0"
			 title=""
		>
	 */
	[[nodiscard]] bool is_valid() const;
	std::vector<t2_patch_settings> t2_patch_list;
	[[nodiscard]] trakem2_fixed_attributes t2_layer_attributes(int time_idx) const;
};

struct trakem2_tile_limits
{
	float min_x, max_x, min_y, max_y;
	trakem2_tile_limits();
	void insert_new_tile(const trakem2_xy& position) noexcept
	{
		min_x = std::min(min_x, position.x);
		min_y = std::min(min_y, position.y);
		max_x = std::max(max_x, position.x);
		max_y = std::max(max_y, position.y);
	}

	[[nodiscard]] frame_size get_size(const frame_size& largest_tile) const noexcept
	{
		return frame_size((max_x - min_x) + largest_tile.width, (max_y - min_y) + largest_tile.height);
	}
};
struct t2_layer_set : trakem2_oid
{
	/*
		<t2_layer_set
		oid="3"
		width="20.0"
		height="20.0"
		transform="matrix(1.0,0.0,0.0,1.0,0.0,0.0)"
		title="Top Level"
		links=""
		layer_width="1077.0"
		layer_height="992.0"
		rot_x="0.0"
		rot_y="0.0"
		rot_z="0.0"
		snapshots_quality="true"
		snapshots_mode="Full"
		color_cues="true"
		area_color_cues="true"
		avoid_color_cue_colors="false"
		n_layers_color_cue="0"
		paint_arrows="true"
		paint_tags="true"
		paint_edge_confidence_boxes="true"
		prepaint="false"
		preload_ahead="0"
	>
	 */
	frame_size layer_dimensions;//not sure what this is used for
	[[nodiscard]] trakem2_fixed_attributes t2_layer_set_attributes() const;
	t2_calibration_settings t2_calibration;
	typedef std::vector<t2_layer> t2_layer_map;
	t2_layer_map t2_layers;
};

struct project_settings
{
	/*
		<project
		id="0"
		title="test_sets.xml"
		unuid="1581127562980.1078193900.74353008"
		mipmaps_folder="trakem2.1581127562980.1078193900.74353008/trakem2.mipmaps/"
		storage_folder=""
		mipmaps_format="4"
		image_resizing_mode="Area downsampling"
		first_mipmap_level_saved="0"
	>
	 */
	std::string mipmaps_folder;
	[[nodiscard]] trakem2_fixed_attributes project_settings_attributes() const;
};

struct calibration_info
{
	trakem2_xy calibration_steps_in_stage_microns;
	float calibration_pixel_ratio;
	calibration_info(const float x, const float y, const float calibration_pixel_ratio) noexcept : calibration_steps_in_stage_microns({ x,y }), calibration_pixel_ratio(calibration_pixel_ratio) {}
	calibration_info() noexcept : calibration_info(0, 0, 0)
	{

	}

	[[nodiscard]] bool has_calibration_info() const
	{
		const auto blank = calibration_info();
		return calibration_steps_in_stage_microns.x != blank.calibration_steps_in_stage_microns.x || calibration_steps_in_stage_microns.y != blank.calibration_steps_in_stage_microns.y;
	}

	[[nodiscard]] bool item_approx_equals(const calibration_info& b) const
	{
		return calibration_steps_in_stage_microns.approx_equal(b.calibration_steps_in_stage_microns) && approx_equals(calibration_pixel_ratio, b.calibration_pixel_ratio);
	}
};

struct trakem2_file : project_settings, t2_layer_set, calibration_info
{
	[[nodiscard]] bool is_valid() const;
	void finalize_file(const capture_item_2dt_hash& hash);
	std::string label;
	[[nodiscard]] std::string get_filename(const capture_item_2dt_hash& hash) const;
	explicit trakem2_file(const std::string& label) : label(label) {};
	static calibration_info calibration_info_from_filename(const std::string& full_filepath, bool& success);
private:
	static std::string get_filename(const capture_item_2dt_hash& hash, const std::string& label, const calibration_info& calibration_info);
};

struct trakem2_acquisition
{
	// would be good to time the performance of unordered map
	std::map<capture_item_2dt_hash, trakem2_file> trakem2_files;
};

struct trakem2_stage_coordinate_to_pixel_mapper
{
	struct abcd
	{
		float a, b, c, d;
		abcd(const float a, const float b, const float c, const float d) noexcept : a(a), b(b), c(c), d(d)
		{

		}

		[[nodiscard]] trakem2_xy apply_mapper(const trakem2_xy& input_vector) const
		{
			const auto u = a * input_vector.x + b * input_vector.y;
			const auto v = c * input_vector.x + d * input_vector.y;
			return trakem2_xy(u, v);
		}
	};
	struct  displacement_vectors
	{
		calibration_info calibration_in_stage_microns;
		trakem2_xy column_step_in_pixels, row_step_in_pixels;
		displacement_vectors() noexcept : displacement_vectors({ 0,0 }, { 0,0 }, calibration_info()) {};
		displacement_vectors(const trakem2_xy& column_step_in_pixels, const trakem2_xy& row_step_in_pixels, const calibration_info& calibration) noexcept : calibration_in_stage_microns(calibration), column_step_in_pixels(column_step_in_pixels), row_step_in_pixels(row_step_in_pixels) {}

		[[nodiscard]] bool is_valid() const
		{
			return calibration_in_stage_microns.has_calibration_info() && column_step_in_pixels.is_valid() && row_step_in_pixels.is_valid();
		}

		[[nodiscard]] bool approx_equal(const displacement_vectors& b) const
		{
			return column_step_in_pixels.approx_equal(b.column_step_in_pixels) && row_step_in_pixels.approx_equal(b.row_step_in_pixels) && calibration_in_stage_microns.item_approx_equals(b.calibration_in_stage_microns);
		}
	};
	displacement_vectors settings;
	[[nodiscard]] trakem2_xy map(const trakem2_xy& xy_stage_coordinates, float current_ratio) const;
	explicit trakem2_stage_coordinate_to_pixel_mapper(const displacement_vectors& settings) noexcept : settings(settings) {}
	static trakem2_stage_coordinate_to_pixel_mapper get_pass_through_mapper(float pixel_ratio);

};

struct trakem2_processor
{
	static trakem2_acquisition acquisition_to_trakem2(const acquisition& acquisition, const trakem2_stage_coordinate_to_pixel_mapper& mapper, float current_pixel_ratio, const calibration_info& calibration = calibration_info());
	static void write_trakem2(const trakem2_acquisition& acquisition, const std::string& basedir);
	static trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors get_vectors_from_xml_file(const std::string& full_path);
private:
	static std::string get_trakem2_metadata();

};

#endif
