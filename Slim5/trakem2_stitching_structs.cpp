#include "stdafx.h"
#include "trakem2_stitching_structs.h"
#include "acquisition.h"
#include "io_work.h"
#include <tinyxml2/tinyxml2.h>
#include <QDir>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
#include <iostream>

#include "qli_runtime_error.h"
size_t trakem2_oid::oid_counter = 0;

std::string trakem2_file::get_filename(const capture_item_2dt_hash& hash) const
{
	return trakem2_file::get_filename(hash, label, *this);
}

calibration_info trakem2_file::calibration_info_from_filename(const std::string& full_filepath, bool& success)
{
	const std::experimental::filesystem::path fs(full_filepath);
	const auto filename = fs.stem().string();
	const auto expected_items = 8;
	std::vector<std::string> results(expected_items);
	boost::split(results, filename, [](const char c) {return c == '_'; });
	if (results.size() == expected_items)
	{
		float y, x, calibration_pixel_ratio;
		const auto x_success = sscanf_s(results.at(5).c_str(), "x%f", &x);
		const auto y_success = sscanf_s(results.at(6).c_str(), "y%f", &y);
		const auto p_success = sscanf_s(results.at(7).c_str(), "p%f", &calibration_pixel_ratio);
		if (x_success == 1 && y_success == 1 && p_success)
		{
			success = true;
			const calibration_info info(x, y, calibration_pixel_ratio);
			return info;
		}
	}
	success = false;
	return calibration_info();
}

std::string trakem2_file::get_filename(const capture_item_2dt_hash& hash, const std::string& label, const calibration_info& calibration_info)
{
	//f0_t0_i0_ch0_c0_r0_z0_mPhase.tif
	const auto has_calibration = calibration_info.has_calibration_info();
	if (has_calibration)
	{
		return boost::str(boost::format("f%1%_i%2%_ch%3%_z%4%_m%5%_x%6%_y%7%_p%8%.xml") % hash.roi % hash.repeat % hash.channel_route_index % hash.page % label % calibration_info.calibration_steps_in_stage_microns.x % calibration_info.calibration_steps_in_stage_microns.y % calibration_info.calibration_pixel_ratio);
	}
	return boost::str(boost::format("f%1%_i%2%_ch%3%_z%4%_m%5%.xml") % hash.roi % hash.repeat % hash.channel_route_index % hash.page % label);
}

bool t2_patch_settings::is_valid() const
{
	return n() > 0 && !file_path.empty();
}

bool t2_layer::is_valid() const
{
	if (t2_patch_list.empty())
	{
		return false;
	}
	for (const auto& patch : t2_patch_list)
	{
		if (!patch.is_valid())
		{
			return false;
		}
	}
	return true;
}

bool trakem2_file::is_valid() const
{
	const auto project_is_valid = !mipmaps_folder.empty() && layer_dimensions.is_valid() && !t2_layers.empty();
	if (project_is_valid)
	{
		for (const auto& layer : t2_layers)
		{
			if (!layer.is_valid())
			{
				return false;
			}
		}
	}
	return true;
}


bool capture_item_2dt_hash::operator<(const capture_item_2dt_hash& ob) const
{
	if (roi < ob.roi)
	{
		return true;
	}
	if (roi == ob.roi)
	{
		if (repeat < ob.repeat)
		{
			return true;
		}
		if (repeat == ob.repeat)
		{
			if (channel_route_index < ob.channel_route_index)
			{
				return true;
			}
			if (channel_route_index == ob.channel_route_index)
			{
				if (page < ob.page)
				{
					return true;
				}
				if (page == ob.page)
				{
					return pattern_idx < ob.pattern_idx;
				}
			}
		}
	}
	return false;
}

bool capture_item_within_layer_hash::operator<(const capture_item_within_layer_hash& ob) const
{
	if (row < ob.row)
	{
		return true;
	}
	if (row == ob.row)
	{
		if (column < ob.column)
		{
			return true;
		}
		if (column == ob.column)
		{
			if (time < ob.time)
			{
				return true;
			}
		}
	}
	return false;
}

trakem2_tile_limits::trakem2_tile_limits() : min_x(std::numeric_limits<float>::max()), max_x(std::numeric_limits<float>::min()), min_y(std::numeric_limits<float>::max()), max_y(std::numeric_limits<float>::min())
{

}

void trakem2_file::finalize_file(const capture_item_2dt_hash& hash)
{
	trakem2_oid::reset_oid();
	trakem2_tile_limits limits;
	for (auto& layer : t2_layers)
	{
		layer.set_oid();
		for (auto& patch : layer.t2_patch_list)
		{
			patch.set_oid();
			limits.insert_new_tile(patch);
		}
	}
	//apply dimensions
	const auto& first_tile = t2_layers.front().t2_patch_list.front();
	layer_dimensions = limits.get_size(first_tile);
	for (auto& layer : t2_layers)
	{
		for (auto& patch : layer.t2_patch_list)
		{
			patch.x = patch.x - limits.min_x;
			patch.y = patch.y - limits.min_y;
		}
	}
	mipmaps_folder = get_filename(hash) + ".mipmaps";
}

trakem2_acquisition trakem2_processor::acquisition_to_trakem2(const acquisition& acquisition, const trakem2_stage_coordinate_to_pixel_mapper& mapper, float current_pixel_ratio, const calibration_info& calibration)
{
#if _DEBUG
	{
		if (!acquisition.is_valid())
		{
			qli_invalid_arguments();
		}
	}
#endif
	trakem2_acquisition trakem2;
	auto& files = trakem2.trakem2_files;
	for (const auto& item : acquisition.cap)
	{
		if (item.action != scope_action::capture)
		{
			continue;
		}
		const auto channel_route_index = item.channel_route_index;
		const auto& channel = acquisition.ch.at(channel_route_index);
		const auto pattern_limit = channel.iterator().cycle_limit.pattern_idx;;
		const auto skip_computed_image_patterns = [&](const int pattern_idx)
		{
			return  !(pattern_idx > 0 && channel.processing != phase_processing::raw_frames);
		};
		for (auto pattern_idx = 0; pattern_idx < pattern_limit && skip_computed_image_patterns(pattern_idx); ++pattern_idx)
		{
			const auto file_hash = item.get_2dt_hash(pattern_idx);
			auto file_idx = files.find(file_hash);
			if (file_idx == files.end())
			{
				const trakem2_file blank_file(channel.label_suffix);
				const auto info = files.insert({ file_hash,blank_file });
				file_idx = info.first;
			}
			auto& layer_set = file_idx->second.t2_layers;
			if (item.time >= layer_set.size())
			{
				const t2_layer blank_layer;
				layer_set.insert(layer_set.begin() + item.time, blank_layer);
			}
			auto& layer = layer_set.at(item.time);

			const auto& size = channel.image_info_per_capture_item_on_disk();
			const auto new_coordinates = mapper.map({ item.x,item.y }, current_pixel_ratio);
			auto first_color_display_range = channel.ranges.front();
			const cycle_position cycle_position(0, pattern_idx);
			const auto file_path = raw_io_work_meta_data::get_full_path(acquisition.filename_grouping, item, channel.label_suffix, channel, cycle_position, file_kind::image, channel_route_index);
			t2_patch_settings patch_settings(size, new_coordinates, file_path, first_color_display_range);
			layer.t2_patch_list.push_back(patch_settings);
		}

	}
	for (auto& item : trakem2.trakem2_files)
	{
		static_cast<calibration_info&>(item.second) = calibration;
		item.second.finalize_file(item.first);
	}
#if _DEBUG
	for (const auto& item : trakem2.trakem2_files)
	{
		const auto& file = item.second;
		if (!file.is_valid())
		{
			qli_runtime_error("Oh Nope");
		}
	}
#endif
	return trakem2;
}


std::string trakem2_processor::get_trakem2_metadata()
{
	const static std::string metadata_array =
		// ReSharper disable StringLiteralTypo
		"DOCTYPE trakem2_anything [\n\
	<!ELEMENT trakem2 (project,t2_layer_set,t2_display)>\n\
	<!ELEMENT project (anything)>\n\
	<!ATTLIST project id NMTOKEN #REQUIRED>\n\
	<!ATTLIST project unuid NMTOKEN #REQUIRED>\n\
	<!ATTLIST project title NMTOKEN #REQUIRED>\n\
	<!ATTLIST project preprocessor NMTOKEN #REQUIRED>\n\
	<!ATTLIST project mipmaps_folder NMTOKEN #REQUIRED>\n\
	<!ATTLIST project storage_folder NMTOKEN #REQUIRED>\n\
	<!ELEMENT anything EMPTY>\n\
	<!ATTLIST anything id NMTOKEN #REQUIRED>\n\
	<!ATTLIST anything expanded NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_layer (t2_patch,t2_label,t2_layer_set,t2_profile)>\n\
	<!ATTLIST t2_layer oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer thickness NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer z NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_layer_set (t2_prop,t2_linked_prop,t2_annot,t2_layer,t2_pipe,t2_ball,t2_area_list,t2_calibration,t2_stack,t2_treeline)>\n\
	<!ATTLIST t2_layer_set oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set composite NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set layer_width NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set layer_height NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set rot_x NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set rot_y NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set rot_z NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set snapshots_quality NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set color_cues NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set area_color_cues NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set avoid_color_cue_colors NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set n_layers_color_cue NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set paint_arrows NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set paint_tags NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set paint_edge_confidence_boxes NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_layer_set preload_ahead NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_calibration EMPTY>\n\
	<!ATTLIST t2_calibration pixelWidth NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_calibration pixelHeight NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_calibration pixelDepth NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_calibration xOrigin NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_calibration yOrigin NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_calibration zOrigin NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_calibration info NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_calibration valueUnit NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_calibration timeUnit NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_calibration unit NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_ball (t2_prop,t2_linked_prop,t2_annot,t2_ball_ob)>\n\
	<!ATTLIST t2_ball oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball composite NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball fill NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_ball_ob EMPTY>\n\
	<!ATTLIST t2_ball_ob x NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball_ob y NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball_ob r NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_ball_ob layer_id NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_label (t2_prop,t2_linked_prop,t2_annot)>\n\
	<!ATTLIST t2_label oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_label layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_label transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_label style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_label locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_label visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_label title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_label links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_label composite NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_filter EMPTY>\n\
	<!ELEMENT t2_patch (t2_prop,t2_linked_prop,t2_annot,ict_transform,ict_transform_list,t2_filter)>\n\
	<!ATTLIST t2_patch oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch composite NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch file_path NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch original_path NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch type NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch false_color NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch ct NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch o_width NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch o_height NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch min NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch max NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch o_width NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch o_height NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch pps NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch mres NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch ct_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_patch alpha_mask_id NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_pipe (t2_prop,t2_linked_prop,t2_annot)>\n\
	<!ATTLIST t2_pipe oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe composite NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe d NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe p_width NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_pipe layer_ids NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_polyline (t2_prop,t2_linked_prop,t2_annot)>\n\
	<!ATTLIST t2_polyline oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_polyline layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_polyline transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_polyline style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_polyline locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_polyline visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_polyline title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_polyline links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_polyline composite NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_polyline d NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_profile (t2_prop,t2_linked_prop,t2_annot)>\n\
	<!ATTLIST t2_profile oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_profile layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_profile transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_profile style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_profile locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_profile visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_profile title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_profile links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_profile composite NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_profile d NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_area_list (t2_prop,t2_linked_prop,t2_annot,t2_area)>\n\
	<!ATTLIST t2_area_list oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_area_list layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_area_list transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_area_list style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_area_list locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_area_list visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_area_list title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_area_list links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_area_list composite NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_area_list fill_paint NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_area (t2_path)>\n\
	<!ATTLIST t2_area layer_id NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_path EMPTY>\n\
	<!ATTLIST t2_path d NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_dissector (t2_prop,t2_linked_prop,t2_annot,t2_dd_item)>\n\
	<!ATTLIST t2_dissector oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_dissector layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_dissector transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_dissector style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_dissector locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_dissector visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_dissector title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_dissector links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_dissector composite NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_dd_item EMPTY>\n\
	<!ATTLIST t2_dd_item radius NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_dd_item tag NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_dd_item points NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_stack (t2_prop,t2_linked_prop,t2_annot,(iict_transform| iict_transform_list)?)>\n\
	<!ATTLIST t2_stack oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_stack layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_stack transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_stack style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_stack locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_stack visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_stack title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_stack links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_stack composite NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_stack file_path CDATA #REQUIRED>\n\
	<!ATTLIST t2_stack depth CDATA #REQUIRED>\n\
	<!ELEMENT t2_tag EMPTY>\n\
	<!ATTLIST t2_tag name NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_tag key NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_node (t2_area*,t2_tag*)>\n\
	<!ATTLIST t2_node x NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_node y NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_node lid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_node c NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_node r NMTOKEN #IMPLIED>\n\
	<!ELEMENT t2_treeline (t2_node*,t2_prop,t2_linked_prop,t2_annot)>\n\
	<!ATTLIST t2_treeline oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_treeline layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_treeline transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_treeline style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_treeline locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_treeline visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_treeline title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_treeline links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_treeline composite NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_areatree (t2_node*,t2_prop,t2_linked_prop,t2_annot)>\n\
	<!ATTLIST t2_areatree oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_areatree layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_areatree transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_areatree style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_areatree locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_areatree visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_areatree title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_areatree links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_areatree composite NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_connector (t2_node*,t2_prop,t2_linked_prop,t2_annot)>\n\
	<!ATTLIST t2_connector oid NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_connector layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_connector transform NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_connector style NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_connector locked NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_connector visible NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_connector title NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_connector links NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_connector composite NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_prop EMPTY>\n\
	<!ATTLIST t2_prop key NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_prop value NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_linked_prop EMPTY>\n\
	<!ATTLIST t2_linked_prop target_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_linked_prop key NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_linked_prop value NMTOKEN #REQUIRED>\n\
	<!ELEMENT t2_annot EMPTY>\n\
	<!ELEMENT t2_display EMPTY>\n\
	<!ATTLIST t2_display id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display layer_id NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display x NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display y NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display magnification NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display srcrect_x NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display srcrect_y NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display srcrect_width NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display srcrect_height NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display scroll_step NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display c_alphas NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display c_alphas_state NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display filter_enabled NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display filter_min_max_enabled NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display filter_min NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display filter_max NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display filter_invert NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display filter_clahe_enabled NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display filter_clahe_block_size NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display filter_clahe_histogram_bins NMTOKEN #REQUIRED>\n\
	<!ATTLIST t2_display filter_clahe_max_slope NMTOKEN #REQUIRED>\n\
	<!ELEMENT ict_transform EMPTY>\n\
	<!ATTLIST ict_transform class CDATA #REQUIRED>\n\
	<!ATTLIST ict_transform data CDATA #REQUIRED>\n\
	<!ELEMENT iict_transform EMPTY>\n\
	<!ATTLIST iict_transform class CDATA #REQUIRED>\n\
	<!ATTLIST iict_transform data CDATA #REQUIRED>\n\
	<!ELEMENT ict_transform_list (ict_transform|iict_transform)*>\n\
	<!ELEMENT iict_transform_list (iict_transform*)>\n\
	]";
	// ReSharper enable StringLiteralTypo
	return metadata_array;
}

const trakem2_fixed_attributes t2_calibration_settings::t2_calibration_settings_attributes = {
	{"pixelWidth","1.0"},	{"pixelHeight","1.0"},
	{"pixelDepth","1.0"},
	{"xOrigin","0.0"},{"yOrigin","0.0"},{"zOrigin","0.0"},
	{"info","null"},
	{"valueUnit","Gray Value"},
	{"timeUnit","sec"},
	{"unit","pixel"},
};

trakem2_fixed_attributes t2_patch_settings::t2_patch_settings_attributes(const bool is_alignment) const
{
	trakem2_fixed_attributes attributes;
	const auto file_path_str = file_path.u8string();
	attributes.push_back({ "oid", std::to_string(oid) });
	attributes.push_back({ "width", std::to_string(width) });
	attributes.push_back({ "height", std::to_string(height) });
	const auto transform_string = boost::str(boost::format("matrix(1.0,0.0,0.0,1.0,%1%,%2%)") % this->x % this->y);
	attributes.push_back({ "transform",transform_string });
	attributes.push_back({ "title", file_path_str });
	attributes.push_back({ "links", "" });
	attributes.push_back({ "type", "1" });
	attributes.push_back({ "file_path", file_path_str });
	attributes.push_back({ "style", is_alignment ? "fill-opacity:0.5;stroke:#ffff00;" : "fill-opacity:0.1;stroke:#ffff00;" });
	attributes.push_back({ "o_width", std::to_string(width) });
	attributes.push_back({ "o_height", std::to_string(height) });
	attributes.push_back({ "min", std::to_string(range.min) });
	attributes.push_back({ "max", std::to_string(range.max) });
	attributes.push_back({ "mres", std::to_string(32) });
	return attributes;
}

trakem2_fixed_attributes t2_layer::t2_layer_attributes(const int time_idx) const
{
	trakem2_fixed_attributes attributes;
	attributes.push_back({ "oid", std::to_string(oid) });
	attributes.push_back({ "thickness", "1.0" });
	attributes.push_back({ "z", std::to_string(time_idx) });
	attributes.push_back({ "title", "" });
	return attributes;
}

trakem2_fixed_attributes t2_layer_set::t2_layer_set_attributes() const
{
	trakem2_fixed_attributes attributes;
	attributes.push_back({ "oid", std::to_string(oid) });
	attributes.push_back({ "width", "20.0" });
	attributes.push_back({ "height", "20.0" });
	attributes.push_back({ "transform", "matrix(1.0,0.0,0.0,1.0,0.0,0.0)" });
	attributes.push_back({ "title", "Top Level" });
	attributes.push_back({ "links", "" });
	attributes.push_back({ "layer_width",std::to_string(layer_dimensions.width) });
	attributes.push_back({ "layer_height",std::to_string(layer_dimensions.height) });
	attributes.push_back({ "rot_x","0.0" });
	attributes.push_back({ "rot_y","0.0" });
	attributes.push_back({ "rot_z","0.0" });
	attributes.push_back({ "snapshots_quality","true" });
	attributes.push_back({ "snapshots_mode","Full" });
	attributes.push_back({ "color_cues","true" });
	attributes.push_back({ "area_color_cues","true" });
	attributes.push_back({ "avoid_color_cue_colors","false" });
	attributes.push_back({ "n_layers_color_cue","0" });
	attributes.push_back({ "paint_arrows","true" });
	attributes.push_back({ "paint_tags","true" });
	attributes.push_back({ "paint_edge_confidence_boxes","true" });
	attributes.push_back({ "prepaint","false" });
	attributes.push_back({ "preload_ahead","0" });
	return attributes;
}

trakem2_fixed_attributes project_settings::project_settings_attributes() const
{
	trakem2_fixed_attributes attributes;
	attributes.push_back({ "id", "0" });
	attributes.push_back({ "title", "QPI" });
	attributes.push_back({ "unuid", mipmaps_folder });//undocumented but oh well
	attributes.push_back({ "mipmaps_folder", mipmaps_folder });
	attributes.push_back({ "storage_folder", "" });
	attributes.push_back({ "mipmaps_format", "0" });//aka JPEGs
	attributes.push_back({ "image_resizing_mode", "Area downsampling" });
	attributes.push_back({ "first_mipmap_level_saved", "0" });
	return attributes;
}


void trakem2_processor::write_trakem2(const trakem2_acquisition& acquisition, const std::string& basedir)
{
	using namespace tinyxml2;

	const auto dir = QDir(QString::fromStdString(basedir));

	for (auto& file : acquisition.trakem2_files)
	{
		const auto& trakem_file_hash = file.first;
		auto& trakem2_item = file.second;
		auto& t2_layers = trakem2_item.t2_layers;

		XMLDocument output_document;

		//Add Trakem2 metadata
		XMLDeclaration* declaration = output_document.NewDeclaration(R"(xml version="1.0" encoding="ISO-8859-1")");
		output_document.InsertEndChild(declaration);
		output_document.InsertEndChild(output_document.NewUnknown(get_trakem2_metadata().c_str()));

		const auto fill_element_with_attributes = [](XMLElement* element, const trakem2_fixed_attributes& attributes)
		{
			for (const auto& item : attributes)
			{
				element->SetAttribute(item.name.c_str(), item.value.c_str());
			}
		};
		//Root
		XMLElement* root = output_document.NewElement("trakem2");
		output_document.InsertEndChild(root);
		//Project
		XMLElement* project_element = output_document.NewElement("project");
		fill_element_with_attributes(project_element, trakem2_item.project_settings_attributes());
		root->InsertEndChild(project_element);
		//Layer Set
		XMLElement* t2_layer_set_element = output_document.NewElement("t2_layer_set");
		fill_element_with_attributes(t2_layer_set_element, trakem2_item.t2_layer_set_attributes());
		root->InsertEndChild(t2_layer_set_element);
		//Calibration
		XMLElement* t2_calibration_element = output_document.NewElement("t2_calibration");
		fill_element_with_attributes(t2_calibration_element, t2_calibration_settings::t2_calibration_settings_attributes);
		t2_layer_set_element->InsertEndChild(t2_calibration_element);
		//Layers
		for (auto t_idx = 0; t_idx < t2_layers.size(); ++t_idx)
		{
			const auto& layer = t2_layers.at(t_idx);
			XMLElement* t2_layer_element = output_document.NewElement("t2_layer");
			fill_element_with_attributes(t2_layer_element, layer.t2_layer_attributes(t_idx));
			for (const t2_patch_settings& patch : layer.t2_patch_list)
			{
				XMLElement* patch_element = output_document.NewElement("t2_patch");
				const auto is_alignment = file.second.has_calibration_info();
				fill_element_with_attributes(patch_element, patch.t2_patch_settings_attributes(is_alignment));
				t2_layer_element->InsertEndChild(patch_element);
			}
			t2_layer_set_element->InsertEndChild(t2_layer_element);
		}

		// Save/Write XML files
		const auto output_filename = trakem2_item.get_filename(trakem_file_hash);
		const auto filepath_xml = dir.filePath(QString::fromStdString(output_filename)).toStdString();
		const auto save_valid = output_document.SaveFile(filepath_xml.c_str());
#if _DEBUG
		if (save_valid != XML_SUCCESS)
		{
			qli_runtime_error("SAVING FAILURE");
		}
#endif
		output_document.Clear();
		//Remove old mimap folder
		{
			const auto file_stub = QString("trakem2.%1").arg(QString::fromStdString(trakem2_item.mipmaps_folder));
			const auto folder_full_path = dir.absoluteFilePath(file_stub);
			QDir dir_two(folder_full_path);
			if (dir_two.exists())
			{
				const auto success = dir_two.removeRecursively();
				if (!success)
				{
					std::cout << "Failed to remove old mipmap folder at " << file_stub.toStdString() << " , do you have it open?" << std::endl;

				}
			}

		}
	}
}

trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors trakem2_processor::get_vectors_from_xml_file(const std::string& full_path)
{

	//get coordinate info
	bool success;
	const auto stage_coordinates = trakem2_file::calibration_info_from_filename(full_path, success);
	const trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors blank;
	if (!success)
	{
		return blank;
	}
	using namespace tinyxml2;
	XMLDocument input_doc;
	const auto load_valid = input_doc.LoadFile(full_path.c_str());
	if (load_valid != XML_SUCCESS)
	{
		return blank;
	}

	const auto get_coordinates_from_matrix_element = [](const std::string& matrix)
	{
		//Ex input "matrix(1.0,0.0,0.0,1.0,4226.75,4319.0)"
		size_t pos = 0;
		auto count = 0;
		while (count != 4)  //matrix has 5 commas, we want 4th one
		{
			pos += 1;
			pos = matrix.find(',', pos);
			if (pos != std::string::npos)
			{
				count++;
			}
		}
		const auto x_position = pos;
		const auto y_position = matrix.find(',', x_position + 1);
		const auto x_coordinate = std::stof(matrix.substr(x_position + 1, y_position - 1 - x_position));
		const auto y_coordinate = std::stof(matrix.substr(y_position + 1, matrix.size() - 2 - y_position));

		return trakem2_xy({ x_coordinate, y_coordinate });
	};
	std::array<trakem2_xy, 4> patch_coordinates_map;
	//Get Patch Elements / Traverse xml
	auto root = input_doc.FirstChildElement();
	while (!root->NoChildren() || root->NextSiblingElement())
	{
		const auto next_sibling = root->NextSiblingElement();
		if (!root->NoChildren())
		{
			root = root->FirstChildElement();
		}
		else if (next_sibling)
		{
			root = root = next_sibling;
		}
		if (strcmp(root->Name(), "t2_patch") == 0)
		{
			const std::string filename = root->Attribute("file_path");
			const auto row_pos = filename.find('r');
			if (row_pos == std::string::npos)
			{
				continue;
			}
			const auto col_value = std::stoi(filename.substr(row_pos - 2, 1));
			const auto row_value = std::stoi(filename.substr(row_pos + 1, 1));
			if ((col_value < 2) && (row_value < 2))
			{
				const auto idx = col_value + row_value * 2;
				const auto patch_coordinate = get_coordinates_from_matrix_element(root->Attribute("transform"));
				patch_coordinates_map.at(idx) = patch_coordinate;
			}
		}
	}
	const auto validity_check = [](const trakem2_xy& xy) {return xy.is_valid(); };
	const auto check_light_path_vector = std::all_of(patch_coordinates_map.begin(), patch_coordinates_map.end(), validity_check);
	if (!check_light_path_vector)
	{
		return blank;
	}
	const auto& c0_r0 = patch_coordinates_map.at(0);
	const auto& c1_r0 = patch_coordinates_map.at(1);
	const auto& c0_r1 = patch_coordinates_map.at(2);
	const auto& c1_r1 = patch_coordinates_map.at(3);
	const auto delta_c = ((c1_r0 - c0_r0) + (c1_r1 - c0_r1)) * 0.5f;
	const auto delta_r = ((c0_r1 - c0_r0) + (c1_r1 - c1_r0)) * 0.5f;
	trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors vectors = { delta_c ,delta_r,stage_coordinates };
	return vectors;
}


trakem2_stage_coordinate_to_pixel_mapper trakem2_stage_coordinate_to_pixel_mapper::get_pass_through_mapper(const float pixel_ratio)
{

	const trakem2_xy column_step(500, 0);
	const trakem2_xy row_step(0, 500);
	const calibration_info calibration_info(500, 500, pixel_ratio);
	const displacement_vectors displacement_vectors(column_step, row_step, calibration_info);
	const trakem2_stage_coordinate_to_pixel_mapper mapper(displacement_vectors);
#if _DEBUG
	{
		const trakem2_xy nothing_should_change(84, 23);
		const auto did_anything_change = mapper.map(nothing_should_change, pixel_ratio);
		if (!nothing_should_change.approx_equal(did_anything_change))
		{
			qli_runtime_error("Oh Nope");
		}
	}
#endif
	return mapper;
}


trakem2_xy trakem2_stage_coordinate_to_pixel_mapper::map(const trakem2_xy& xy_stage_coordinates, const float current_ratio) const
{
#if _DEBUG
	{
		const auto settings_valid = this->settings.is_valid();
		if (!settings_valid)
		{
			qli_runtime_error("Oh Nope");
		}
	}
#endif
	//So we moved [500,0] but actually moved [21,95]
	//So for every [1,0] we need to multiply by [21,95]/500
	//Next we moved [0,492] but actually moved [-52,12]
	//So for every [0,1] we need to multiply by [-52,12]/492
	const auto mag_factor = settings.calibration_in_stage_microns.calibration_pixel_ratio / current_ratio;
	const auto x_step_per_micron = settings.column_step_in_pixels / settings.calibration_in_stage_microns.calibration_steps_in_stage_microns.x;
	const auto y_step_per_micron = settings.row_step_in_pixels / settings.calibration_in_stage_microns.calibration_steps_in_stage_microns.y;
	const auto pixel_coordinates = x_step_per_micron * xy_stage_coordinates.x * mag_factor + y_step_per_micron * xy_stage_coordinates.y * mag_factor;
	return pixel_coordinates;
}
