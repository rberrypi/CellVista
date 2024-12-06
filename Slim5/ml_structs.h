#pragma once
#ifndef ML_STRUCTS_H
#define ML_STRUCTS_H
#include "frame_size.h"
#include <unordered_map>
#include <set>

#include "display_settings.h"

struct ml_remapper_qli_v2_network
{
	float auxiliary_x1;
	float auxiliary_x2;
	ml_remapper_qli_v2_network();
	ml_remapper_qli_v2_network(const float x_min, const float x_max) noexcept: auxiliary_x1(x_min), auxiliary_x2(x_max)
	{

	}
};

struct ml_remapper_file : ml_remapper_qli_v2_network
{
	//stores hard coded ml settings
	std::string network_label;
	std::string network_resource_path;
	bool do_input_scale;
	bool do_output_scale;
	float input_min, input_max;
	float output_min_in, output_max_in;
	float output_min_out, output_max_out;
	float designed_pixel_ratio;
	bool imagenet_preprocessing;
	int output_channel;
	bool shift_argmax;
	ml_remapper_file() noexcept: ml_remapper_file("Off", "", frame_size(), -3, 3, 0, 1, -3, 3, 1, true, ml_remapper_qli_v2_network(), false) {}
	ml_remapper_file(const std::string& ai_label, const std::string& network_resource_path, const frame_size& network_size, const float input_min, const float input_max, const float output_min_in, const float output_max_in, const float output_min_out, const float output_max_out, const float designed_pixel_ratio, const bool do_input_scale, const ml_remapper_qli_v2_network& ml_remapper_qli_v2_network, int output_channel = 1, bool imagenet_preprocessing = false, bool do_output_scale=true, bool shift_argmax = false) noexcept: ml_remapper_qli_v2_network(ml_remapper_qli_v2_network),
		network_label(ai_label), network_resource_path(network_resource_path), do_input_scale(do_input_scale), input_min(input_min), input_max(input_max), output_min_in(output_min_in), output_max_in(output_max_in), output_min_out(output_min_out), output_max_out(output_max_out), designed_pixel_ratio(designed_pixel_ratio), network_size(network_size), output_channel(output_channel), imagenet_preprocessing(imagenet_preprocessing), do_output_scale(do_output_scale), shift_argmax(shift_argmax)
	{

	};
	enum class ml_remapper_types {
		off, pass_through_test_engine, sperm_slim_40x, glim_dapi_20x, glim_dapi_20x_480, glim_dil_20x, slim_dapi_10x, hrslim_dapi_10x, dpm_slim, viability
	};
	static std::unordered_map<ml_remapper_types, ml_remapper_file> ml_remappers;
	const static std::set<ml_remapper_types> mappers_to_prebake;

	[[nodiscard]] frame_size get_network_size() const;
	void set_network_size(const frame_size& network_size) noexcept
	{
		this->network_size = network_size;
	}
private:
	frame_size network_size;
};

struct ml_remapper
{
	ml_remapper_file::ml_remapper_types ml_remapper_type;
	display_range ml_display_range;
	int ml_lut;
	enum class display_mode { only_remap, only_phase, overlay, none };
	display_mode ml_display_mode;

	[[nodiscard]] bool item_approx_equals(const ml_remapper& b) const noexcept
	{
		return b.ml_display_mode == ml_display_mode && b.ml_lut == ml_lut && b.ml_remapper_type == ml_remapper_type && ml_display_range.item_approx_equals(b.ml_display_range);
	}
	explicit ml_remapper(const ml_remapper_file::ml_remapper_types remapper_kind, const display_range& ml_display_range, const int ml_lut, const display_mode ml_display_mode) noexcept: ml_remapper_type(remapper_kind), ml_display_range(ml_display_range), ml_lut(ml_lut), ml_display_mode(ml_display_mode)
	{
	}
	ml_remapper() noexcept: ml_remapper(ml_remapper_file::ml_remapper_types::off, { 0,1 }, display_settings::bw_lut, display_mode::only_remap) {}

};
#ifdef QT_DLL
#include <QMetaType>
Q_DECLARE_METATYPE(ml_remapper_file::ml_remapper_types)
#endif

#endif
