#pragma once
#ifndef RENDER_SETTINGS_H
#define RENDER_SETTINGS_H
#include <unordered_map>
#include "display_settings.h"
#include "ml_structs.h"

enum class segmentation_mode { off, threshold, onnx_file };
const std::unordered_map<segmentation_mode, std::string> threshold = { { segmentation_mode::off, "Off"},{ segmentation_mode::threshold, "Threshold" },{ segmentation_mode::onnx_file, "ONNX File" } };

struct segmentation_feature_bounding
{
	float segmentation_bounding_min, segmentation_bounding_max;
	[[nodiscard]] bool item_approx_equals(const segmentation_feature_bounding& b) const noexcept
	{
		return approx_equals(segmentation_bounding_min, b.segmentation_bounding_min) && approx_equals(segmentation_bounding_max, b.segmentation_bounding_max);
	}
};

struct segmentation_feature_circularity
{
	float segmentation_circ_min, segmentation_circ_max;
	[[nodiscard]] bool item_approx_equals(const segmentation_feature_circularity& b) const noexcept
	{
		return approx_equals(segmentation_circ_min, b.segmentation_circ_min) && approx_equals(segmentation_circ_max, b.segmentation_circ_max);
	}
};

struct segmentation_feature_area
{
	float segmentation_area_min, segmentation_area_max;
	[[nodiscard]] bool item_approx_equals(const segmentation_feature_area& b) const noexcept
	{
		return approx_equals(segmentation_area_min, b.segmentation_area_min) && approx_equals(segmentation_area_max, b.segmentation_area_max);
	}
};

struct segmentation_save_settings
{
	bool segmentation_keep_originals;
	[[nodiscard]] bool item_approx_equals(const segmentation_save_settings& b) const noexcept
	{
		return segmentation_keep_originals== b.segmentation_keep_originals;
	}
};

struct segmentation_onnx_settings
{
	std::string onnx_file_path;
	[[nodiscard]] bool item_approx_equals(const segmentation_onnx_settings& b) const noexcept
	{
		return onnx_file_path== b.onnx_file_path;
	}
};

struct segmentation_settings : segmentation_feature_bounding, segmentation_feature_circularity, segmentation_feature_area, segmentation_save_settings, segmentation_onnx_settings
{
	segmentation_mode segmentation;
	float segmentation_min_value;
	//if I come up with a different segmentation mode, we can make this a separate structure/union
	explicit segmentation_settings(const segmentation_mode segmentation, const float segmentation_min_value, const segmentation_feature_bounding& bounding, const segmentation_feature_circularity& circ, const segmentation_feature_area& area, const segmentation_save_settings& save_settings) noexcept : segmentation_feature_bounding(bounding), segmentation_feature_circularity(circ), segmentation_feature_area(area), segmentation_save_settings(save_settings), segmentation(segmentation), segmentation_min_value(segmentation_min_value) {}
	segmentation_settings() noexcept: segmentation_settings(segmentation_mode::off, 0.0f, { 0,0 }, { 0,0 }, { 0,0 }, { true }) {}
	static segmentation_settings default_segmentation_settings();

	[[nodiscard]] bool writes_files() const;

	[[nodiscard]] bool item_approx_equals(const segmentation_settings& b) const noexcept
	{
		return segmentation_feature_bounding::item_approx_equals(b) && segmentation_feature_circularity::item_approx_equals(b) && segmentation_feature_area::item_approx_equals(b) && segmentation_save_settings::item_approx_equals(b) && segmentation_onnx_settings::item_approx_equals(b) && segmentation == b.segmentation && approx_equals(segmentation_min_value,b.segmentation_min_value);
	}
};


struct render_modifications
{
	bool show_crosshair, live_auto_contrast,  do_ft;
	render_modifications() noexcept: render_modifications(false, false, false) {}
	render_modifications(const bool show_crosshair, const bool live_auto_contrast,  const bool do_ft) noexcept: show_crosshair(show_crosshair), live_auto_contrast(live_auto_contrast), do_ft(do_ft) {}
	[[nodiscard]] bool item_approx_equals(const render_modifications& b) const noexcept
	{
		return show_crosshair==b.show_crosshair && live_auto_contrast==b.live_auto_contrast && do_ft==b.do_ft;
	}
};

struct render_shifter
{
	[[nodiscard]] bool item_approx_equals(const render_shifter& b) const noexcept
	{
		return approx_equals(b.ty, ty) && approx_equals(b.tx, tx);
	}
	float tx, ty;
	render_shifter(const float tx, const float ty) noexcept: tx(tx), ty(ty) {};
	render_shifter() noexcept: render_shifter(0, 0) {};

	[[nodiscard]] bool do_shift() const noexcept
	{
		return !(approx_equals(tx, 0.0f) && approx_equals(ty, 0.0f));
	}
};

struct render_settings :  display_settings, render_modifications, ml_remapper, render_shifter
{
	render_settings(const render_modifications& render_modifications, const display_settings& display_settings,  const ml_remapper& ml_remapper, const render_shifter& render_shifter) noexcept: display_settings(display_settings), render_modifications(render_modifications), ml_remapper(ml_remapper), render_shifter(render_shifter) {}
	render_settings() noexcept :render_settings(render_modifications(), display_settings(),  ml_remapper(), render_shifter()) {}
	[[nodiscard]] bool item_approx_equals(const  render_settings& b) const noexcept
	{
		return display_settings::item_approx_equals(b) && render_modifications::item_approx_equals(b) && ml_remapper::item_approx_equals(b) && render_shifter::item_approx_equals(b);
	}
};

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(segmentation_mode)
Q_DECLARE_METATYPE(render_settings)
#endif

#endif