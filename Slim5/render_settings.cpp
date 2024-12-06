#include "stdafx.h"
#include "render_settings.h"
bool segmentation_settings::writes_files() const
{
	const auto has_write = segmentation_keep_originals || segmentation == segmentation_mode::off;
	return has_write;
}

segmentation_settings segmentation_settings::default_segmentation_settings()
{
	const auto threshold_value = 0.1f;
	const segmentation_feature_bounding bounding = { 0,1 };
	const segmentation_feature_circularity circularity = { 0,1 };
	const auto max_area = std::numeric_limits<float>::max();
	const segmentation_feature_area area = { 1, max_area };
	return segmentation_settings(segmentation_mode::threshold, threshold_value, bounding, circularity, area, { true });
}