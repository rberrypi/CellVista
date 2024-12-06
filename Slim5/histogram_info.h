#pragma once
#ifndef HISTOGRAM_INFO_H
#define HISTOGRAM_INFO_H
#include "common_limits.h"
#include "display_settings.h"
#include <algorithm>
struct histogram_meta_info final
{
	int bot_idx, top_idx; //hack hack hack, needs to be combined with bot and top & display range
	float standard_deviation, median, bot, top; //these are in histogram bin indexes
	[[nodiscard]] bool is_valid() const;
};

struct histogram_info final
{
	//hmm, why are these not atomic?
	std::array<std::array<int, 256>, max_samples_per_pixel> histogram_channels{};
	std::array<histogram_meta_info, max_samples_per_pixel> info{};
	[[nodiscard]] display_settings::display_ranges predict_display_ranges() const;
	int samples_per_pixel;

	histogram_info() noexcept: samples_per_pixel(1) //-V730
	{
		info = { {histogram_meta_info()} };
		for (auto&& channel : histogram_channels)
		{
			std::fill(channel.begin(), channel.end(), 0);
		}
	}
};
#endif
