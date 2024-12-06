#include "stdafx.h"
#include "histogram_info.h"
display_settings::display_ranges histogram_info::predict_display_ranges() const
{
	display_settings::display_ranges ret(samples_per_pixel);
	const auto comp = [](const histogram_meta_info& info)
	{
		return display_range{ info.bot,info.top };
	};
	std::transform(info.begin(), info.begin()+samples_per_pixel, ret.begin(), comp);
	return ret;
}
