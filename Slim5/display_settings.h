#pragma once
#ifndef DISPLAY_SETTINGS_H
#define DISPLAY_SETTINGS_H
#include <string>
#include <array>
#include "common_limits.h"
#include "approx_equals.h"
#include <boost/container/small_vector.hpp>
struct lut final
{
	std::string name;
	std::array<unsigned char, 768> data;
};

struct display_range final
{
	float min, max;
	[[nodiscard]] bool operator ==(const display_range& b) const noexcept
	{
		return b.min == min && b.max == max;
	}

	[[nodiscard]] bool item_approx_equals(const display_range& a) const noexcept
	{
		return approx_equals(a.min, min) && approx_equals(a.max, max);
	}

	[[nodiscard]] display_range predict_max_possible() const noexcept
	{
		const auto max_value = std::max(abs(min), abs(max));
		return { -max_value,max_value };
	}

};

const display_range camera_intensity_placeholder = { 70.0f,65535.0f };

struct display_settings
{
	constexpr static int total_luts = 22;
	constexpr static int bw_lut = 0;
	constexpr static int jet_lut = 2;
	constexpr static int blank_lut = 20;
	typedef std::array<lut, total_luts> lut_collection;
	const static lut_collection luts;
	static lut_collection get_luts() noexcept;
	static void deploy_luts(const std::string& lut_directory);
	int display_lut;
	typedef boost::container::small_vector<display_range, max_samples_per_pixel> display_ranges;
	display_ranges ranges{};
	display_settings(const int lut_idx, const display_ranges& ranges) : display_lut(lut_idx), ranges(ranges)
	{
	}
	display_settings(const int lut_idx, const display_range range) noexcept: display_lut(lut_idx)
	{
		ranges.push_back(range);
	}
	display_settings() noexcept : display_settings(0, { 0,0 }) {}

	void set_samples_per_pixel(const int samples)
	{
		const auto default_range = ranges.empty() ? display_range()  : ranges.front();
		ranges.resize(samples,default_range);
	}
	
	[[nodiscard]] display_ranges predict_max_possible() const
	{
		auto return_me = this->ranges;
		for (auto& range : return_me)
		{
			range = range.predict_max_possible();
		}
		return return_me;
	}

	[[nodiscard]] bool item_approx_equals(const display_settings& b) const
	{
		const auto check_ranges = [&] {
			const auto size = ranges.size();
			for (display_ranges::size_type i = 0; i < size; ++i)
			{
				const auto is_approximately_equal = this->ranges.at(i).item_approx_equals(b.ranges.at(i));
				if (!is_approximately_equal)
				{
					return false;
				}
			}
			return true;
		};
		return display_lut == b.display_lut && check_ranges();
	}
	bool operator== (const display_settings& b) const
	{
		return display_lut == b.display_lut && ranges == b.ranges;
	}

	bool operator< (const display_settings& b) const
	{
		const auto a_range = ranges.at(0).max - ranges.at(0).min;
		const auto b_range = b.ranges.at(0).max - b.ranges.at(0).min;
		return a_range > b_range;
	}
};

#endif