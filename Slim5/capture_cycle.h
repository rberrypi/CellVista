#pragma once
#ifndef SLIM_STRUCTS_H
#define SLIM_STRUCTS_H
#include "phase_processing.h"

struct cycle_position
{
	int denoise_idx, pattern_idx;
	[[nodiscard]] bool is_first() const noexcept
	{
		return denoise_idx == 0 && pattern_idx == 0;
	}
	[[nodiscard]] bool operator== (const cycle_position& b) const noexcept
	{
		return pattern_idx == b.pattern_idx && denoise_idx == b.denoise_idx;
	}
	[[nodiscard]] static cycle_position start_position() noexcept
	{
		return cycle_position(0,-1);
	}
	cycle_position(const int denoise_idx, const int pattern_idx) noexcept : denoise_idx(denoise_idx), pattern_idx(pattern_idx) {}
	cycle_position() noexcept : cycle_position(0, 0)
	{

	}
	void advance(const cycle_position& cycle_limit) noexcept
	{
		pattern_idx = (pattern_idx + 1);
		if (pattern_idx == cycle_limit.pattern_idx)
		{
			denoise_idx = (denoise_idx + 1) % cycle_limit.denoise_idx;
		}
		pattern_idx = pattern_idx % cycle_limit.pattern_idx;

	}
};

struct capture_iterator_view final
{
	cycle_position cycle_limit;
	phase_retrieval retrieval;
	phase_processing processing;
	denoise_mode denoise_mode;
	[[nodiscard]] int frame_count() const noexcept
	{
		return cycle_limit.denoise_idx * cycle_limit.pattern_idx;
	}
};
#endif