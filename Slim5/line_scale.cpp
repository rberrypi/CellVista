#include "stdafx.h"
#include "line_scale.h"
#include "channel_settings.h"
#define _USE_MATH_DEFINES
#include <math.h>

line_scale line_scale::compute_mass(const float lambda, const float increment)
{
	//lambda in microns
	const auto grams_per_micron = static_cast<float>(lambda / (increment * 2.0f * M_PI));
	const auto factor = grams_per_micron;
	const auto femtograms = 1000 * factor;
	return { femtograms, 0, true };
}

line_scale line_scale::compute_height(const float lambda, const float delta_n)
{
	const auto factor = static_cast<float>(lambda / delta_n / (2 * M_PI));
	return{ factor, 0.0f, false };
}

line_scale line_scale::compute_refractive_index(const float lambda, const float height, const float n_media)
{
	const auto factor = static_cast<float>(lambda / (2 * M_PI * height));
	const auto displacement = n_media;
	return{ factor, displacement, false };
}

line_scale line_scale::compute_qdic(const float dx)
{
	return{ 1.0f / dx, 0.0f, false };
}

line_scale line_scale::compute_filter(const channel_settings& settings, const int color_idx)
{
	//auto n = info.out.n();
	const auto h = settings.obj_height;
	const auto med = settings.n_media;
	const auto obj = settings.n_cell;
	const auto delta_n = obj - med;//maybe an abs here?
	const auto lambda = settings.wave_lengths.at(color_idx);
	const auto processing = settings.processing;
	switch (processing)
	{
	case phase_processing::mass: return compute_mass(lambda, settings.mass_inc);
	case phase_processing::height: return compute_height(lambda, delta_n);
	case phase_processing::refractive_index: return compute_refractive_index(lambda, h, med);
	default: return{};
	}
}