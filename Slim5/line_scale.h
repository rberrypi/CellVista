#pragma once
#ifndef LINE_SCALE_H
#define LINE_SCALE_H	

struct channel_settings;

struct line_scale final
{
	float m, b;
	bool threshold;
	line_scale() noexcept:line_scale(1, 0, false) {}

	line_scale(const float m, const float b, const bool threshold) noexcept:
		m(m), b(b), threshold(threshold)
	{
	}
	bool operator== (const line_scale& test) const noexcept
	{
		return m == test.m && b == test.b && threshold == test.threshold;
	}

	[[nodiscard]] bool no_scale() const noexcept
	{
		const auto default_object = line_scale();
		return *this == default_object;
	}
	//
	[[nodiscard]] static line_scale compute_mass(float lambda, float increment);
	[[nodiscard]] static line_scale compute_height(float lambda, float delta_n);
	[[nodiscard]] static line_scale compute_refractive_index(float lambda, float height, float n_media);
	[[nodiscard]] static line_scale compute_qdic(float dx);
	[[nodiscard]] static line_scale compute_filter(const channel_settings& settings, int color_idx);
};
#endif