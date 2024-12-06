#pragma once
#ifndef FRAME_SIZE_H
#define FRAME_SIZE_H

struct frame_size
{
	int width, height;
	frame_size() noexcept : frame_size(0, 0) {}
	explicit frame_size(const int width, const int height) noexcept : width(width), height(height) {
	}

	[[nodiscard]] size_t n() const noexcept
	{
		return width * height;
	}

	[[nodiscard]]  bool operator==(const frame_size& rhs) const noexcept
	{
		return height == rhs.height && width == rhs.width;
	}
	[[nodiscard]]  bool operator!=(const frame_size& rhs) const noexcept
	{
		return !(*this == rhs);
	}
	[[nodiscard]]  bool operator<(const frame_size& rhs) const noexcept
	{
		return n() < rhs.n();
	}
	[[nodiscard]] bool is_valid() const noexcept
	{
		//we have low standards
		return width * height > 0;
	}
};

struct image_info : frame_size
{
	int samples_per_pixel;
	enum class complex { yes, no };
	complex complexity;

	[[nodiscard]] bool is_complex() const noexcept
	{
		return complexity == complex::yes;
	}
	image_info() noexcept: image_info(frame_size(0, 0), 0, complex::no) {}
	image_info(const frame_size& frame_size, const int samples_per_pixel, const complex complexity) noexcept : frame_size(frame_size), samples_per_pixel(samples_per_pixel), complexity(complexity)
	{
	}
	[[nodiscard]] size_t samples() const noexcept
	{
		const auto complex_count = is_complex() ? 2 : 1;
		return n() * samples_per_pixel * complex_count;
	}
	[[nodiscard]] bool operator==(const image_info& rhs) const noexcept
	{
		return samples_per_pixel==rhs.samples_per_pixel&& complexity==rhs.complexity && frame_size::operator==(rhs);
	}
	[[nodiscard]] bool is_valid() const noexcept
	{
		return (samples_per_pixel == 3 || samples_per_pixel == 1)&& frame_size::is_valid();
	}
	[[nodiscard]] bool info_matches_except_complexity(const image_info& rhs) const noexcept
	{
		return samples_per_pixel==rhs.samples_per_pixel && frame_size::operator==(rhs);
	}
};

struct render_dimensions final : frame_size
{
	float digital_scale;
	render_dimensions() noexcept :render_dimensions(frame_size(), 1.0f) {}
	render_dimensions(const frame_size& frame_size, const float scale) noexcept :frame_size(frame_size), digital_scale(scale)
	{

	}
};
#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(frame_size)
Q_DECLARE_METATYPE(render_dimensions)
#endif

#endif