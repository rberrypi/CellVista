#pragma once
#ifndef CUFFT_SHARED_H
#define CUFFT_SHARED_H
#include <boost/noncopyable.hpp>
#include <thrust/device_vector.h>

struct ft_rectangle final
{
	int pitch_width_numel, pitch_height_numel, left, top;

	[[nodiscard]] bool is_simple(const int width, const int height) const
	{
		return pitch_width_numel == width && pitch_height_numel == height && top == 0 && left == 0;
	}
	bool operator==(const ft_rectangle& rhs) const
	{
		return pitch_width_numel == rhs.pitch_width_numel && pitch_height_numel == rhs.pitch_height_numel && left == rhs.left && top == rhs.top;
	}

	bool operator!=(const ft_rectangle& rhs) const
	{
		return !(*this == rhs);
	}
};

struct ft_settings
{
	ft_rectangle src, dst;
	int width, height;

	[[nodiscard]] bool is_simple_case() const
	{
		return src.is_simple(width, height) && dst.is_simple(width, height);
	}
	ft_settings(const int width, const int height, const ft_rectangle& src, const ft_rectangle& dst) : src(src), dst(dst), width(width), height(height)
	{

	}
	ft_settings(const int width, const int height) : src({ width,height,0,0 }), dst({ width,height,0,0 }), width(width), height(height)
	{

	}
	ft_settings() : ft_settings(0, 0) {}
	bool operator==(const ft_settings& rhs) const
	{
		return width == rhs.width && height == rhs.height && src == rhs.src && dst == rhs.dst;
	}

	bool operator!=(const ft_settings& rhs) const
	{
		return !(*this == rhs);
	}
};

class cufft_wrapper final : boost::noncopyable, ft_settings
{
	int id_;
	bool is_initialized_;
	//plan is CUFFT_C2C
	void free_handle();
public:
	void take_ft(thrust::device_vector<float2>& input, thrust::device_vector<float2>& out, int width, int height, bool is_forward);
	void take_ft(float2* input, float2* out, int width, int height, bool is_forward);
	void take_ft(float2* input, float2* out, const ft_settings& settings, bool is_forward);
	cufft_wrapper() : id_(0), is_initialized_(false) {}
	~cufft_wrapper();
};
#endif