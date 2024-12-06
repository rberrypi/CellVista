#pragma once
#ifndef COMPUTE_ENGINE_SHARED_H
#define COMPUTE_ENGINE_SHARED_H
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "frame_size.h"
#include "frame_meta_data.h"
typedef const thrust::device_vector<unsigned short>& in_frame;
typedef thrust::device_vector<float>& out_frame;
typedef thrust::host_vector<float>& out_frame_h;
typedef thrust::device_vector<float> camera_frame_internal_buffer;

struct internal_frame_meta_data : image_info, frame_meta_data
{
	internal_frame_meta_data(const image_info& frame, const frame_meta_data& frame_meta_data) noexcept : image_info(frame), frame_meta_data(frame_meta_data)
	{
	}
	internal_frame_meta_data() noexcept : internal_frame_meta_data(image_info(), frame_meta_data()) {}
	
};

struct camera_frame_internal final : internal_frame_meta_data
{
	camera_frame_internal_buffer* buffer;
	camera_frame_internal(camera_frame_internal_buffer* buffer, const internal_frame_meta_data& meta_data) noexcept : internal_frame_meta_data(meta_data), buffer(buffer)
	{
	}
	camera_frame_internal() noexcept :camera_frame_internal(nullptr, internal_frame_meta_data())
	{
	}
	[[nodiscard]] bool is_valid() const noexcept;
};

struct background_frame final : internal_frame_meta_data
{
	camera_frame_internal_buffer buffer;//same as before but now it owns the buffer!
	void load_buffer(const camera_frame_internal& frame);
	[[nodiscard]] image_info info() const noexcept
	{
		return static_cast<image_info>(*this);
	}
	[[nodiscard]] bool is_valid() const noexcept
	{
		return image_info::is_valid() && samples() == buffer.size();
	}
};

#endif