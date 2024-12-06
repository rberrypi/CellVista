#pragma once
#ifndef CAMERA_FRAME_H
#define CAMERA_FRAME_H

#include "frame_size.h"
#include "frame_meta_data.h"
template <typename T>
struct camera_frame : image_info, frame_meta_data
{
	T* img;
	explicit camera_frame(T* data, const image_info& info, const frame_meta_data& meta_data) noexcept: image_info(info), frame_meta_data(meta_data), img(data)
	{
	}
	camera_frame() : camera_frame(nullptr, image_info(), frame_meta_data()) {}

	[[nodiscard]] bool is_valid() const noexcept
	{
		return (img != nullptr) && image_info::is_valid();
	}

	[[nodiscard]] bool is_valid_for_render() const noexcept
	{
		return is_valid() && !is_complex();
	}
};
#endif