#pragma once
#ifndef SAVE_IMAGE_H
#define SAVE_IMAGE_H

#include "frame_meta_data.h"
#include <filesystem>
#include "qli_runtime_error.h"
template <typename T>
struct tiff_image : image_info
{
	//todo merge this with frame_size
	std::vector<T> img;
	[[nodiscard]] size_t bytes() const noexcept
	{
		return samples() * sizeof(T);
	}
	[[nodiscard]] size_t samples() const noexcept
	{
		const auto vector_samples = img.size();
#if _DEBUG
		{
			const auto expected_samples = static_cast<image_info>(*this).samples();
			if (expected_samples != vector_samples)
			{
				qli_runtime_error("Buffer Mismatch");
			}
		}
#endif
		return vector_samples;
	}
	explicit tiff_image(const image_info& info) noexcept : image_info(info)
	{
		const auto samples = info.samples();
		img.resize(samples);
	}
	[[nodiscard]] bool is_garbage() const noexcept
	{
		const auto all_same_value = [&](const T& t)
		{
			return t == img.front();
		};
		const auto all_the_same = all_of(img.begin(), img.end(), all_same_value);
		return all_the_same;
	}
	tiff_image() noexcept : tiff_image(image_info()) {}
};

template <typename T> tiff_image<T> read_buffer(const std::string& file_name);
template <typename T> void read_buffer(const std::string& file_name, tiff_image<T>& ret);
template void read_buffer(const std::string& file_name, tiff_image<float>& ret);
template tiff_image<int> read_buffer(const std::string& file_name);
template tiff_image<unsigned short> read_buffer(const std::string& file_name);
template tiff_image<float> read_buffer(const std::string& file_name);

//

template <typename  T>
void write_tif(const std::string& name, const T* ptr, unsigned int cols, unsigned int rows, int samples_per_pixel, const  frame_meta_data* meta = nullptr);

template void write_tif(const std::string& name, const unsigned char* ptr, unsigned int cols, unsigned int rows, int samples_per_pixel, const  frame_meta_data* meta);
template void write_tif(const std::string& name, const unsigned short* ptr, unsigned int cols, unsigned int rows, int samples_per_pixel, const  frame_meta_data* meta);
template void write_tif(const std::string& name, const int* ptr, unsigned int cols, unsigned int rows, int samples_per_pixel, const  frame_meta_data* meta);
template void write_tif(const std::string& name, const float* ptr, unsigned int cols, unsigned int rows, int samples_per_pixel, const  frame_meta_data* meta);
template void write_tif(const std::string& name, const double* ptr, unsigned int cols, unsigned int rows, int samples_per_pixel, const  frame_meta_data* meta);

#endif