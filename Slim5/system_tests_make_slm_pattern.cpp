#include "stdafx.h"
#include "write_tif.h"
#include <random>
#include "device_factory.h"
#include "slm_device.h"
#include <algorithm>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>

#include "qli_runtime_error.h"

std::string make_slm_pattern(const int seed)
{
	qli_not_implemented();
	/*
	auto filename = std::to_string(seed) + "_slm.tif";
	if (!std::experimental::filesystem::exists(filename))
	{
		const auto slm_frame_size = static_cast<frame_size>(*D->slm);
		std::vector<unsigned char> data(slm_frame_size.n());
		std::random_device rnd_device;
		// Specify the engine and distribution.
		std::mt19937 mersenne_engine{ rnd_device() };  // Generates random integers
		std::uniform_int_distribution<unsigned short> dist{ 0, static_cast<unsigned short>(255) };

		const auto gen = [&dist, &mersenne_engine]() {
			return static_cast<unsigned char>(dist(mersenne_engine));
		};
		std::generate(begin(data), end(data), gen);
		write_tif(filename, data.data(), slm_frame_size.width, slm_frame_size.height, 1, nullptr);
	}
	return filename;
	*/
}
