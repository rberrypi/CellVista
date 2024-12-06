#pragma once
#ifndef COMPUTE_ENGINE_DEMOSAIC_H
#define COMPUTE_ENGINE_DEMOSAIC_H
#include "camera_frame.h"
#include <thrust/device_vector.h>
#include "pitched_memory_unsafe_pointer.h"
#include <boost/container/static_vector.hpp>
#include <mutex>

enum class polarizer_demosaic_kind { p0, p45, p90, p135 };

struct demosaic_info final
{
	typedef boost::container::static_vector<int, 4> frames;
	frames frames_made;
	frame_size output_size;
	int samples_per_pixel;
	demosaic_info(const frames& frames, const frame_size& output_size, const int samples_per_pixel) :frames_made(frames), output_size(output_size), samples_per_pixel(samples_per_pixel) {}
};

class demosaic_structs
{
	pitched_memory_unsafe_pointer<unsigned short> demosaic_buffer_;
	cudaTextureObject_t demosaic_buffer_tex_ = 0;
	frame_size demosaic_last_input_;
	typedef thrust::device_vector<unsigned short> input_buffer;
	frame_size demosaic_rggb_14(input_buffer& output_buffer, const camera_frame<unsigned short>& input_image_h);
	frame_size demosaic_polarizer_0_45_90_135(input_buffer& output_buffer_a, input_buffer& output_buffer_b, input_buffer& output_buffer_c, input_buffer& output_buffer_d, const camera_frame<unsigned short>& input_image_h);
	frame_size demosaic_polarizer_doubles(input_buffer& output_buffer_a, input_buffer& output_buffer_d, const camera_frame<unsigned short>& input_image_h, demosaic_mode mode);
	frame_size demosaic_polarizer_pass_one(input_buffer& output_buffer_a, const camera_frame<unsigned short>& input_image_h, polarizer_demosaic_kind kind);
protected:
	std::vector<input_buffer> input_frames;//kinda like a dirty hack so we only pass around the meta data?
	std::mutex protect_input_lock;
	void fuzz(const demosaic_info& patterns);
public:
	[[nodiscard]] demosaic_info load_resize_and_demosaic(const camera_frame<unsigned short>& input_image_h, const processing_quad& quad, bool is_live);
	[[nodiscard]] static int pattern_to_load( phase_retrieval retrieval, int pattern_idx);
};

#endif