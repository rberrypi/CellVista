#pragma once
#include <program_config.h>

#ifndef ML_TRANSFORMER_H
#define ML_TRANSFORMER_H
#include "compute_engine_shared.h"

struct ml_engine;
struct ml_transformer
{
	thrust::device_vector<float> resize_buffer;
	thrust::device_vector<float> ml_input_buffer;
	thrust::device_vector<float> ml_output_buffer;

	// Stack input into RGB
	thrust::device_vector<float> ml_input_rgb_buffer;
	thrust::device_vector<float> ml_input_r_buffer;
	thrust::device_vector<float> ml_input_g_buffer;
	thrust::device_vector<float> ml_input_b_buffer;
	// Raw output for classification problem
	thrust::device_vector<float> ml_output_raw_buffer; 
	typedef  std::unordered_map<ml_remapper_file::ml_remapper_types, std::shared_ptr<ml_engine>>  file_to_engine_mapper;
	static file_to_engine_mapper engines;
	const static int divisibility_constraint = 4;
	[[nodiscard]] static int round_up_division(const int dimension, const float scale_ratio)
	{
		return divisibility_constraint * std::ceil(static_cast<float>(dimension * scale_ratio) / divisibility_constraint);
	}
public:
	thrust::device_vector<float> phase_resized;
	//Pre bakes all available networks
	static std::shared_ptr<ml_engine> safe_get_ml_engine(ml_remapper_file::ml_remapper_types item);
	//this should be a settings structure
	[[nodiscard]] static frame_size get_ml_output_size(const frame_size& camera_frame, float input_pixel_ratio, ml_remapper_file::ml_remapper_types ml_type, const bool skip_rescale = false);
	[[nodiscard]] bool do_ml_transform(float* destination_ml_pointer, const frame_size& destination_frame_size, const  float* input_ptr, const frame_size& input_frame_size, const ml_remapper& settings, float input_pixel_ratio, bool skip_ml);
	bool fuck_this_shit(float* destination_ml_pointer, const frame_size& destination_frame_size, const  float* input_ptr, const frame_size& input_frame_size, const ml_remapper& settings, float input_pixel_ratio, const bool skip_ml);
	explicit ml_transformer(const frame_size& camera_frame_size = frame_size());//this pre-allocates, kinda
	void set_network_size(const frame_size& camera_frame_size);
	~ml_transformer();
	static void pre_bake();
};


#endif
