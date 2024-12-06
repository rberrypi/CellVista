#pragma once
#ifndef RENDER_PASS_THROUGH_H
#define RENDER_PASS_THROUGH_H
#include "render_engine.h"
#include <thrust/device_vector.h>

struct ml_transformer;
class render_engine_pass_through final : public render_engine
{
	std::unique_ptr<ml_transformer> ml_transformer_;
	thrust::device_vector<float> transform;
	thrust::device_vector<int> label_buffer;
public:
	void paint_surface(bool is_live, const camera_frame<float>& img_d, const gui_message& msg, const dpm_settings* dpm_settings) override;
	explicit render_engine_pass_through(const frame_size& max_camera_size);
	~render_engine_pass_through();
};

#endif