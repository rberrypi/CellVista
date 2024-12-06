#pragma once
#ifndef RENDER_ENGINE_H
#define RENDER_ENGINE_H
#include "gui_message.h"
#include "camera_frame.h"
#include <boost/noncopyable.hpp>
class render_engine : boost::noncopyable
{
	static void write_camera_frame_d(const camera_frame<float>& img_d, const QString& full_path);
	static void write_segmentation_d(const int* img_d, const frame_size& img_d_frame, const QString& full_path);
protected:
	static void process_messages(const camera_frame<float>& img_d, const int* label_ptr, const gui_message& msg);
public:
	virtual void paint_surface(bool is_live, const camera_frame<float>& img_d, const gui_message& msg, const dpm_settings* dpm_settings = nullptr) = 0;
	virtual ~render_engine() = default;
};

#endif