#include "stdafx.h"
#include "render_engine.h"
#include "write_debug_gpu.h"
#include "device_factory.h"
#include <QDir>
#include "qli_runtime_error.h"

void render_engine::write_camera_frame_d(const camera_frame<float>& img_d, const QString& full_path)
{
	const auto full_path_std_string = full_path.toStdString();
	write_debug_gpu(img_d.img, img_d.width, img_d.height, img_d.samples_per_pixel, full_path_std_string.c_str(), true);
}

void render_engine::write_segmentation_d(const int* img_d, const frame_size& img_d_frame, const QString& full_path)
{
	const auto full_path_std_string = full_path.toStdString();
	write_debug_gpu(img_d, img_d_frame.width, img_d_frame.height, 1, full_path_std_string.c_str(), true);
}

void render_engine::process_messages(const camera_frame<float>& img_d, const int* label_ptr, const gui_message& msg)
{
	switch (msg.kind)
	{
	case gui_message_kind::none:
	{
		return;
	}
	case gui_message_kind::live_image_to_file:
	{
		write_camera_frame_d(img_d, msg.val.toString());
		break;
	}
	default:
		qli_not_implemented();
	}
}