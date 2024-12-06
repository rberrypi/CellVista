#include "render_engine_pass_through.h"
#include "thrust_resize.h"
#include "write_debug_gpu.h"
#include "gui_message.h"
#include "ml_transformer.h"
#include "device_factory.h"
render_engine_pass_through::render_engine_pass_through(const frame_size& max_camera_size) :  ml_transformer_(std::make_unique<ml_transformer>(max_camera_size))
{

}

render_engine_pass_through::~render_engine_pass_through() = default;

void render_engine_pass_through::paint_surface(bool, const camera_frame<float> & img_d, const gui_message & msg, const dpm_settings*)
{
	//This stuff is kinda fucked because the labels should be written, for example. Sadly this component is fixed in the newer version
	int* label_ptr = nullptr;
#if INCLUDE_ML
	if (img_d.ml_remapper_type != ml_remapper_file::ml_remapper_types::off)
	{
		//thrust::device_vector<float>& destination_buffer, frame_size& output_frame_size, thrust::device_vector<float>& input_buffer, const frame_size& input_frame_size, const ml_remapper& settings, float input_pixel_ratio
		const auto output_size = ml_transformer::get_ml_output_size(img_d, img_d.pixel_ratio, img_d.ml_remapper_type);
		auto destination_ptr = thrust_safe_get_pointer(transform, output_size.n());
		ml_transformer_->do_ml_transform(destination_ptr, output_size, img_d.img, img_d, img_d, img_d.pixel_ratio, false);
		write_debug_gpu(label_ptr, output_size.width, output_size.height, 1, "ml_transformed.tif");
	}
#endif
	process_messages(img_d, label_ptr, msg);
}