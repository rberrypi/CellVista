#include "render_widget.h"
#include "compute_engine.h"
#include "cuda_error_check.h"
#include <cuda_gl_interop.h>
#include <QResizeEvent>
#include <QPainter>
#include <QOpenGLPaintDevice>
#include "device_factory.h"
#include "radial_spectrum_widget.h"
#include "ml_transformer.h"
#include "thrust_resize.h"
#include "write_debug_gpu.h"
#include "program_config.h"
#include "ml_timing.h"
#include "windows_sleep.h"
#include "chrono_converters.h"
std::chrono::microseconds ml_quick_timestamp()
{
	const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
	return microseconds;
}
	
std::unique_ptr<ml_transformer> render_widget::ml_transform;

const std::array<float, 3> render_widget::no_phase_value_under_cursor = { std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN() };

render_widget::render_widget(std::shared_ptr<compute_engine>& compute_engine,  QWindow* parent) : QWindow(parent), click_coordinates(QPointF()), selected_value(qQNaN()), selected_label(0), phase_value_under_cursor(no_phase_value_under_cursor), img_size(render_dimensions()), samples_per_pixel(0), digital_x(0.0), digital_y(0.0), scroll_offset_width(0), scroll_offset_height(0), max_label(0), dragging(false),  mouse_inside_(false), previous_program_(ogl_program::unspecified), m_update_pending_(false), first_show_(false), m_context_(nullptr), m_device_(nullptr), comp_(compute_engine), phase_resource_(nullptr), label_resource_(nullptr) {
	ml_transform = std::move(std::make_unique<ml_transformer>());
	ml_transform->pre_bake();
	//highFidelityMouseTracking = new QTimer(this);
	const auto idx_max = std::numeric_limits<GLuint>::max();
	vao_handle_ = idx_max;
	vbo_handles_ = { idx_max, idx_max };
	img_buffer_ = idx_max;
	label_buffer_ = idx_max;
	remapper_buffer_ = idx_max;
	img_texture_ = idx_max;
	label_texture_ = idx_max;
	remapper_texture_ = idx_max;
	vertex_shader_ = idx_max;
	setSurfaceType(OpenGLSurface);
}

render_widget::~render_widget() = default;

void render_widget::init_render_invariants()
{
	//
//Make VAO
	{
		glDeleteBuffers(2, vbo_handles_.data()); OGL_ERROR_CHECK();
		glDeleteVertexArrays(1, &vao_handle_); OGL_ERROR_CHECK();
		const GLfloat x = 1;
		const GLfloat y = 1;
		std::array<GLfloat, 12> vec_data = {
			-x, y,
			-x, -y,
			x, y,
			-x, -y,
			x, -y,
			x, y };
		std::array<GLfloat, 12> tex_data = {
			0, 1,
			0, 0,
			1, 1,
			0, 0,
			1, 0,
			1, 1 };
		glGenBuffers(2, vbo_handles_.data()); OGL_ERROR_CHECK();
		const auto vec_buf = vbo_handles_[0];
		const auto tex_buf = vbo_handles_[1];
		glBindBuffer(GL_ARRAY_BUFFER, vec_buf); OGL_ERROR_CHECK();

		glBufferData(GL_ARRAY_BUFFER, 8 * 2 * sizeof(GLfloat), vec_data.data(), GL_STATIC_DRAW); OGL_ERROR_CHECK();
		glBindBuffer(GL_ARRAY_BUFFER, tex_buf); OGL_ERROR_CHECK();
		glBufferData(GL_ARRAY_BUFFER, 8 * 2 * sizeof(GLfloat), tex_data.data(), GL_STATIC_DRAW); OGL_ERROR_CHECK();
		glBindBuffer(GL_ARRAY_BUFFER, 0); OGL_ERROR_CHECK();
		OGL_ERROR_CHECK();
		glGenVertexArrays(1, &vao_handle_); OGL_ERROR_CHECK();
		glBindVertexArray(vao_handle_); OGL_ERROR_CHECK();
		glEnableVertexAttribArray(0); OGL_ERROR_CHECK();
		glEnableVertexAttribArray(1); OGL_ERROR_CHECK();
		glBindBuffer(GL_ARRAY_BUFFER, vec_buf); OGL_ERROR_CHECK();
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, static_cast<GLubyte*>(nullptr)); OGL_ERROR_CHECK();
		glBindBuffer(GL_ARRAY_BUFFER, tex_buf); OGL_ERROR_CHECK();
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, static_cast<GLubyte*>(nullptr)); OGL_ERROR_CHECK();
		glBindBuffer(GL_ARRAY_BUFFER, 0); OGL_ERROR_CHECK();
		glBindVertexArray(0); OGL_ERROR_CHECK();
		OGL_ERROR_CHECK();
	}
	const auto texturing_mode = GL_NEAREST;//GL_LINEAR, blocky idk?
	const auto color = samples_per_pixel == 3;
	//Make Main Texture
	{
		glActiveTexture(GL_TEXTURE0); OGL_ERROR_CHECK();
		glDeleteTextures(1, &img_texture_); OGL_ERROR_CHECK();
		glGenTextures(1, &img_texture_); OGL_ERROR_CHECK();
		glBindTexture(GL_TEXTURE_2D, img_texture_); OGL_ERROR_CHECK();
		const auto pixel_fmt = color ? GL_RGB8 : GL_R8;
		const auto input_pixel_fmt = color ? GL_RGB : GL_RED;
		glTexImage2D(GL_TEXTURE_2D, 0, pixel_fmt, img_size.width, img_size.height, 0, input_pixel_fmt, GL_UNSIGNED_BYTE, static_cast<GLubyte*>(nullptr)); OGL_ERROR_CHECK();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, texturing_mode); OGL_ERROR_CHECK();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, texturing_mode); OGL_ERROR_CHECK();
		glBindTexture(GL_TEXTURE_2D, 0); OGL_ERROR_CHECK();
		//Bind the data buffer
		glDeleteBuffers(1, &img_buffer_); OGL_ERROR_CHECK();
		glGenBuffers(1, &img_buffer_); OGL_ERROR_CHECK();
		const auto buffer_size = sizeof(GLubyte) * img_size.n() * samples_per_pixel;
		i_dont_remember_why_this_is_here.resize(buffer_size);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, img_buffer_); OGL_ERROR_CHECK();
		glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer_size, i_dont_remember_why_this_is_here.data(), GL_DYNAMIC_COPY); OGL_ERROR_CHECK();//maybe stream copy?
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); OGL_ERROR_CHECK();
		CUDASAFECALL(cudaGraphicsGLRegisterBuffer(&phase_resource_, img_buffer_, cudaGraphicsRegisterFlagsNone)); OGL_ERROR_CHECK();
		CUDASAFECALL(cudaDeviceSynchronize()); OGL_ERROR_CHECK();
	}
	//Make overlay texture
	if (render_settings_.ml_remapper_type != ml_remapper_file::ml_remapper_types::off)
	{
		if (color)
		{
			qli_not_implemented();
		}
		glActiveTexture(GL_TEXTURE2); OGL_ERROR_CHECK();
		glDeleteTextures(1, &remapper_texture_); OGL_ERROR_CHECK();
		glGenTextures(1, &remapper_texture_); OGL_ERROR_CHECK();
		glBindTexture(GL_TEXTURE_2D, remapper_texture_); OGL_ERROR_CHECK();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, img_size.width, img_size.height, 0, GL_RED, GL_UNSIGNED_BYTE, static_cast<GLubyte*>(nullptr)); OGL_ERROR_CHECK();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, texturing_mode); OGL_ERROR_CHECK();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, texturing_mode); OGL_ERROR_CHECK();
		glBindTexture(GL_TEXTURE_2D, 0); OGL_ERROR_CHECK();
		//Bind the data buffer
		glDeleteBuffers(1, &remapper_buffer_); OGL_ERROR_CHECK();
		glGenBuffers(1, &remapper_buffer_); OGL_ERROR_CHECK();
		const auto buffersize = sizeof(GLubyte) * img_size.n() * samples_per_pixel;
		i_dont_remember_why_this_is_here.resize(buffersize);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, remapper_buffer_); OGL_ERROR_CHECK();
		glBufferData(GL_PIXEL_UNPACK_BUFFER, buffersize, i_dont_remember_why_this_is_here.data(), GL_DYNAMIC_COPY); OGL_ERROR_CHECK();//maybe stream copy?
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); OGL_ERROR_CHECK();
		CUDASAFECALL(cudaGraphicsGLRegisterBuffer(&ml_resource_, remapper_buffer_, cudaGraphicsRegisterFlagsNone)); OGL_ERROR_CHECK();
		CUDASAFECALL(cudaDeviceSynchronize()); OGL_ERROR_CHECK();
		//Make Overlay LUT

	}
	//Make Lut
	const auto make_lut = [&](const GLenum texture_unit, const int texture_unit_number, const int old_lut_idx, const std::string& id)
	{
		lut_texture_sampler lut(texture_unit, texture_unit_number, id);
		glDeleteTextures(1, &lut.texture); OGL_ERROR_CHECK();
		glDeleteSamplers(1, &lut.sampler); OGL_ERROR_CHECK();
		glGenTextures(1, &lut.texture); OGL_ERROR_CHECK();
		glGenSamplers(1, &lut.sampler); OGL_ERROR_CHECK();
		glActiveTexture(texture_unit); OGL_ERROR_CHECK();
		glBindTexture(GL_TEXTURE_1D, lut.texture); OGL_ERROR_CHECK();
		const auto default_or_current = std::max(0, old_lut_idx);
		glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, render_settings::luts[default_or_current].data.data()); OGL_ERROR_CHECK();
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); OGL_ERROR_CHECK();
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_BASE_LEVEL, 0); OGL_ERROR_CHECK();
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0); OGL_ERROR_CHECK();
		glSamplerParameteri(lut.sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST); OGL_ERROR_CHECK();
		glSamplerParameteri(lut.sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST); OGL_ERROR_CHECK();
		glSamplerParameteri(lut.sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); OGL_ERROR_CHECK();
		glBindTexture(GL_TEXTURE_1D, 0); OGL_ERROR_CHECK();
		lut.old_lut_idx = old_lut_idx;
		return lut;
	};
	img_lut = make_lut(GL_TEXTURE3, 3, (-1), "img_lut");
	remapper_lut = make_lut(GL_TEXTURE4, 4, (-1), "img_transformed_lut");
	emit frame_size_changed(img_size);
}

void render_widget::init_render_invariants(const camera_frame<float> & img_d)
{
	//
	// Should be called every time our render loop is dirtied
	//
	if (img_d.samples() == 0)
	{
		qli_invalid_arguments();
	}
#if INCLUDE_ML!=0
	const auto actual_image_size = ml_transformer::get_ml_output_size(img_d, img_d.pixel_ratio, img_d.ml_remapper_type);
#else
	const frame_size& actual_image_size = img_d;
#endif
	if (img_size == actual_image_size && samples_per_pixel == img_d.samples_per_pixel  && render_settings_.ml_remapper_type == img_d.ml_remapper_type)
	{
		return;
	}
	static_cast<frame_size&>(img_size) = actual_image_size;
	static_cast<render_settings&>(render_settings_) = img_d;
	samples_per_pixel = img_d.samples_per_pixel;
	init_render_invariants();
}

void render_widget::paint_surface(const bool is_live, const camera_frame<float> & img_d, const gui_message & msg, const dpm_settings * dpm)
{ // this shit always runs
	if (!img_d.is_valid_for_render())
	{
		qli_runtime_error("input frame to painting surface isn't valid");
	}
	{
		std::lock_guard<std::mutex> lk(ogl_context_mutex);
		if (!m_context_)
		{
			return;
		}
		const auto success = m_context_->makeCurrent(this);
		if (!success)
		{
			return;
		}
		OGL_ERROR_CHECK();
		init_render_invariants(img_d); 
		OGL_ERROR_CHECK();
		render_settings_ = img_d;
		CUDA_DEBUG_SYNC();



		//Do ML
		float* img_pointer = [&] {
#if INCLUDE_ML
			const bool dont_debug = false;
			const auto do_ml = render_settings_.ml_remapper_type != ml_remapper_file::ml_remapper_types::off;
			if (do_ml)
			{
				//Step 1 : DO ML
				LOGGER_INFO("Starting to fuck popescu with an AI dildo...");
				LOGGER_INFO("The ML image size " << img_size.height << " x " << img_size.width << " giving " << img_size.n());
				float* ml_destination = thrust_safe_get_pointer(buffer, img_size.n()); // <- memory leak, good luck
				write_debug_gpu(ml_destination, img_size.width, img_size.height, samples_per_pixel, "ML_display_0.tif", dont_debug);
				const bool rare_skip = render_settings_.ml_display_mode == ml_remapper::display_mode::only_phase;
				write_debug_gpu(img_d.img, img_d.width, img_d.height, samples_per_pixel, "With_ML_Input_in_render_thread.tif", dont_debug);
				const std::chrono::microseconds before_ml = ml_quick_timestamp();
				LOGGER_INFO("ml in_img size(w, h): " << img_size.width << ", " << img_size.height);
				LOGGER_INFO("ml out_size(w, h): " << img_d.width << ", " << img_d.height);
				const bool resized = ml_transform->do_ml_transform(ml_destination, img_size, img_d.img, img_d, img_d, img_d.pixel_ratio, rare_skip);
				const auto after_ml = ml_quick_timestamp();
				const auto ml_workflow_time = (std::chrono::duration_cast<std::chrono::milliseconds>(after_ml - before_ml)).count();
				LOGGER_INFO("Rendering Thread: ML Entire workflow took " << ml_workflow_time << " ms");
				write_debug_gpu(ml_destination, img_size.width, img_size.height, samples_per_pixel, "ML_display_1.tif", dont_debug);
				//Step 2: Apply ML Filer (aka write the file form the GUI)
				ml_filter(ml_destination, static_cast<frame_size>(img_size));
				write_debug_gpu(ml_destination, img_size.width, img_size.height, samples_per_pixel, "ML_display_2.tif", dont_debug);
				//Step 3 : Load into 8 bit OGL buffer
				GLubyte* ogl_device_ptr;
				size_t cuda_buffer_size;//Perhaps because you might not know the OpenGL size?
				CUDASAFECALL(cudaGraphicsMapResources(1, &ml_resource_));
				CUDASAFECALL(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ogl_device_ptr), &cuda_buffer_size, ml_resource_));
				const auto buffer_matches_expected_size = cuda_buffer_size == img_size.n() * samples_per_pixel * sizeof(unsigned char);
				if (!buffer_matches_expected_size)
				{
					qli_runtime_error("Oh Nope");
				}
				const auto ml_display_setting = display_settings(img_d.ml_lut, img_d.ml_display_range);
				write_debug_gpu(ml_destination, img_size.width, img_size.height, samples_per_pixel, "ML_display_3.tif", true);
				write_debug_gpu(ogl_device_ptr, img_size.width, img_size.height, samples_per_pixel, "ML_display_3_8_bit.tif", true);
				comp_->move_clamp_and_scale(ogl_device_ptr, ml_destination, img_size, 1, ml_display_setting.ranges);
				write_debug_gpu(ml_destination, img_size.width, img_size.height, samples_per_pixel, "ML_display_4.tif", true);
				write_debug_gpu(ogl_device_ptr, img_size.width, img_size.height, samples_per_pixel, "ML_display_4_8_bit.tif", true);
				CUDASAFECALL(cudaGraphicsUnmapResources(1, &ml_resource_));
				CUDA_DEBUG_SYNC();
				if (resized)
				{
					// debug
					write_debug_gpu(this->ml_transform->phase_resized, img_size.width, img_size.height, samples_per_pixel, "With_ML_display_GUI_resized.tif", true);
					auto img_vector = this->ml_transform->phase_resized;
					std::cout << "Size comparison " << img_vector.size() << " ==?" << img_d.width << ". " << img_d.height << std::endl;
					return thrust::raw_pointer_cast(this->ml_transform->phase_resized.data());
				}
				write_debug_gpu(img_d.img, img_d.width, img_d.height, samples_per_pixel, "With_ML_display_GUI.tif", true);
				return img_d.img;
			}
#endif
			write_debug_gpu(img_d.img, img_d.width, img_d.height, samples_per_pixel, "Without_ML_display_GUI.tif", true);
			return img_d.img;
		}();
		//Move to OGL + Calculate Histogram
		{
			move_to_old_and_calculate_histogram(is_live, img_pointer, img_d.processing);
		}
		//write_debug_gpu(img_d.img,img_d.width,img_d.height,img_d.samples_per_pixel,"Test.tif",true);
		//
		//so we can do the segmentation now, or maybe free up the resource for a while?
		GLint* label_ptr = nullptr;
		{
			const auto ogl_program_enum =  ogl_program::lut_lookup ;
			if (ogl_program_enum == ogl_program::segmentation_overlay)
			{
				qli_not_implemented();
			}
			else
			{
				selected_label = -1;
			}
			//Value Under Mouse Label
			{
				current_mouse_coordinates_in_global = QCursor::pos();
				const auto current_mouse_coordinates_local = mapFromGlobal(current_mouse_coordinates_in_global);
				const auto current_position_on_sample = to_sample_coordinates(current_mouse_coordinates_local, { scroll_offset_width ,scroll_offset_height }, img_size.digital_scale);
				const auto bounds = QRectF(0, 0, img_size.width - 1, img_size.height - 1);
				const int x_pos = std::round(current_position_on_sample.x());
				const int y_pos = std::round(current_position_on_sample.y());
				phase_value_under_cursor = no_phase_value_under_cursor;
				if (bounds.contains(QPoint(x_pos, y_pos)))
				{
					const auto unwrapped_index = x_pos + y_pos * img_size.width * samples_per_pixel;
					const auto src_ptr = img_pointer + unwrapped_index;
					CUDA_DEBUG_SYNC();
					CUDASAFECALL(cudaMemcpy(phase_value_under_cursor.data(), src_ptr, sizeof(float) * samples_per_pixel, cudaMemcpyDeviceToHost));
				}
			}
			//DPM
			dpm_overpaint_settings = dpm != nullptr ? *dpm : dpm_settings();
		}
		OGL_ERROR_CHECK();
		m_context_->doneCurrent();
		process_messages(img_d, label_ptr, msg);
	}
	render_later();
}

void render_widget::setup_viewport()
{
	const auto scale = img_size.digital_scale;
	const auto  expected_width = ceil(img_size.width * scale);
	const auto  expected_height = ceil(img_size.height * scale);
	const auto  shift_up = scroll_offset_height + height() - (1 + 1) * expected_height / 2;
	const auto  shift_left = -scroll_offset_width;
	const auto fix = [&](auto value) {return devicePixelRatio() * value; };
	glViewport(fix(shift_left), fix(shift_up), fix(expected_width), fix(expected_height)); OGL_ERROR_CHECK();
	last_viewport = { shift_left , shift_up, expected_width, expected_height };
}

void render_widget::update_translation(const GLuint program_handle)
{
	const auto  move_me = glGetUniformLocation(program_handle, "moveMe"); OGL_ERROR_CHECK();
	if (move_me == -1)
	{
		qli_invalid_arguments();
	}
	std::array<GLfloat, 16>  trans =
	{ 1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		digital_x, digital_y, 0.0f, 1.0f };
	glUniformMatrix4fv(move_me, 1, GL_FALSE, trans.data()); OGL_ERROR_CHECK();
}

void render_widget::update_image_size(const GLuint program_handle)
{
	const auto  width_id = glGetUniformLocation(program_handle, "width"); OGL_ERROR_CHECK();
	glUniform1f(width_id, img_size.width); OGL_ERROR_CHECK();
	const auto  height_id = glGetUniformLocation(program_handle, "height"); OGL_ERROR_CHECK();
	glUniform1f(height_id, img_size.height); OGL_ERROR_CHECK();
	const auto  cross_id = glGetUniformLocation(program_handle, "show_cross"); OGL_ERROR_CHECK();
	glUniform1f(cross_id, render_settings_.show_crosshair); OGL_ERROR_CHECK();
}

void render_widget::render()
{
	OGL_ERROR_CHECK();//checks if the context is actually created
	if (img_size.n() == 0)
	{
		return;
	}
	setup_viewport();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); OGL_ERROR_CHECK();
	//Set Program
	auto ogl_program_enum = [&]
	{
		if (render_settings_.ml_remapper_type != ml_remapper_file::ml_remapper_types::off)
		{
			return ogl_program::ml_transform_overlay;
		}
		else
		{
			return  ogl_program::lut_lookup;
		}
	}();
	const auto program_changed = ogl_program_enum != previous_program_;
	const auto  program_handle = ogl_program_lookup_.at(std::make_pair(ogl_program_enum, samples_per_pixel));
	glUseProgram(program_handle); OGL_ERROR_CHECK();
	update_translation(program_handle); OGL_ERROR_CHECK();
	update_image_size(program_handle);
	//DMA Phase Texture
	glActiveTexture(GL_TEXTURE0); OGL_ERROR_CHECK();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, img_buffer_); OGL_ERROR_CHECK();
	const auto  phase_id = glGetUniformLocation(program_handle, "img"); OGL_ERROR_CHECK();
	glUniform1i(phase_id, 0); OGL_ERROR_CHECK();
	glBindTexture(GL_TEXTURE_2D, img_texture_); OGL_ERROR_CHECK();
	const auto  gl_format = samples_per_pixel == 3 ? GL_RGB : GL_RED;
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img_size.width, img_size.height, gl_format, GL_UNSIGNED_BYTE, nullptr); OGL_ERROR_CHECK();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); OGL_ERROR_CHECK();
	//DMA Label Texture
	const auto overlay_program = ogl_program_enum == ogl_program::segmentation_overlay;
	const auto transform_program = ogl_program_enum == ogl_program::ml_transform_overlay;
	if (overlay_program)
	{
		glActiveTexture(GL_TEXTURE1); OGL_ERROR_CHECK();
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, label_buffer_); OGL_ERROR_CHECK();
		const auto  label_id = glGetUniformLocation(program_handle, "labels"); OGL_ERROR_CHECK();
		glUniform1i(label_id, 1); OGL_ERROR_CHECK();//second argument is the texture unit
		glBindTexture(GL_TEXTURE_2D, label_texture_); OGL_ERROR_CHECK();
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img_size.width, img_size.height, GL_RED_INTEGER, GL_INT, nullptr); OGL_ERROR_CHECK();
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); OGL_ERROR_CHECK();
		//
		const auto  max_label_id = glGetUniformLocation(program_handle, "max_label"); OGL_ERROR_CHECK();
		const auto  max_label_f = static_cast<float>(max_label);//ie -1 (bg), 0,1,2
		glUniform1f(max_label_id, max_label_f); OGL_ERROR_CHECK();
		//
		//No race condition because drawing holds a mutex
		//Also this happens on the Qt GUI thread (so should be safe)
		const auto  label_to_highlight = click_coordinates.isNull() ? -1 : selected_label;
	}
	else if (transform_program)
	{
		glActiveTexture(GL_TEXTURE2); OGL_ERROR_CHECK();
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, remapper_buffer_); OGL_ERROR_CHECK();
		const auto  ml_transform_id = glGetUniformLocation(program_handle, "img_transformed"); OGL_ERROR_CHECK();
		glUniform1i(ml_transform_id, 2); OGL_ERROR_CHECK();//second argument is the texture unit
		glBindTexture(GL_TEXTURE_2D, remapper_texture_); OGL_ERROR_CHECK();
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img_size.width, img_size.height, GL_RED, GL_UNSIGNED_BYTE, nullptr); OGL_ERROR_CHECK();
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); OGL_ERROR_CHECK();
	}
	//Main LUT
	const auto activate_lut = [&](lut_texture_sampler& lut, const int lut_in)
	{
		glActiveTexture(lut.texture_unit); OGL_ERROR_CHECK();
		glBindTexture(GL_TEXTURE_1D, lut.texture); OGL_ERROR_CHECK();
		if (lut.old_lut_idx != lut_in || program_changed)
		{
			const auto  src = static_cast<const GLvoid*>(render_settings::luts[lut_in].data.data());//Every time? Todo cache this
			glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 256, GL_RGB, GL_UNSIGNED_BYTE, src); OGL_ERROR_CHECK();
			lut.old_lut_idx = lut_in;
		}
		const auto  lut_id = glGetUniformLocation(program_handle, lut.program_id.c_str());
		glUniform1i(lut_id, lut.texture_unit_number); OGL_ERROR_CHECK();//second argument is the texture unit #3
		glBindSampler(lut.texture_unit_number, lut.sampler); OGL_ERROR_CHECK();
	};
	{
		const auto kill_phase_lut = transform_program && render_settings_.ml_display_mode == ml_remapper::display_mode::only_remap;
		const auto lut_to_use = kill_phase_lut ? display_settings::blank_lut : render_settings_.display_lut;
		activate_lut(img_lut, lut_to_use);
	}
	if (transform_program)
	{
		const auto kill_overlay_lut = render_settings_.ml_display_mode == ml_remapper::display_mode::only_phase;
		const auto lut_to_use = kill_overlay_lut ? display_settings::blank_lut : render_settings_.ml_lut;
		activate_lut(remapper_lut, lut_to_use);
	}
	//Draw
	glBindVertexArray(vao_handle_); OGL_ERROR_CHECK();
	glDrawArrays(GL_TRIANGLES, 0, 8); OGL_ERROR_CHECK();
	glBindVertexArray(0); OGL_ERROR_CHECK();
	//Unbind (important because we might want to use Qt to overpaint
	//
	const auto unbind_lut = [&](const lut_texture_sampler& sampler)
	{
		glActiveTexture(sampler.texture_unit); OGL_ERROR_CHECK();
		glBindSampler(sampler.texture_unit_number, 0); OGL_ERROR_CHECK();
		glBindTexture(GL_TEXTURE_1D, 0); OGL_ERROR_CHECK();
	};
	unbind_lut(img_lut);
	if (transform_program)
	{
		unbind_lut(remapper_lut);
	}
	//
	if (overlay_program)
	{
		glActiveTexture(GL_TEXTURE1); OGL_ERROR_CHECK();
		glBindTexture(GL_TEXTURE_2D, 0); OGL_ERROR_CHECK();
	}
	else if (transform_program)
	{
		glActiveTexture(GL_TEXTURE2); OGL_ERROR_CHECK();
		glBindTexture(GL_TEXTURE_2D, 0); OGL_ERROR_CHECK();
	}
	glActiveTexture(GL_TEXTURE0); OGL_ERROR_CHECK();
	glBindTexture(GL_TEXTURE_2D, 0); OGL_ERROR_CHECK();
	glUseProgram(0); OGL_ERROR_CHECK();
	//QT's OGL overpaint
	const auto  paint_dpm = dpm_overpaint_settings.dpm_phase_is_complete() || dpm_overpaint_settings.dpm_amp_is_complete();
	if (overlay_program || paint_dpm)
	{
		if (!m_device_)
		{
			m_device_ = new QOpenGLPaintDevice;
		}
		m_device_->setSize(size());
		const auto  paint_outline = !click_coordinates.isNull();
		QPainter painter(m_device_);
		if (overlay_program && paint_outline)
		{
			//Draw Anchor Point
			{
				const auto as_viewport = to_view_port_coordinates(click_coordinates, { scroll_offset_width, scroll_offset_height }, img_size.digital_scale);
				QRectF point_w(0, 0, 10, 2);
				point_w.moveCenter(as_viewport);
				painter.fillRect(point_w, Qt::red);
				QRectF point_h(0, 0, 2, 10);
				point_h.moveCenter(as_viewport);
				painter.fillRect(point_h, Qt::red);
			}
			if (selected_label >= 0)
			{
				qli_not_implemented();
				//box drawing used to go here...
			}
		}
		if (paint_dpm)
		{
			{
				const auto  as_viewport_phase = to_view_port_coordinates(QPointF(dpm_overpaint_settings.dpm_phase_left_column, dpm_overpaint_settings.dpm_phase_top_row), { scroll_offset_width, scroll_offset_height }, img_size.digital_scale);
				const auto  dpm_width_phase = dpm_overpaint_settings.dpm_phase_width * img_size.digital_scale;
				const QRectF qbox(as_viewport_phase, QSizeF(dpm_width_phase, dpm_width_phase));
				const QColor color(91, 44, 111, 100);
				painter.fillRect(qbox, color);
			}
			{
				const auto  as_viewport_amp = to_view_port_coordinates(QPointF(dpm_overpaint_settings.dpm_amp_left_column, dpm_overpaint_settings.dpm_amp_top_row), { scroll_offset_width, scroll_offset_height }, img_size.digital_scale);
				const auto  dpm_width_amp = dpm_overpaint_settings.dpm_amp_width * img_size.digital_scale;
				const QRectF qbox(as_viewport_amp, QSizeF(dpm_width_amp, dpm_width_amp));
				const QColor color(91, 44, 111, 100);
				painter.fillRect(qbox, color);
			}
		}
	}
	//Draw Tooltip (this isn't an overpaint)
	show_tooltip();
}
