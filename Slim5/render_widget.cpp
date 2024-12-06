#include "stdafx.h"
#include "render_widget.h"
#include <QCoreApplication>
#include <QResource>
#include <QResizeEvent>
#include <iostream>
#include <gl/GLU.h>

#include "qli_runtime_error.h"
#pragma comment(lib,"Glu32.lib")

void render_widget::showEvent(QShowEvent* event)
{
	if (first_show_)
	{
		first_show_ = false;
		emit first_show();
	}
	QWindow::showEvent(event);
}

bool render_widget::event(QEvent* event)
{
	switch (event->type()) {
	case QEvent::UpdateRequest:
		m_update_pending_ = false;
		render_now();
		return true;
	default:
		return QWindow::event(event);
	}
}

void render_widget::exposeEvent(QExposeEvent* event)
{
	Q_UNUSED(event);
	if (isExposed())
	{
		render_now();
	}
}

void render_widget::render_later()
{
	if (!m_update_pending_) {
		m_update_pending_ = true;
		QCoreApplication::postEvent(this, new QEvent(QEvent::UpdateRequest));
	}
}

GLuint render_widget::init_program(const QString& fragment_shader)
{
	const QResource resource(fragment_shader);
	if (!resource.isValid() || resource.size() < 1)
	{
		qli_runtime_error("OGL Program error");
	}
	QByteArray byte_array(reinterpret_cast<const char*>(resource.data()), resource.size());
	auto* as_gl_char = static_cast<const GLchar*>(byte_array.data());
	const auto fs_h = glCreateShader(GL_FRAGMENT_SHADER); OGL_ERROR_CHECK();
	glShaderSource(fs_h, 1, &as_gl_char, nullptr); OGL_ERROR_CHECK();
	glCompileShader(fs_h); OGL_ERROR_CHECK();
	OGL_SHADER_CHECK(fs_h); OGL_ERROR_CHECK();
	const auto program_id = glCreateProgram(); OGL_ERROR_CHECK();
	glAttachShader(program_id, vertex_shader_); OGL_ERROR_CHECK();
	glAttachShader(program_id, fs_h); OGL_ERROR_CHECK();
	glLinkProgram(program_id); OGL_ERROR_CHECK();
	OGL_PROGRAM_CHECK(program_id);
	return program_id;
}

void render_widget::ogl_program_check(const GLint ph, const char* file, const int line)
{
	GLint status;
	glGetProgramiv(ph, GL_LINK_STATUS, &status);
	if (GL_FALSE == status)
	{
		GLint log_len;
		glGetProgramiv(ph, GL_INFO_LOG_LENGTH, &log_len);
		if (log_len > 0)
		{
			std::vector<char> log(log_len);
			auto written = static_cast<GLsizei>(0);
			glGetProgramInfoLog(ph, log_len, &written, log.data());
			std::cout << "Program Error:" << file << ":" << line << ": " << log.data() << std::endl;
		}
		qli_runtime_error("OGL program creation failed, this is fixed on the S3 Savage 2000");
	}
}

void render_widget::ogl_shader_check(const GLint sh, const char* file, const int line)
{
	GLint status;
	glGetShaderiv(sh, GL_COMPILE_STATUS, &status);
	if (GL_FALSE == status)
	{
		GLint log_len;
		glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &log_len);
		if (log_len > 0)
		{
			std::vector<char> log(log_len);
			auto written = static_cast<GLsizei>(0);
			glGetShaderInfoLog(sh, log_len, &written, log.data());
			std::cout << "Shader Error:" << file << ":" << line << ": " << log.data() << std::endl;
		}
		qli_runtime_error("OGL shader creation failed, this is fixed on the S3 Savage 2000");
	}
}


void render_widget::init_shaders()
{
	//Reserve the textures and make the shaders
	const auto* vector_shader_source =
		"#version 120\n"	//GL version not same as GLSL version, good work team
		"attribute vec2 vecPosIn;\n"
		"attribute vec4 texPosIn;\n"
		"uniform mat4 moveMe;\n"
		"varying vec4 texPosOut;\n"
		"void main()\n"
		"{\n"
		"gl_Position = moveMe*vec4(vecPosIn* vec2(1.0, -1.0), 0.0, 1.0);\n"
		"texPosOut = texPosIn;\n"
		"}\n";
	vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader_, 1, &vector_shader_source, nullptr);
	glCompileShader(vertex_shader_);
	OGL_SHADER_CHECK(vertex_shader_);
	OGL_ERROR_CHECK();
	//
	ogl_program_lookup_.insert_or_assign({ ogl_program::lut_lookup,1 }, init_program(":/ogl_shaders/lut_grayscale.glsl"));
	ogl_program_lookup_.insert_or_assign({ ogl_program::lut_lookup,3 }, init_program(":/ogl_shaders/lut_color.glsl"));
	ogl_program_lookup_.insert_or_assign({ ogl_program::segmentation_overlay,1 }, init_program(":/ogl_shaders/segmentation_overlay.glsl"));
	ogl_program_lookup_.insert_or_assign({ ogl_program::ml_transform_overlay,1 }, init_program(":/ogl_shaders/ml_transform_overlay.glsl"));
}


// ReSharper disable once CppMemberFunctionMayBeStatic
// ReSharper disable CppMemberFunctionMayBeConst
void render_widget::ogl_error_check(const char* file, const int line)
// ReSharper restore CppMemberFunctionMayBeConst
{
	Q_UNUSED(file); Q_UNUSED(line);
#if _DEBUG
	const auto current_error = glGetError();
	if (current_error == GL_NO_ERROR)
	{
		return;
	}
	const auto err_string = gluErrorString(current_error);
	std::cout << "GLStr Error:" << file << ":" << line << ": Code:" << current_error;
	if (err_string != nullptr)
	{
		std::cout << " MSG:" << err_string;
	}
	std::cout << std::endl;
	qli_runtime_error("OpenGL Error");
#endif
}

void render_widget::wheelEvent(QWheelEvent* event)
{
	//Yes this is required
	if (!dragging)
	{
		emit wheel_event_external(event);
	}
	QWindow::wheelEvent(event);
}

void render_widget::keyPressEvent(QKeyEvent* event)
{
	if (!dragging)
	{
		emit key_press_event_external(event);
	}
	QWindow::keyPressEvent(event);
}

QPointF render_widget::to_sample_coordinates(const QPointF& widget_local_pos, const std::array<float, 2>& scroll, const float scaling)
{
	const auto sample_coordinates = QPointF((scroll[0] + widget_local_pos.x()) / scaling, (scroll[1] + widget_local_pos.y()) / scaling);
	return sample_coordinates;
}

std::pair<render_dimensions, std::array<qreal, 2>> render_widget::get_zoom_under_pointer_dimension(const float new_scale) const
{
	render_dimensions new_size(img_size, new_scale);
	{
		const auto fuck_it_don_t_divide_by_zero_okay = [](const int val) {return std::max(val, 1); };
		const auto max_scale = std::min(max_viewport.width / fuck_it_don_t_divide_by_zero_okay(new_size.width), max_viewport.height / fuck_it_don_t_divide_by_zero_okay(new_size.height));
		if (new_size.digital_scale > max_scale)
		{
			new_size.digital_scale = max_scale;
		}
	}//
	const auto global_position = QCursor::pos();
	const auto local_position = mapFromGlobal(global_position);
	const auto current_position_on_sample = to_sample_coordinates(local_position, { scroll_offset_width ,scroll_offset_height }, img_size.digital_scale);
	const auto scaled_position_on_viewport = to_view_port_coordinates(current_position_on_sample, { scroll_offset_width , scroll_offset_height }, new_size.digital_scale);
	const auto current_position_on_viewport = to_view_port_coordinates(current_position_on_sample, { scroll_offset_width , scroll_offset_height }, img_size.digital_scale);
	const auto difference_in_viewport_pixels = scaled_position_on_viewport - current_position_on_viewport;
	std::array<qreal, 2> difference = { difference_in_viewport_pixels.x(), difference_in_viewport_pixels.y() };
	return std::make_pair(new_size, difference);
}

QPointF render_widget::to_view_port_coordinates(const QPointF& image_pos, const std::array<float, 2>& scroll, const float scaling)
{
	const auto viewport_coordinates = QPointF(scaling * image_pos.x() - scroll.at(0), scaling * image_pos.y() - scroll.at(1));
	return viewport_coordinates;
}

void render_widget::mouseDoubleClickEvent(QMouseEvent* event)
{
	QWindow::mouseDoubleClickEvent(event);
}

void render_widget::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton)
	{
		setCursor(Qt::ClosedHandCursor);
		last_point = event->localPos();
		dragging = true;
		//
		click_coordinates = to_sample_coordinates(last_point, { scroll_offset_width ,scroll_offset_height }, img_size.digital_scale);
	}
	QWindow::mousePressEvent(event);
}

void render_widget::mouseMoveEvent(QMouseEvent* event)
{
	if (event->buttons() == Qt::LeftButton && dragging)
	{
		const auto mouse_pos_better = event->localPos();
		const auto pointer_type = drag_to(mouse_pos_better);
		setCursor(pointer_type ? Qt::ClosedHandCursor : Qt::ArrowCursor);
	}
	QWindow::mouseMoveEvent(event);
}

void render_widget::mouseReleaseEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton && dragging)
	{
		const auto mouse_pos_better = event->localPos();
		drag_to(mouse_pos_better);
		setCursor(Qt::ArrowCursor);
		dragging = false;
	}
	QWindow::mouseReleaseEvent(event);
}

void render_widget::render_now()
{
	//
	if (!isExposed())
	{
		return;
	}

	auto needs_initialize = false;

	{
		std::lock_guard<std::mutex> lk(ogl_context_mutex);
		if (!m_context_) {
			m_context_ = new QOpenGLContext(this);
			m_context_->setFormat(requestedFormat());
			const auto success = m_context_->create();
			if (!success)
			{
				return;
			}
			needs_initialize = true;
		}

		const auto success = m_context_->makeCurrent(this);
		if (success)
		{
			if (needs_initialize)
			{
				initializeOpenGLFunctions();
				init_shaders();
				{
					GLint viewport_max[4];
					glGetIntegerv(GL_MAX_VIEWPORT_DIMS, viewport_max);
					max_viewport.width = viewport_max[0];
					max_viewport.height = viewport_max[1];
				}
			}

			render();

			m_context_->swapBuffers(this);
			m_context_->doneCurrent();
		}
	}
}