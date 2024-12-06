#pragma once
#ifndef RENDER_WIDGET_H
#define RENDER_WIDGET_H

//#include <QOpenGLWidget>
#include <QWindow>
#include "gui_message.h"
//
#include <memory>
#include <mutex>
#include <QOpenGLFunctions_4_3_Core>
#include <QTimer>
#include "render_engine.h"
#include "histogram_info.h"
#include <thrust/device_vector.h>

// ReSharper disable once CppInconsistentNaming
struct cudaGraphicsResource;
class compute_engine;
class qli_segmentation_wrapper;
class ml_transformer;
class radial_spectrum_widget;

enum class ogl_program { unspecified, lut_lookup, segmentation_overlay, ml_transform_overlay };

class render_widget final : public QWindow, public render_engine, protected QOpenGLFunctions_4_3_Core
{

	Q_OBJECT

signals:
	void load_histogram();
	void load_auto_contrast_settings(const display_settings::display_ranges& range);
	void write_log(const QString& message) const;
	void first_show();
	void frame_size_changed(const frame_size& new_size);
	void key_press_event_external(QKeyEvent* event);
	void wheel_event_external(QWheelEvent* event);
	void move_slider(const QPointF& new_width_height);
	void show_tooltip_value(const QPoint& global_coordinate, const QString& text);

public:
	QPointF click_coordinates;
	QPoint current_mouse_coordinates_in_global;//actually contains the old value, lol
	float selected_value;
	int selected_label;
	const static std::array<float, 3> no_phase_value_under_cursor;
	std::array<float, 3> phase_value_under_cursor;
	render_dimensions img_size;
	int samples_per_pixel;

	float pixel_ratio;
	void render();
	void setup_viewport();
	void update_translation(GLuint program_handle);
	void update_image_size(GLuint program_handle);
	std::mutex ogl_context_mutex;//todo maybe regular mutex? Or is the call order too confusing?
	GLfloat digital_x, digital_y;//easier then writing getters or setters
	//organize this more maybe?
	std::mutex histogram_m;
	histogram_info histogram;
	explicit render_widget(std::shared_ptr<compute_engine>& compute_engine, QWindow* parent = Q_NULLPTR);
	virtual ~render_widget();
	//Viewport Stuff
	std::array<float, 4> last_viewport;
	[[nodiscard]] std::pair<render_dimensions, std::array<qreal, 2>> get_zoom_under_pointer_dimension(float new_scale) const;
	static QPointF to_sample_coordinates(const QPointF& widget_local_pos, const std::array<float, 2>& scroll, float scaling);
	static QPointF to_view_port_coordinates(const QPointF& image_pos, const std::array<float, 2>& scroll, float scaling);
	//
	//
	thrust::device_vector<float> buffer;
	void paint_surface(bool is_live, const camera_frame<float>& img_d, const gui_message& msg, const dpm_settings* dpm) override;

	dpm_settings dpm_overpaint_settings;
	float scroll_offset_width, scroll_offset_height;
	frame_size max_viewport;
	//
	int max_label;
	bool dragging;
	QPointF last_point;
	void calc_histogram(const display_settings::display_ranges& expected_range, histogram_info& histogram_to_fill, unsigned char* out_d_8_bit_img, const float* input_image, const frame_size& frame_size, int samples_per_pixel);
	bool drag_to(const QPointF& local_position);
	void show_tooltip();
	void move_to_old_and_calculate_histogram(bool is_live, const float* img_d, phase_processing processing);
	typedef std::function<void(const float*, frame_size)> ml_filter_function;
	ml_filter_function ml_filter;
	//
public slots:
	void render_later();
	void render_now();

protected:
	bool mouse_inside_;
	//	
	void showEvent(QShowEvent* event) override;
	bool event(QEvent* event) override;
	void exposeEvent(QExposeEvent* event) override;
	void wheelEvent(QWheelEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void focusOutEvent(QFocusEvent* event) override;

private:
	std::vector<GLubyte> i_dont_remember_why_this_is_here;
	std::vector<GLint> i_dont_remember_why_this_is_here2;
	std::map<std::pair<ogl_program, int>, GLuint> ogl_program_lookup_;
	render_settings render_settings_;
	ogl_program previous_program_;
	bool m_update_pending_;
	bool first_show_;
	QOpenGLContext* m_context_;
	QOpenGLPaintDevice* m_device_;
	//
	std::shared_ptr<compute_engine> comp_;
	static std::unique_ptr<ml_transformer> ml_transform;
	GLuint init_program(const QString& fragment_shader);
	void init_render_invariants(const camera_frame<float>& img_d);
	void init_render_invariants();
	void init_shaders();
	GLuint vao_handle_;
	std::array<GLuint, 2> vbo_handles_;

	cudaGraphicsResource* phase_resource_, * label_resource_, * ml_resource_;
	GLuint img_buffer_, label_buffer_, remapper_buffer_;//rename something like PBO
	GLuint img_texture_, label_texture_, remapper_texture_;
	struct lut_texture_sampler
	{
		GLuint texture, sampler;
		GLenum texture_unit;
		int texture_unit_number;
		int old_lut_idx;
		std::string program_id;
		lut_texture_sampler() : lut_texture_sampler(GL_TEXTURE3, 0, "") {};
		lut_texture_sampler(const GLenum texture_unit, const  int texture_unit_number, const std::string& program_id) :texture(0), sampler(0), texture_unit(texture_unit), texture_unit_number(texture_unit_number), old_lut_idx(0), program_id(program_id) {}
	};
	lut_texture_sampler img_lut, remapper_lut;
	GLuint vertex_shader_;
	//
	//todo should be std::vector of std::array
	//
#define OGL_ERROR_CHECK() this->ogl_error_check(__FILE__,__LINE__)
	void ogl_error_check(const char* file, int line);
#define OGL_SHADER_CHECK(sh) this->ogl_shader_check(sh,__FILE__,__LINE__)
	void ogl_shader_check(GLint sh, const char* file, int line);
#define OGL_PROGRAM_CHECK(ph) this->ogl_program_check(ph,__FILE__,__LINE__)
	void ogl_program_check(GLint ph, const char* file, int line);
	//
};

#endif // render_widget_H
