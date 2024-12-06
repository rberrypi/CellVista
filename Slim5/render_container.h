#pragma once
#ifndef RENDER_CONTAINER_H
#define RENDER_CONTAINER_H
#include <QScrollBar>

struct frame_size;
struct render_dimensions;
// ReSharper disable CppInconsistentNaming
class QGridLayout;
class QPushButton;
class QLabel;
// ReSharper restore CppInconsistentNaming

class render_widget;
class place_holder_widget;

class render_container final : public QWidget
{
	Q_OBJECT

		render_widget* render_surface_;
	QScrollBar* vertical_scroll_bar_;
	QScrollBar* horizontal_scroll_bar_;
	QGridLayout* grid_layout_;
	place_holder_widget* fml_;
	QLabel* zoom_level_label_;
	bool block_scrollbar_signals_;//Yes this is thread safe
public:
	explicit render_container(render_widget* render_me, QWidget* parent = Q_NULLPTR);
	void set_scale_bars(const render_dimensions& new_size) const;
	void set_zoom(float value) const;
	const float static zoom_increment;

public slots:
	void fit_scale(const frame_size& new_size) const;
	void zoom_event(bool zoom_in) const;
	void move_slider(const QPointF& new_width_height);
	void show_tool_tip_value(const QPoint& global_coordinate, const QString& value) const;

protected:
	void resizeEvent(QResizeEvent* event) override;
	void wheelEvent(QWheelEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;
	//bool eventFilter(QObject *obj, QEvent *event) override;
};


#endif
