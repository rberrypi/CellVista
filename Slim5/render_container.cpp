#include "stdafx.h"
#include "render_container.h"
#include "render_widget.h"
#include <QScrollBar>
#include <QVBoxLayout>
#include "place_holder_widget.h"
#include <QPushButton>
#include <QWheelEvent>
#include <qmath.h>
#include <QApplication>
#include <QLabel>
#include <QToolTip>

render_container::render_container(render_widget* render_me, QWidget* parent) : QWidget(parent), render_surface_(render_me), block_scrollbar_signals_(false)
{
	setFocusPolicy(Qt::FocusPolicy::StrongFocus);
	connect(render_surface_, &render_widget::frame_size_changed, this, &render_container::fit_scale, Qt::QueuedConnection);
	connect(render_surface_, &render_widget::wheel_event_external, this, &render_container::wheelEvent);
	connect(render_surface_, &render_widget::show_tooltip_value, this, &render_container::show_tool_tip_value);
	grid_layout_ = new QGridLayout();
	//Render Surface
	{
		auto window_container = createWindowContainer(render_surface_);
		window_container->setLayout(new QGridLayout());
		fml_ = new place_holder_widget;
		//fml->setContentsMargins(0, 0, 0, 0);
		window_container->layout()->addWidget(fml_);
		grid_layout_->addWidget(window_container, 0, 0);
		grid_layout_->setColumnStretch(0, 1);
	}
	//Horizontal Scroll Bar
	{
		auto bottom_strip = new  QHBoxLayout();
		auto plus_button = new QPushButton;
		plus_button->setIcon(QIcon(":/images/zoom-in.svg"));
		connect(plus_button, &QPushButton::clicked, [&] {zoom_event(true); });
		bottom_strip->addWidget(plus_button);

		zoom_level_label_ = new QLabel("Zoom");
		bottom_strip->addWidget(zoom_level_label_);
		auto minus_button = new QPushButton;
		minus_button->setIcon(QIcon(":/images/zoom-out.svg"));
		connect(minus_button, &QPushButton::clicked, [&] {zoom_event(false); });
		bottom_strip->addWidget(minus_button);
		horizontal_scroll_bar_ = new QScrollBar(Qt::Horizontal);
		horizontal_scroll_bar_->setFocusPolicy(Qt::FocusPolicy::StrongFocus);
		bottom_strip->addWidget(horizontal_scroll_bar_, 1);
		//grid_layout->addLayout(bottom_strip, 1, 0, 1, 2);
		grid_layout_->addLayout(bottom_strip, 1, 0);
		grid_layout_->setRowStretch(0, 1);
		connect(horizontal_scroll_bar_, &QScrollBar::valueChanged, [&](const int value) {
			if (!block_scrollbar_signals_)
			{
				render_surface_->scroll_offset_width = value;
			}
		});
	}
	//
	vertical_scroll_bar_ = new QScrollBar;
	vertical_scroll_bar_->setFocusPolicy(Qt::FocusPolicy::StrongFocus);
	grid_layout_->addWidget(vertical_scroll_bar_, 0, 1);
	connect(vertical_scroll_bar_, &QScrollBar::valueChanged, [&](const int value) {
		if (!block_scrollbar_signals_)
		{
			render_surface_->scroll_offset_height = value;
		}
	});
	connect(render_surface_, &render_widget::move_slider, this, &render_container::move_slider);
	setLayout(grid_layout_);
	//qApp->installEventFilter(this);
}

const float render_container::zoom_increment = 0.1;

void render_container::zoom_event(const bool zoom_in) const
{
	const auto zoom_amount = zoom_in ? zoom_increment : -zoom_increment;
	set_zoom(render_surface_->img_size.digital_scale + zoom_amount);
}

void render_container::set_zoom(const float value) const
{
#if 0
	{
		auto last_click_position = QCursor::pos();
		auto local_coordinates = render_surface->mapFromGlobal(last_click_position);
		auto sample_coordinates = render_widget::toSampleCoordinates(local_coordinates, { render_surface->scroll_offset_width , render_surface->scroll_offset_height }, render_surface->img_size.digital_scale);
		std::cout << "Sample Coordinates " << sample_coordinates.x() << std::endl;
	}
#endif
	const auto zoom_scale = std::max(zoom_increment, value);
	const auto new_state = render_surface_->get_zoom_under_pointer_dimension(zoom_scale);
	auto dimensions = new_state.first;
	if (dimensions.is_valid())
	{
		set_scale_bars(dimensions);
		auto scroll_position = new_state.second;
		//if (scroll_position[0] > 0)
		{
			//scroll_position
			const auto new_scroll_x = std::max(horizontal_scroll_bar_->value() + scroll_position[0], 0.0);
			horizontal_scroll_bar_->setValue(new_scroll_x);
		}
		//if (scroll_position[1] > 0)
		{
			const auto new_scroll_y = std::max(vertical_scroll_bar_->value() + scroll_position[1], 0.0);
			vertical_scroll_bar_->setValue(new_scroll_y);
		}
		//
	}
	//
#if 0
	if (0)
	{
		auto last_click_position = QCursor::pos();
		auto local_coordinates = render_surface->mapFromGlobal(last_click_position);
		auto sample_coordinates = render_widget::toSampleCoordinates(local_coordinates, { render_surface->scroll_offset_width , render_surface->scroll_offset_height }, render_surface->img_size.digital_scale);
		std::cout << "New Sample Coordinates " << sample_coordinates.x() << std::endl << std::endl;
	}
#endif
}
/*
bool render_container::eventFilter(QObject *obj, QEvent *event)
{
	//this should probably consume the events? aka set event->accept() ?
	if (event->type() == QEvent::KeyPress)
	{
		keyPressEvent(static_cast<QKeyEvent *>(event));
	}
	if (event->type() == QEvent::Wheel)
	{

		wheelEvent(static_cast<QWheelEvent *>(event));
	}
	return QWidget::eventFilter(obj, event);
}
*/
void render_container::wheelEvent(QWheelEvent* event)
{
	//todo digital scale should live only in the render widget
	const auto factor = static_cast<float>(qPow(1.2, event->angleDelta().y() / 240.0) * render_surface_->img_size.digital_scale);
	set_zoom(std::max(factor, 0.0f));
	QWidget::wheelEvent(event);
}

void render_container::keyPressEvent(QKeyEvent* event)
{
	const auto key = event->key();
	switch (key)//non-paradigmatic but oh well
	{
	case Qt::Key_Up:
	case Qt::Key_Down:
		vertical_scroll_bar_->event(event);
		break;
	case Qt::Key_Left:
	case Qt::Key_Right:
		horizontal_scroll_bar_->event(event);
		break;
	default://shuts up the static analysis
		break;
	}
	QWidget::keyPressEvent(event);
}

void render_container::set_scale_bars(const render_dimensions& new_size) const
{
	//
	render_surface_->img_size.digital_scale = new_size.digital_scale;
	//
	const int predicted_width = ceil(new_size.digital_scale * new_size.width);
	const int predicted_height = ceil(new_size.digital_scale * new_size.height);
	//so if this is bigger than the maxi viewport we reject the change
	const auto label = QString::number(100.0 * predicted_width / new_size.width, 'f', 2).append("%");
	zoom_level_label_->setText(label);
	auto margins = grid_layout_->contentsMargins();
	const auto margin_width = margins.left() + margins.right();//this logic might be wrong, but the difference is the 15 pixel margin for soe widget...
	fml_->set_size_hint(QSize(predicted_width - margin_width, predicted_height));//maybe not needed, check...
	//
	{
		//render_surface->setWidth(predicted_width);//has no effect
		const auto magic_feel_scale = 50;// I know that feel
		const auto area_size = render_surface_->size();
		vertical_scroll_bar_->setPageStep(area_size.height());
		const auto range_height = predicted_height - area_size.height();
		vertical_scroll_bar_->setRange(0, range_height);
		vertical_scroll_bar_->setSingleStep(std::max(1, range_height / magic_feel_scale));
		horizontal_scroll_bar_->setPageStep(area_size.width());
		const auto range_width = predicted_width - area_size.width();
		horizontal_scroll_bar_->setRange(0, range_width);
		horizontal_scroll_bar_->setSingleStep(std::max(1, range_width / magic_feel_scale));
	}
}

void render_container::resizeEvent(QResizeEvent* event)
{
	auto frame_size = render_surface_->img_size;
	if (frame_size.n() > 0)
	{
		set_scale_bars(frame_size);
	}
	QWidget::resizeEvent(event);
}

void render_container::fit_scale(const frame_size& new_size) const
{
	const auto actual_height = render_surface_->height();
	const auto ratio = static_cast<float>(actual_height) / new_size.height;
	render_surface_->img_size.digital_scale = ratio;
	set_scale_bars(render_dimensions(new_size, ratio));
}

void render_container::move_slider(const QPointF& new_width_height)
{
	//KISS
	block_scrollbar_signals_ = true;
	horizontal_scroll_bar_->setValue(new_width_height.x());
	vertical_scroll_bar_->setValue(new_width_height.y());
	block_scrollbar_signals_ = false;
}

void render_container::show_tool_tip_value(const QPoint& global_coordinate, const QString& value) const
{
	//auto maybe = underMouse();
	const auto active = isActiveWindow();
	const auto local_coordinates = mapFromGlobal(global_coordinate);
	const auto scene_rect = rect();
	static auto call = 0;
	call = (call + 1) % 2;
	//
	if (scene_rect.contains(local_coordinates) && active)
	{
		static QString last_text;
		static QPoint last_point;
		if (last_text == value && global_coordinate != last_point)
		{
			//clear to force display!
			QToolTip::hideText();
		}
		QToolTip::showText(global_coordinate, value);
		last_text = value;
		last_point = global_coordinate;
	}

}