#include "stdafx.h"
#include "render_widget.h"
#include <QToolTip>
#include <QApplication>

bool render_widget::drag_to(const QPointF& local_position)
{
	const auto moved = local_position != last_point;
	const auto safe_bound_check = [](auto min_value, auto current_value, auto max_value)
	{
		auto value = qBound(min_value, current_value, max_value);
		auto outside = (value != current_value);
		return std::make_pair(value, outside);
	};
	if (moved)
	{
		const auto diff = local_position - last_point;
		//
		const auto scroll_end_width = std::max(0.0f, img_size.width * img_size.digital_scale - width());
		const auto w = safe_bound_check(0.0f, static_cast<float>(scroll_offset_width - diff.x()), scroll_end_width);
		scroll_offset_width = w.first;

		const auto scroll_end_height = std::max(0.0f, img_size.height * img_size.digital_scale - height());
		const auto h = safe_bound_check(0.0f, static_cast<float>(scroll_offset_height - diff.y()), scroll_end_height);
		scroll_offset_height = h.first;

		last_point = local_position;
		emit move_slider(QPointF(scroll_offset_width, scroll_offset_height));
	}
	return moved;
}

void render_widget::focusOutEvent(QFocusEvent* event)
{
	QWindow::focusOutEvent(event);
}

void render_widget::show_tooltip()
{
	if (!qIsNaN(phase_value_under_cursor[0]))
	{
		const auto tooltip_precision = 3;
		const auto color_label = [&]
		{
			const auto r = QString::number(phase_value_under_cursor[0], 'f', tooltip_precision);
			const auto g = QString::number(phase_value_under_cursor[1], 'f', tooltip_precision);
			const auto b = QString::number(phase_value_under_cursor[2], 'f', tooltip_precision);
			return QString("%1,%2,%3").arg(r).arg(g).arg(b);
		};
		const auto bw_label = [&]
		{
			return QString::number(phase_value_under_cursor[0], 'f', tooltip_precision);
		};
		const auto text = samples_per_pixel == 3 ? color_label() : bw_label();
		{
			if (isActive())
			{
				auto static last_show = timestamp();
				const auto now = timestamp();
				const auto re_raster = now - last_show >= ms_to_chrono(100);
				if (re_raster)
				{
					//can't call too often or visual glitches
					emit show_tooltip_value(current_mouse_coordinates_in_global, text);
					last_show = now;
				}
			}
		}
	}
}

