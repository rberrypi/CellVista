#include "stdafx.h"
#include "graphics_view_zoom.h"
#include <QMouseEvent>
#include <QApplication>
#include <QScrollBar>
#include <qmath.h>

graphics_view_zoom::graphics_view_zoom(QGraphicsView* view)
	: QObject(view), view_(view)
{
	view_->viewport()->installEventFilter(this);
	view_->setMouseTracking(true);
	modifiers_ = Qt::ControlModifier;
	zoom_factor_base_ = 1.0015;
}

void graphics_view_zoom::gentle_zoom(const double factor) {
	view_->scale(factor, factor);
	view_->centerOn(target_scene_pos_);
	const auto delta_viewport_pos = target_viewport_pos_ - QPointF(view_->viewport()->width() / 2.0,
		view_->viewport()->height() / 2.0);
	const auto viewport_center = view_->mapFromScene(target_scene_pos_) - delta_viewport_pos;
	view_->centerOn(view_->mapToScene(viewport_center.toPoint()));
	emit zoomed();
}

void graphics_view_zoom::set_modifiers(const Qt::KeyboardModifiers modifiers) {
	modifiers_ = modifiers;

}

void graphics_view_zoom::set_zoom_factor_base(const double value) {
	zoom_factor_base_ = value;
}

bool graphics_view_zoom::eventFilter(QObject* object, QEvent* event) {
	if (event->type() == QEvent::MouseMove) {
		const auto* mouse_event = dynamic_cast<QMouseEvent*>(event);
		const auto delta = target_viewport_pos_ - mouse_event->pos();
		if (qAbs(delta.x()) > 5 || qAbs(delta.y()) > 5) {
			target_viewport_pos_ = mouse_event->pos();
			target_scene_pos_ = view_->mapToScene(mouse_event->pos());
		}
	}
	else if (event->type() == QEvent::Wheel) {
		const auto* wheel_event = dynamic_cast<QWheelEvent*>(event);
		//TODO FIX THIS, check if angle()
		if (QApplication::keyboardModifiers() == modifiers_) 
		{
			//if (wheel_event->orientation() == Qt::Vertical)
			const double angle = wheel_event->angleDelta().y();
			if (angle!=0)
			{
				const auto factor = qPow(zoom_factor_base_, angle);
				gentle_zoom(factor);
				return true;
			}
		}
	}
	Q_UNUSED(object)
		return false;
}