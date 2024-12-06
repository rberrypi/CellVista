#include "stdafx.h"
#include "click_surface.h"
#include <QToolTip>
#include <QMouseEvent>

click_surface::click_surface(QWidget* parent) :
	QGraphicsView(parent)
{
	setMouseTracking(true);
}

void click_surface::mouseMoveEvent(QMouseEvent* event)
{
	//http://stackoverflow.com/questions/12417636/qt-show-mouse-position-like-tooltip
	const auto pt = mapToScene(event->pos());
	const auto field_width = 8;
	const auto fmt = 'f';
	const auto precision = 2;
	const auto text = QString("%1,%2").arg(pt.x(), field_width, fmt, precision).arg(pt.y(), field_width, fmt, precision);
	QToolTip::showText(event->globalPos(), text, this, rect());
	QGraphicsView::mouseMoveEvent(event);
}

void click_surface::mousePressEvent(QMouseEvent* e)
{
	const auto pt = mapToScene(e->pos());
	if (e->modifiers() & Qt::AltModifier)
	{
		const auto item = itemAt(e->pos());
		if (item)
		{
			emit move_to(pt, item);
		}
		else {
			emit move_to(pt);
		}
	}
	else
	{
		const auto item = itemAt(e->pos());
		if (item)
		{
			emit clicked_graphics_item(item);
		}
	}
	//
	QGraphicsView::mousePressEvent(e);
}

void click_surface::wheelEvent(QWheelEvent*)
{
}