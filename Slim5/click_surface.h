#pragma once
#ifndef CLICK_SURFACE_H
#define CLICK_SURFACE_H

#include <QGraphicsView>
// ReSharper disable once CppInconsistentNaming
class QMouseEvent;

class click_surface final : public QGraphicsView
{
	Q_OBJECT
public:
	explicit click_surface(QWidget* parent);
	void mousePressEvent(QMouseEvent* e) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void wheelEvent(QWheelEvent* event) override;
signals:
	void move_to(QPointF loc, QGraphicsItem* item = nullptr);
	void clicked_graphics_item(QGraphicsItem* item);
};

#endif