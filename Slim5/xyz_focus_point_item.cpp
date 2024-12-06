#include "stdafx.h"
#include "xyz_focus_point_item.h"
#include <QPainter>
#include <QBrush>
#include "cgal_triangulator.h"
#include "qli_runtime_error.h"
void xyz_focus_point_item::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	const auto label = QString("%1").arg(id_);
	QGraphicsEllipseItem::paint(painter, option, widget);
	if (label.isEmpty()) return;
	const auto rect = boundingRect();
	const QFontMetrics fm(painter->font());
	const auto pad = 0.5;
	const auto sx = rect.width() * pad / fm.horizontalAdvance(label);
	const auto sy = rect.height() * pad / fm.height();
	const auto s = qMin(sx, sy);
	painter->save();
	painter->translate(rect.center());
	painter->scale(s, s);
	painter->translate(-rect.center());
	painter->drawText(rect, label, Qt::AlignHCenter | Qt::AlignVCenter);
	painter->restore();
}

void xyz_focus_point_item::set_verified(const bool verified)
{
	verified_ = verified;
	const auto color = verified ? verified_color() : selected_color();
	const QBrush brush(color);
	setBrush(brush);
}

void xyz_focus_point_item::set_selected(const bool selected)
{
	auto color = selected ? selected_color() : not_selected_color();
	if (selected && verified_) {
		color = verified_color();
	}
	const QBrush brush(color);
	setBrush(brush);
}

bool xyz_focus_point_item::get_verified() const
{
	// const auto color = brush().color();
	// return color == verified_color();
	return verified_;
}

bool xyz_focus_point_item::set_x_center(const qreal x_center)
{
	auto current = this->rect();
	const auto center = current.center();
	const auto current_src = scope_location_xyz(center.x(), center.y(), get_z());
	const auto dst = scope_location_xyz(x_center, current_src.y, current_src.z);
	const auto new_pos = QPointF(x_center, current_src.y);
	current.moveCenter(new_pos);
	this->setRect(current);
	this->update();
	return triangulator_->move_point(current_src, dst);
}

bool xyz_focus_point_item::set_y_center(const qreal y_center)
{
	auto current = this->rect();
	const auto center = current.center();
	const auto current_src = scope_location_xyz(center.x(), center.y(), get_z());
	const auto dst = scope_location_xyz(current_src.x, y_center, current_src.z);
	const auto new_pos = QPointF(center.x(), y_center);
	current.moveCenter(new_pos);
	this->setRect(current);
	this->update();
	return triangulator_->move_point(current_src, dst);
}

scope_location_xyz xyz_focus_point_item::get_serializable() const
{
	const auto center = this->rect().center();
	return scope_location_xyz(center.x(), center.y(), get_z());
}

xyz_focus_point_item::xyz_focus_point_item(cgal_triangulator* triangulation, const int id, const scope_location_xyz& data, const QRectF& rect, QGraphicsItem* parent) :
	QGraphicsEllipseItem(rect, parent), id_(id), verified_(false), triangulator_(triangulation)
{
	triangulator_->insert_point(data);
	auto moved_rect = rect;
	moved_rect.moveCenter(QPointF(0, 0));
	this->setRect(moved_rect);
	set_x_center(data.x);
	set_y_center(data.y);
	set_verified(false);
}

xyz_focus_point_item::~xyz_focus_point_item()
{
	const auto xyz_point = get_serializable();
	triangulator_->remove_point(xyz_point);
}

bool xyz_focus_point_item::set_z_center(const qreal z) const
{
	const auto current_src = this->get_serializable();
	auto new_pos = current_src;
	new_pos.z = z;
	return triangulator_->move_point(current_src, new_pos);
}

qreal xyz_focus_point_item::get_z() const
{
	const auto center = this->rect().center();

#if _DEBUG
	{
		if (triangulator_ == nullptr)
		{
			qli_runtime_error("!!!!");
		}
	}
#endif

	return triangulator_->get_z(scope_location_xy(center.x(), center.y()));
}
