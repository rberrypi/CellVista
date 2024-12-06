#pragma once
#ifndef XYZ_FOCUS_POINT_ITEM
#define XYZ_FOCUS_POINT_ITEM

#define XYZ_MODEL_ITEM_ID_IDX (0)
#define XYZ_MODEL_ITEM_X_IDX (1)
#define XYZ_MODEL_ITEM_Y_IDX (2)
#define XYZ_MODEL_ITEM_Z_IDX (3)
#define XYZ_MODEL_ITEM_VALID_IDX (4)

#include <QGraphicsEllipseItem>
class cgal_triangulator;
#include "instrument_configuration.h"
class xyz_focus_point_item final : public QGraphicsEllipseItem
{
	int id_;
	bool verified_;
	cgal_triangulator* triangulator_;

	static QColor verified_color() noexcept
	{
		return{ 46, 204, 113 }; //green
	}
	static QColor not_selected_color() noexcept
	{
		return{ 128, 128, 128 }; //gray
	}
	static QColor selected_color() noexcept
	{
		return{ 231, 76, 60 }; //red/orange
	}

public:
	xyz_focus_point_item(cgal_triangulator* triangulation, int id, const scope_location_xyz& data, const QRectF& rect, QGraphicsItem* parent = nullptr);
	~xyz_focus_point_item();
	[[nodiscard]] scope_location_xyz get_serializable() const;
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = Q_NULLPTR) override;

	[[nodiscard]] int id() const noexcept
	{
		return id_;
	}
	void set_id(const int id)
	{
		id_ = id;
		update();
	}
	void set_verified(bool verified);
	void set_selected(bool selected);
	[[nodiscard]] bool get_verified() const;

	[[nodiscard]] qreal x_center() const
	{
		return this->rect().center().x();
	}

	[[nodiscard]] qreal y_center() const
	{
		return this->rect().center().y();
	}

	[[nodiscard]] bool set_z_center(qreal z) const;
	[[nodiscard]] qreal get_z() const;
	bool set_x_center(qreal x_center);
	bool set_y_center(qreal y_center);

	enum { Type = UserType + 1 };

	[[nodiscard]] int type() const override
	{
		return Type;
	}
};

#endif
