#pragma once
#ifndef XYZ_FOCUS_POINTS_MODEL_H
#define XYZ_FOCUS_POINTS_MODEL_H

#include "grid_steps.h"
#include "rectangle_model.h"
#include "xyz_focus_point_item.h"
#include <boost/core/noncopyable.hpp>
#include "instrument_configuration.h"
class roi_item;
class xyz_focus_points_model final : public rectangle_model, boost::noncopyable
{
	Q_OBJECT

		void reindex_data_view(int start_idx);
	float default_width_, default_height_;
	//bool is_model_selected_;

public:
	roi_item* roi_ptr;
	QVector<xyz_focus_point_item*> data_view_;

	void update_four_points(const grid_steps& steps, const scope_location_xy& center);
	const static auto min_rows = 4;

	explicit xyz_focus_points_model(roi_item* scene, QObject* parent = Q_NULLPTR);
	~xyz_focus_points_model();

	[[nodiscard]] int rowCount(const QModelIndex & = QModelIndex()) const override;
	[[nodiscard]] int columnCount(const QModelIndex& parent = QModelIndex()) const override;
	//for editing
	[[nodiscard]] QVariant data(const QModelIndex& index, int role) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
	[[nodiscard]] Qt::ItemFlags flags(const QModelIndex& /*index*/) const override;
	//for resizing
	bool insertRows(int row_position, int rows, const QModelIndex& index = QModelIndex()) override;
	bool insertColumns(int column, int count, const QModelIndex& parent = QModelIndex()) override;
	bool removeRows(int row_position, int rows, const QModelIndex& index = QModelIndex()) override;

	[[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
	QStyledItemDelegate* get_column_delegate(int col, QWidget* parent) override;
	[[nodiscard]] int find_unset_row(int start_row_idx) const;
	//
	[[nodiscard]] scope_location_xyz get_focus_point_location(int row) const;
	void set_serializable_point(const scope_location_xyz& value, int row);

	void set_serializable_focus_points(const std::vector<scope_location_xyz>& points);
	[[nodiscard]] std::vector<scope_location_xyz> get_serializable_focus_points() const;
	//
	void load_xml(cereal::JSONInputArchive& archive) override {}
	void save_xml(cereal::JSONOutputArchive& archive) const  override {}
	//
	void set_visible();
	void set_selection_color(bool selected);
	void check_selection_status();

};

#endif
