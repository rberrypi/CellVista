#pragma once
#ifndef ROI_MODEL_H
#define ROI_MODEL_H
#include <QGraphicsScene>
#include "rectangle_model.h"
#include "roi_item.h"

typedef std::function<roi_item_serializable()> get_default_serializable_item_functor;

class roi_model final : public rectangle_model
{
	Q_OBJECT

		QGraphicsScene* scene_;
	void reindex_data_view(int start_idx);
	get_default_serializable_item_functor get_default_serializable_item_;

public:
	roi_model(QGraphicsScene* in, const get_default_serializable_item_functor& get_channel_info, QObject* parent = Q_NULLPTR);
	QVector<roi_item*> data_view_;

	[[nodiscard]] int rowCount(const QModelIndex & = QModelIndex()) const override;
	[[nodiscard]] int columnCount(const QModelIndex& parent = QModelIndex()) const override;
	//for editing
	[[nodiscard]] QVariant data(const QModelIndex& index, int role) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
	[[nodiscard]] Qt::ItemFlags flags(const QModelIndex& /*index*/) const override;
	//for resizing//
	bool insertRows(int row_position, int rows, const QModelIndex& index = QModelIndex()) override;
	bool insertColumns(int column, int rows, const QModelIndex& index = QModelIndex()) override;
	bool removeRows(int row_position, int rows, const QModelIndex& index = QModelIndex()) override;

	[[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
	QStyledItemDelegate* get_column_delegate(int col, QWidget* parent) override;
	//
	void load_xml(cereal::JSONInputArchive& archive) override;
	void save_xml(cereal::JSONOutputArchive& archive) const  override;
	//
	void set_serializable_item(const roi_item_serializable& value, int row);
	[[nodiscard]] roi_item_serializable get_serializable_item(int row) const;
	[[nodiscard]] int find_unset_row(int start_row_idx) const;
	[[nodiscard]] QRectF get_bounding_rectangle() const;
	//
	[[nodiscard]] xy_pairs get_xy_focus_points(int idx) const;
	[[nodiscard]] std::vector<float> get_z_for_whole_roi(int roi_idx) const;
	void set_xy_focus_points(int idx, const xy_pairs& xy_focus_points, float displacement_x, float displacement_y);

	void set_zee_for_whole_roi(float value, int idx) const;
	void set_zee_for_whole_roi(const std::vector<float>& z_values, int idx) const;
	void increment_zee_for_whole_roi(float value, int idx) const;

	typedef std::function<bool(int)> channel_filter;
	channel_filter channel_validity_filter;
public slots:
	void update_center(int row, scope_location_xy center);

signals:
	void update_capture_info();
	void updated_channel_info(int row);
	void setup_xyz(int row);
	void row_col_changed();
	void channels_valid(fl_channel_index_list& new_channels);

};
#endif