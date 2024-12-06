#pragma once
#ifndef TOTAL_CONTROL_ROI_MODEL_H
#define TOTAL_CONTROL_ROI_MODEL_H
#include "qli_cca_shared.h"
#include <QFile>
#include <QAbstractTableModel>

class segmentation_model final : public QAbstractTableModel
{
	Q_OBJECT

	inline void debug_size_check() const;
	int last_issued_size_;
public:
	//
	std::vector<int> area;
	std::vector<float> mass;
	std::vector<bounding_box> bounding_boxes;
	std::vector<float> solidities;
	void issue_update(int select_label);
	void resize_wrapper(size_t elements);
	//
	explicit segmentation_model(QObject* parent = Q_NULLPTR) : QAbstractTableModel(parent), last_issued_size_(0) {}
	void drop_all();
	[[nodiscard]] int rowCount(const QModelIndex & = QModelIndex() /*parent*/) const override;
	[[nodiscard]] int columnCount(const QModelIndex & = QModelIndex() /*parent*/) const override;
	[[nodiscard]] QVariant data(const QModelIndex& index, int role) const override;
	[[nodiscard]] QVariant data(int row, int col) const;
	bool set_data(int row, int col, const QVariant& v);
	bool setData(const QModelIndex& index, const QVariant& value, int role = Qt::EditRole) override;
	[[nodiscard]] Qt::ItemFlags flags(const QModelIndex& /*index*/) const override;
	bool insertRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
	bool insertColumns(int column, int count, const QModelIndex& parent = QModelIndex()) override;
	bool removeRows(int row, int count, const QModelIndex& parent = QModelIndex()) override;
	[[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
signals:
	void change_index_to(int new_row);
	void save_to_file(const QFile& filename);
};


#endif
