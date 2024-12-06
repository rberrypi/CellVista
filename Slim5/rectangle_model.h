#pragma once
#ifndef RECTANGLE_MODEL_H
#define RECTANGLE_MODEL_H

// ReSharper disable CppInconsistentNaming
namespace cereal
{
	class JSONOutputArchive;
	class JSONInputArchive;
}

class QStyledItemDelegate;
// ReSharper restore CppInconsistentNaming

#include <QAbstractTableModel>
class rectangle_model : public QAbstractTableModel
{
	Q_OBJECT
public:
	explicit rectangle_model(QObject* parent = Q_NULLPTR) noexcept: QAbstractTableModel(parent)
	{
	}
	virtual QStyledItemDelegate* get_column_delegate(int col, QWidget* parent) = 0;
	virtual void load_xml(cereal::JSONInputArchive& archive) = 0;
	virtual void save_xml(cereal::JSONOutputArchive& archive) const = 0;
	void fill_column(const QVariant& value, int column_index);
	void resize_to(int new_size);
};

#endif