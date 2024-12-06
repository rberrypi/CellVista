#pragma once
#ifndef SCOPE_DELAY_DELEGATE_H
#define SCOPE_DELAY_DELEGATE_H
#include <QStyledItemDelegate>
// ReSharper disable CppInconsistentNaming
class QSpinBox;
// ReSharper restore CppInconsistentNaming


class scope_delay_delegate final : public  QStyledItemDelegate
{
	Q_OBJECT

public:
	explicit scope_delay_delegate(QWidget* parent) : QStyledItemDelegate(parent) { }

	QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
	void setEditorData(QWidget* editor, const QModelIndex& index) const override;
	void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const override;
	void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
};

#endif
