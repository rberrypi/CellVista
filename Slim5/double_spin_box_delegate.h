#pragma once
#ifndef SPINBOX_PRECISION_DELEGATE_H
#define SPINBOX_PRECISION_DELEGATE_H
#include <QStyledItemDelegate>

//http://www.qtforum.org/article/35255/solved-qtableview-set-column-decimal-places.html
class double_spin_box_delegate final : public QStyledItemDelegate
{
	Q_OBJECT

		const int precision_;
	bool non_negative_;
public:
	explicit double_spin_box_delegate(int precision, bool non_negative, QObject* parent = nullptr);
	QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option,
		const QModelIndex& index) const override;


	void setEditorData(QWidget* editor, const QModelIndex& index) const override;
	void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const override;

	void updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem& option, const QModelIndex& index) const override;

};

#endif