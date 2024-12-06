#pragma once
#ifndef LIMITED_SPINNING_BOX_DELEGATE_H
#define LIMITED_SPINNING_BOX_DELEGATE_H
#include <QStyledItemDelegate>
#include <utility>
class limited_spinning_box_delegate final : public  QStyledItemDelegate
{
	Q_OBJECT

		const double min_value_, max_value_;
	const int decimals_;
	bool non_negative_;
	const QString special_text_;
public:

	explicit limited_spinning_box_delegate(QWidget* parent) : limited_spinning_box_delegate(3, false, qQNaN(), qQNaN(), tr(""), parent) {}

	explicit limited_spinning_box_delegate(const int decimals, const bool non_negative, const double min_value, const double max_value, QString special_text, QWidget* parent) : QStyledItemDelegate(parent), min_value_(min_value), max_value_(max_value), decimals_(decimals), non_negative_(non_negative), special_text_(std::move(special_text))
	{

	}

	QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
	void setEditorData(QWidget* editor, const QModelIndex& index) const override;
	void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const override;
	void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
};

#endif
