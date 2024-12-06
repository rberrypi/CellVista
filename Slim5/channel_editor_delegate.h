#pragma once
#ifndef CHANNEL_EDITOR_DELEGATE_H
#define CHANNEL_EDITOR_DELEGATE_H
#include <QStyledItemDelegate>
#include "instrument_configuration.h"

class channel_editor_delegate final : public QStyledItemDelegate
{
	Q_OBJECT

		static QString fl_channel_index_list_to_text(const fl_channel_index_list& list);
	static fl_channel_index_list qstring_to_fl_channel_index_list(const QString& text);
public:
	explicit channel_editor_delegate(QWidget* parent) :QStyledItemDelegate(parent) {}
	QWidget* createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
	void setEditorData(QWidget* editor, const QModelIndex& index) const override;
	void setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const override;
	void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
};

#endif