#include "stdafx.h"
#include "channel_editor_delegate.h"
#include <QLineEdit>
#include <QPainter>
#include <QStringBuilder>
#include <iostream>

QWidget* channel_editor_delegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<fl_channel_index_list>())
	{
		const auto as_channels = as_variant.value<fl_channel_index_list>();
		auto* editor = new QLineEdit(parent);
		editor->setText(fl_channel_index_list_to_text(as_channels));
		editor->setAlignment(Qt::AlignCenter);
		return editor;
	}
	return QStyledItemDelegate::createEditor(parent, option, index);
}


QString channel_editor_delegate::fl_channel_index_list_to_text(const fl_channel_index_list& list)
{
	QString result;
	for (auto idx = 0; idx < list.size(); ++idx)
	{
		const auto is_last = idx == (list.size() - 1);
		const auto value = list.at(idx);
		if (is_last)
		{
			result = result % QString("%1").arg(value);
		}
		else
		{
			result = result % QString("%1,").arg(value);
		}
	}
	return result;
}

fl_channel_index_list channel_editor_delegate::qstring_to_fl_channel_index_list(const QString& text)
{
	fl_channel_index_list list;
	auto items = text.split(",", Qt::SkipEmptyParts);
	if (items.isEmpty())
	{
		return fl_channel_index_list{ 0 };
	}
	for (auto& item : items)
	{
		const auto as_int = item.toUInt();
		list.push_back(as_int);
	}
	return list;
}

void channel_editor_delegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<fl_channel_index_list>())
	{
		auto edit_band = qobject_cast<QLineEdit*>(editor);
		const auto as_channels = as_variant.value<fl_channel_index_list>();
		edit_band->setText(fl_channel_index_list_to_text(as_channels));
	}
	QStyledItemDelegate::setEditorData(editor, index);
}

void channel_editor_delegate::setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<fl_channel_index_list>())
	{
		const auto cmb = qobject_cast<QLineEdit*>(editor);
		const auto as_text = cmb->text();
		const auto as_value = qstring_to_fl_channel_index_list(as_text);
		const auto new_value_as_variant = QVariant::fromValue(as_value);
		model->setData(index, new_value_as_variant);
	}
	else
	{
		QStyledItemDelegate::setModelData(editor, model, index);
	}
}

void channel_editor_delegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<fl_channel_index_list>())
	{
		if (option.state & QStyle::State_Selected)
		{
			painter->fillRect(option.rect, option.palette.highlight());
		}
		//
		const auto as_channels = as_variant.value<fl_channel_index_list>();
		const auto as_text = fl_channel_index_list_to_text(as_channels);
		const auto font = index.data(Qt::FontRole).value<QFont>();
		painter->save();
		painter->setFont(font);
		painter->drawText(option.rect, Qt::AlignCenter, as_text);
		painter->restore();
	}
	else
	{
		QStyledItemDelegate::paint(painter, option, index);
	}
}