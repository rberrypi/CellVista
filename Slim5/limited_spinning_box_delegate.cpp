#include "stdafx.h"
#include "limited_spinning_box_delegate.h"
#include <QDoubleSpinBox>
#include <QPainter>
#include "compute_and_scope_state.h"

QWidget* limited_spinning_box_delegate::createEditor(QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<float>())
	{
		const auto as_float = as_variant.value<float>();
		auto* editor = new QDoubleSpinBox(parent);
		editor->setButtonSymbols(QAbstractSpinBox::NoButtons);
		editor->setAlignment(Qt::AlignCenter);
		//could be the official stage min and max, although ideally we'd put a pre-flight check so that the pages wouldn't go above this value
		const auto some_placeholder = 99999.0;
		const auto qsb_min_value = isnan(max_value_) ? -some_placeholder : min_value_;
		editor->setMinimum(non_negative_ ? 0 : qsb_min_value);
		const auto qsb_max_value = isnan(max_value_) ? some_placeholder : max_value_;
		editor->setMaximum(qsb_max_value);
		editor->setDecimals(decimals_);
		editor->setValue(as_float);
		if (!special_text_.isEmpty())
		{
			editor->setSpecialValueText(special_text_);
		}
		if (isfinite(max_value_) && isfinite(min_value_))
		{
			const static auto steps = 20;
			const auto increment = (max_value_ - min_value_) / steps;
			editor->setSingleStep(increment);
		}
		return editor;
	}
	return QStyledItemDelegate::createEditor(parent, option, index);
}

void limited_spinning_box_delegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<float>())
	{
		auto edit_band = qobject_cast<QDoubleSpinBox*>(editor);
		const auto as_float = as_variant.value<float>();
		edit_band->setValue(as_float);
	}
	QStyledItemDelegate::setEditorData(editor, index);
}

void limited_spinning_box_delegate::setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<band_pass_settings>())
	{
		const auto cmb = qobject_cast<QDoubleSpinBox*>(editor);
		const auto new_value = cmb->value();
		const auto new_value_as_variant = QVariant::fromValue(new_value);
		model->setData(index, new_value_as_variant);
	}
	else
	{
		QStyledItemDelegate::setModelData(editor, model, index);
	}
}

void limited_spinning_box_delegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	const auto as_variant = index.data();
	if (as_variant.canConvert<float>())
	{
		if (option.state & QStyle::State_Selected)
		{
			painter->fillRect(option.rect, option.palette.highlight());
		}
		//
		const auto as_float = as_variant.value<float>();
		const auto str = as_float > min_value_ || isnan(min_value_) ? QString::number(as_float, 'f', decimals_) : special_text_;
		const auto  font = index.data(Qt::FontRole).value<QFont>();

		painter->save();
		painter->setFont(font);
		painter->drawText(option.rect, Qt::AlignCenter, str);
		painter->restore();
	}
	else
	{
		QStyledItemDelegate::paint(painter, option, index);
	}
}