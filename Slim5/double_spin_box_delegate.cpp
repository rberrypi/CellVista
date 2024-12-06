#include "stdafx.h"
#include "double_spin_box_longer.h"
#include "double_spin_box_delegate.h"

double_spin_box_delegate::double_spin_box_delegate(const int precision, const bool non_negative, QObject* parent) : QStyledItemDelegate(parent), precision_(precision), non_negative_(non_negative)
{
}

QWidget* double_spin_box_delegate::createEditor(QWidget* parent,
	const QStyleOptionViewItem&/* option */,
	const QModelIndex&/* index */) const
{
	auto editor = new double_spin_box_longer(parent);
	editor->setFrame(false);
	//editor->setButtonSymbols(QAbstractSpinBox::NoButtons);
	editor->setDecimals(precision_);
	//editor->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	const auto max_val = std::numeric_limits<double>::max();
	editor->setMinimum(non_negative_ ? 0 : -max_val);
	editor->setMaximum(max_val);
	return editor;
}

void double_spin_box_delegate::setEditorData(QWidget* editor,
	const QModelIndex& index) const
{
	const auto value = index.model()->data(index, Qt::EditRole).toFloat();
	const auto spin_box = dynamic_cast<double_spin_box_longer*>(editor);
	spin_box->setValue(value);
}

void double_spin_box_delegate::setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
	const auto spin_box = dynamic_cast<double_spin_box_longer*>(editor);
	spin_box->interpretText();
	const auto value = spin_box->value();
	model->setData(index, value, Qt::EditRole);
}

void double_spin_box_delegate::updateEditorGeometry(QWidget* editor,
	const QStyleOptionViewItem& option, const QModelIndex&) const
{
	editor->setGeometry(option.rect);
}
