#include "stdafx.h"
#include "snap_to_min_spinbox.h"

void snap_to_min_spinbox::fixup(QString& input) const
{
	const auto as_value = input.toDouble();
	const auto new_value = std::max(as_value, this->minimum());
	input = QString::number(new_value);
	QDoubleSpinBox::fixup(input);
}

QValidator::State snap_to_min_spinbox::validate(QString& text, int& pos) const
{
	return QDoubleSpinBox::validate(text, pos);
}
