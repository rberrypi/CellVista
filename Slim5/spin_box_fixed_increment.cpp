#include "stdafx.h"
#include "spin_box_fixed_increment.h"

QValidator::State spin_box_fixed_increment::validate(QString& input, int& pos) const
{
	bool okay;
	const auto foo = input.toUInt(&okay);
	if (!okay)
	{
		return QValidator::Invalid;
	}
	return foo % increment != 0 ? QValidator::Invalid : QSpinBox::validate(input, pos);
}

void spin_box_fixed_increment::set_fixed_increment(const int increment)
{
	this->increment = increment;
	setSingleStep(increment);
	const auto value = this->value();
	const auto fixed_value = increment * (value / increment);
	setValue(fixed_value);
}