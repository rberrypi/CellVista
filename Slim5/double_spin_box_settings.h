#pragma once
#ifndef DOUBLE_SPINBOX_SETTINGS_H
#define DOUBLE_SPINBOX_SETTINGS_H

class QDoubleSpinBox;
struct double_spin_box_settings
{
	double min_value, max_value;
	int decimal_places;
	double_spin_box_settings(const double min_value,const double max_value, const int decimal_places) noexcept: min_value(min_value), max_value(max_value),decimal_places(decimal_places)
	{
		
	}
	void style_spin_box(QDoubleSpinBox* spin_box) const;
	[[nodiscard]] bool inside_range(const double value) const noexcept
	{
		return value >=min_value && value <=max_value;
	}
};


#endif