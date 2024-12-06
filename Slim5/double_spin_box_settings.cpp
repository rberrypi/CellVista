#include "stdafx.h"
#include "double_spin_box_settings.h"
#include <QDoubleSpinBox>
void double_spin_box_settings::style_spin_box(QDoubleSpinBox* spin_box) const
{
	spin_box->setMinimum(min_value);
	spin_box->setMaximum(max_value);
	spin_box->setDecimals(decimal_places);
	spin_box->setButtonSymbols(QAbstractSpinBox::NoButtons);
}
