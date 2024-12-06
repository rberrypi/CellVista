#include "stdafx.h"
#include "QSilentDoubleSpinBox.h"

QSilentDoubleSpinBox::QSilentDoubleSpinBox(QWidget* parent) : QDoubleSpinBox(parent)
{

}
void QSilentDoubleSpinBox::silentSetValue(double value)
{
	QSignalBlocker blocker(*this);
	setValue(value);
}