#pragma once
#ifndef QSilentDoubleSpinBox_H
#define QSilentDoubleSpinBox_H

#include <QDoubleSpinBox>
class QSilentDoubleSpinBox final : public QDoubleSpinBox
{
public:
	explicit QSilentDoubleSpinBox(QWidget* parent=nullptr);
public slots:
	void silentSetValue( double value);
};
#endif