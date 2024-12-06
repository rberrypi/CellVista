#pragma once
#ifndef SNAP_TO_MIN_SPINBOX_H
#define SNAP_TO_MIN_SPINBOX_H

#include <QDoubleSpinBox>

class snap_to_min_spinbox final : public QDoubleSpinBox
{
public:
	explicit snap_to_min_spinbox(QWidget* parent = Q_NULLPTR) :QDoubleSpinBox(parent) {}
	void  fixup(QString& input) const override;
	QValidator::State 	validate(QString& text, int& pos) const override;
};

#endif