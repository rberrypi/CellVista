#pragma once
#ifndef QSPINBOXFIXED_H
#define QSPINBOXFIXED_H

#include <QSpinBox>
class spin_box_fixed_increment final : public QSpinBox
{
	Q_OBJECT
public:
	explicit spin_box_fixed_increment(QWidget* parent = nullptr) :
		QSpinBox(parent),increment(16) {
		set_fixed_increment(increment);
	}

	int increment;
	QValidator::State validate(QString& input, int& pos) const override;
	void set_fixed_increment( int increment);
};


#endif