#pragma once
#ifndef  QDOUBLESPINBOX_LONGER_H
#define QDOUBLESPINBOX_LONGER_H
#include <QDoubleSpinBox>

class double_spin_box_longer final :public QDoubleSpinBox
{
	Q_OBJECT

public:
	explicit double_spin_box_longer(QWidget* parent = nullptr) :QDoubleSpinBox(parent) {}
};

#endif