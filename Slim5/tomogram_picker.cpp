#include "stdafx.h"
#include "tomogram_picker.h"
#include "device_factory.h"
#include "scope.h"
#include "ui_tomogram_picker.h"
tomogram_picker::tomogram_picker(const float xy_pixel_ratio, QWidget* parent) : QDialog(parent)
{
	ui_ = std::make_unique<Ui::tomogram_picker>();
	ui_->setupUi(this);
	setAttribute(Qt::WA_DeleteOnClose, true);
	setWindowModality(Qt::WindowModal);
	setSizeGripEnabled(false);
	adjustSize();
	//
	connect(ui_->btnBottom, &QPushButton::toggled, this, &tomogram_picker::goto_bottom);
	connect(ui_->btnTop, &QPushButton::toggled, this, &tomogram_picker::goto_top);
	connect(ui_->btnAdd, &QPushButton::pressed,
		[&] {
		const auto tomogram = get_tomogram();
		add_tomogram(tomogram);
		emit accept(); }
	);
	connect(ui_->btnCancel, &QPushButton::pressed, this, &tomogram_picker::reject);
	//
	connect(ui_->qsbIncrements, qOverload<double>(&QDoubleSpinBox::valueChanged), this,&tomogram_picker::increment_change);
	connect(ui_->qsbTop, qOverload<double>(&QDoubleSpinBox::valueChanged),this,&tomogram_picker::increment_change);
	connect(ui_->qsbBottom, qOverload<double>(&QDoubleSpinBox::valueChanged),this,&tomogram_picker::increment_change);
	//
	const auto here = D->scope->get_state().z;//this way structure is filled with stuff
	ui_->qsbTop->setValue(here);
	ui_->qsbBottom->setValue(here);
	const auto increment = 1 / xy_pixel_ratio;
	ui_->qsbIncrements->setValue(increment);
}

tomogram tomogram_picker::get_tomogram() const
{
	const auto t = ui_->qsbTop->value();
	const auto b = ui_->qsbBottom->value();
	const auto z_inc = ui_->qsbIncrements->value();
	const auto z_middle = (t + b) / 2;
	const auto steps_plus_minus = static_cast<int>(ceil(abs(t - z_middle) / z_inc));
	return { z_middle,z_inc, steps_plus_minus };
}

void tomogram_picker::increment_change() const
{
	//fix?
	const auto tomogram = get_tomogram();
	ui_->qsbSteps->setValue(tomogram.steps);
}

void tomogram_picker::goto_top(const bool enable) const
{
	if (!enable)
	{//on release
		//for testing
#ifdef testingtomogram
		{
			auto here = scope->getPos();
			here.z = here.z + 89.2;
			scope->move_to(here);
		}
#endif
		const auto here = D->scope->get_state().z;
		ui_->qsbTop->setValue(here);
	}
}

void tomogram_picker::goto_bottom(const bool enable) const
{
	if (!enable)
	{//on release
		//for testing
#ifdef testingtomogram
		{
			auto here = scope->getPos();
			here.z = here.z - 12.1;
			scope->move_to(here);
		}
#endif
		const auto here = D->scope->get_state().z;
		ui_->qsbBottom->setValue(here);
	}
}