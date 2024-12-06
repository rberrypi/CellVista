#include "stdafx.h"
#include "distorted_donut_selector.h"
#include "ui_distorted_donut_selector.h"
#include "qli_runtime_error.h"
distorted_donut_selector::distorted_donut_selector(QWidget *parent) : QWidget(parent)
{
	ui_=std::make_unique<Ui::distorted_donut_selector>();
	ui_->setupUi(this);
	for (auto* item : { ui_->x_center,ui_->y_center,ui_->inner_diameter,ui_->outer_diameter,ui_->ellipticity_e,ui_->ellipticity_f })
	{
		QObject::connect(item, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &distorted_donut_selector::update_values);
	}
	set_complete(get_distorted_donut().is_complete());
}

void distorted_donut_selector::set_complete(const bool is_complete)
{
	const auto* color_label = is_complete ? "" : "color: red;";
	for (auto* item : { ui_->outer_diameter,ui_->inner_diameter,ui_->ellipticity_e,ui_->ellipticity_f })
	{
		item->setStyleSheet(color_label);
	}
}

distorted_donut distorted_donut_selector::get_distorted_donut() const
{
	const auto x_center = ui_->x_center->value();
	const auto y_center = ui_->y_center->value();
	const auto inner_diameter = ui_->inner_diameter->value();
	const auto outer_diameter = ui_->outer_diameter->value();
	const auto ellipticity_e = ui_->ellipticity_e->value();
	const auto ellipticity_f = ui_->ellipticity_f->value();
	const distorted_donut doughnut(x_center, y_center, inner_diameter, outer_diameter, ellipticity_e, ellipticity_f);
	return doughnut;
}

void distorted_donut_selector::set_distorted_donut_silent(const distorted_donut& distorted_donut)
{
	QSignalBlocker blk(*this);
	set_distorted_donut(distorted_donut);
}

void  distorted_donut_selector::set_distorted_donut(const distorted_donut& distorted_donut)
{
	ui_->x_center->setValue(distorted_donut.x_center);
	ui_->y_center->setValue(distorted_donut.y_center);
	ui_->inner_diameter->setValue(distorted_donut.inner_diameter);
	ui_->outer_diameter->setValue(distorted_donut.outer_diameter);
	ui_->ellipticity_e->setValue(distorted_donut.ellipticity_e);
	ui_->ellipticity_f->setValue(distorted_donut.ellipticity_f);
#if _DEBUG
	{
		const auto what_we_got = this->get_distorted_donut();
		if (!what_we_got.item_approx_equals(distorted_donut))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void distorted_donut_selector::bump_donut(const int x, const int y, const int inner, const int outer)
{
	const auto bump = [](QDoubleSpinBox* what, const int bumps)
	{
		if (bumps != 0)
		{
			// maybe doesn't screw up like step by?
			const auto to_move = bumps * what->singleStep();
			const auto current_value = what->value();
			what->setValue(current_value + to_move);
		}
	};
	bump(ui_->x_center, x);
	bump(ui_->y_center, y);
	bump(ui_->inner_diameter, inner);
	bump(ui_->outer_diameter, outer);
}

void  distorted_donut_selector::update_values()
{
	const auto what = get_distorted_donut();
	set_complete(what.is_complete());
	emit value_changed(what);
}