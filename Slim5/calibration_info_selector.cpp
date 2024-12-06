#include "stdafx.h"
#include "calibration_info_selector.h"
#include "ui_calibration_info_selector.h"
#include "qli_runtime_error.h"

void calibration_info_selector::update_calibration_info_selector()
{
	const auto info = this->get_calibration_info();
	emit this->calibration_info_changed(info);
}

calibration_info_selector::calibration_info_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::calibration_info_selector>();
	ui_->setupUi(this);
	QObject::connect(ui_->calibration_steps, &trakem2_xy_selector::trakem2_xy_changed, this, &calibration_info_selector::update_calibration_info_selector);
	QObject::connect(ui_->pixel_ratio, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &calibration_info_selector::update_calibration_info_selector);
}

void calibration_info_selector::set_calibration_info(const calibration_info& calibration_info)
{
	ui_->calibration_steps->set_trakem2_xy(calibration_info.calibration_steps_in_stage_microns);
	ui_->pixel_ratio->setValue(calibration_info.calibration_pixel_ratio);
#if _DEBUG
	{
		const auto info = this->get_calibration_info();
		if (!info.item_approx_equals(calibration_info))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

calibration_info calibration_info_selector::get_calibration_info() const
{
	const auto xy = ui_->calibration_steps->get_trakem2_xy();
	const auto pixel_ratio = ui_->pixel_ratio->value();
	const calibration_info info(xy.x, xy.y, pixel_ratio);
	return info;
}

