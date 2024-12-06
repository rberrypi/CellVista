#include "stdafx.h"
#include "trakem2_stitching_dialog.h"
#include "ui_trakem2_stitching_dialog.h"

trakem2_stitching_dialog::trakem2_stitching_dialog(QWidget* parent)
{
	ui_ = std::make_unique<Ui::trakem2_stitching_dialog_ui>();
	ui_->setupUi(this);
	const auto calibration_button = [&]
	{
		const auto pixel_ratio = static_cast<float>(ui_->qsbCurrentPixelRatio->value());
		const auto micron_step = ui_->qsb_step_size->value();
		const calibration_info info(micron_step, micron_step, pixel_ratio);
		const auto column_step_in_pixels = trakem2_xy(micron_step * pixel_ratio, 0);
		const auto row_step_in_pixels = trakem2_xy(0, micron_step * pixel_ratio);
		const trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors vectors(column_step_in_pixels, row_step_in_pixels, info);
		const auto pass_though_mapper = trakem2_stage_coordinate_to_pixel_mapper(vectors);
		emit do_calibration(info);
		emit write_trakem2(pass_though_mapper, info);
	};
	QObject::connect(ui_->btnTrakem2Calibration, &QPushButton::clicked, calibration_button);
	const auto write_trakem2_button = [&]
	{
		const auto actual_mapper = get_mapper();
		if (actual_mapper.settings.is_valid())
		{
			const auto unused_info = calibration_info();
			emit write_trakem2(actual_mapper, unused_info);
		}
	};
	QObject::connect(ui_->btnWriteTrakem, &QPushButton::clicked, write_trakem2_button);
}

void trakem2_stitching_dialog::set_step_size_um(const float step_size_um)
{
	ui_->qsb_step_size->setValue(step_size_um);
}

void trakem2_stitching_dialog::set_pixel_ratio(const float pixel_ratio)
{
	ui_->qsbCurrentPixelRatio->setValue(pixel_ratio);
}

trakem2_stage_coordinate_to_pixel_mapper trakem2_stitching_dialog::get_mapper() const
{
	const auto vectors = ui_->displacement_vectors->get_displacement_vectors();
	const auto pass_though_mapper = trakem2_stage_coordinate_to_pixel_mapper(vectors);
	return pass_though_mapper;
}

float trakem2_stitching_dialog::default_step_size_um_from_pixel_ratio_and_frame(const frame_size& sensor_size_in_pixels, const float pixel_ratio)
{
	const auto static jump = 0.75f;//jump should be high for better sensitivity, but not too high or else you can't align yo.
	const auto min_dimension = std::min(sensor_size_in_pixels.height, sensor_size_in_pixels.width);
	const auto microns_in_frame = min_dimension / pixel_ratio;
	return microns_in_frame * jump;
}
