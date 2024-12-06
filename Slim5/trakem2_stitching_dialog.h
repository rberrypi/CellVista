#pragma once
#ifndef TRAKEM2_STITCHING_DIALOG_H
#define TRAKEM2_STITCHING_DIALOG_H
#include <QDialog>
#include "trakem2_stitching_structs.h"
namespace Ui {
	class trakem2_stitching_dialog_ui;
}

class trakem2_stitching_dialog final : public QDialog
{
	Q_OBJECT
		std::unique_ptr<Ui::trakem2_stitching_dialog_ui> ui_;

public:
	explicit trakem2_stitching_dialog(QWidget* parent = nullptr);
	[[nodiscard]] trakem2_stage_coordinate_to_pixel_mapper get_mapper() const;
	static float default_step_size_um_from_pixel_ratio_and_frame(const frame_size& sensor_size_in_pixels,  float pixel_ratio);
public slots:
	void set_pixel_ratio(float pixel_ratio);
	void set_step_size_um(float step_size_um);

signals:
	void do_calibration(const calibration_info& xy);
	void write_trakem2(const trakem2_stage_coordinate_to_pixel_mapper& mapper, const calibration_info& calibration);
};

#endif