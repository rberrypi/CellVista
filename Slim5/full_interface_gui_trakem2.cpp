#include "stdafx.h"
#include "full_interface_gui.h"
#include "device_factory.h"
#include "trakem2_stitching_dialog.h"
#include "ui_full_interface_gui.h"
#include <QMessageBox>
void full_interface_gui::setup_trakem2()
{
	const auto open_trakem2_settings = [&]
	{
		if (trakem2_dialog_ == nullptr)
		{
			trakem2_dialog_ = new trakem2_stitching_dialog(nullptr);
			const auto pixel_ratio = this->settings_file_.pixel_ratio;
			trakem2_dialog_->set_pixel_ratio(pixel_ratio);
			const auto sensor_size_in_pixels = default_sensor_size_in_pixels();
			trakem2_dialog_->set_step_size_um(trakem2_stitching_dialog::default_step_size_um_from_pixel_ratio_and_frame(sensor_size_in_pixels, pixel_ratio));
			connect(this, &full_interface_gui::pixel_ratio_changed, trakem2_dialog_, &trakem2_stitching_dialog::set_pixel_ratio);
			connect(trakem2_dialog_, &trakem2_stitching_dialog::do_calibration, this, &full_interface_gui::trakem2_calibration);
			connect(trakem2_dialog_, &trakem2_stitching_dialog::write_trakem2, [&](const trakem2_stage_coordinate_to_pixel_mapper& mapper, const calibration_info& calibration)
			{
				const auto base_directory = get_dir().toStdString();
				write_trakem2_xml(mapper, base_directory, calibration);
			});
			connect(trakem2_dialog_, &QMainWindow::destroyed, [&]
			{
				trakem2_dialog_ = nullptr;
			});
		}
		trakem2_dialog_->showNormal();
		trakem2_dialog_->raise();
	};
	connect(ui_->btnTrakem2Settings, &QPushButton::clicked, open_trakem2_settings);
}


void full_interface_gui::trakem2_calibration(const calibration_info& xy)
{
	//1. make 2 by 2 ROI using current ROI selected Or first ROI
	//2. Acquire 4 images and Output them in file called "Alignment_x500_y450.xml"  (500 and 400 should be the x and y steps of the ROI)

	//Remove Current ROIs
	const auto rows = rois_->rowCount();
	if (rows > 0)
	{
		QMessageBox msg_box;
		msg_box.setText("Existing items will be cleared");
		msg_box.setInformativeText("Continue?");
		msg_box.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
		msg_box.setDefaultButton(QMessageBox::No);
		if (msg_box.exec() == QMessageBox::No)
		{
			return;
		}
	}
	if (rows > 0)
	{
		rois_->removeRows(rows - 1, rows);
	}
	insert_point();
	const auto roi_idx = get_roi_index();
	rois_->setData(rois_->index(roi_idx, ITEM_COLS_IDX), 2);
	rois_->setData(rois_->index(roi_idx, ITEM_ROWS_IDX), 2);

	const auto x_step = xy.calibration_steps_in_stage_microns.x;
	const auto y_step = xy.calibration_steps_in_stage_microns.y;

	rois_->setData(rois_->index(roi_idx, ITEM_STEP_COL_IDX), x_step);
	rois_->setData(rois_->index(roi_idx, ITEM_STEP_ROW_IDX), y_step);

	const fl_channel_index_list channels = { 0 };
	rois_->setData(rois_->index(roi_idx, ITEM_CHANNEL_IDX), QVariant::fromValue(channels));

	ui_->metadata->setPlainText("TrakEM2 Calibration");
	do_scan();
}

void full_interface_gui::write_trakem2_xml(const trakem2_stage_coordinate_to_pixel_mapper& mapper, const std::string& basedir, const calibration_info& xy)
{
	acquisition route;
	std::set<int> channels_used;
	wrangle_capture_items(route, channels_used);
	const auto current_pixel_ratio = settings_file_.pixel_ratio;
	const auto trakem2 = trakem2_processor::acquisition_to_trakem2(route, mapper, current_pixel_ratio, xy);
	trakem2_processor::write_trakem2(trakem2, basedir);
}