#include "stdafx.h"
#include "displacement_vectors_selector.h"
#include "ui_displacement_vectors_selector.h"
#include <QMessageBox>

#include "qli_runtime_error.h"

void displacement_vectors_selector::update_displacement_vectors()
{
	const auto info = this->get_displacement_vectors();
	emit displacement_vector_changed(info);
}

displacement_vectors_selector::displacement_vectors_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::displacement_selector_ui>();
	ui_->setupUi(this);
	for (auto selector : { ui_->column_step_in_pixels ,ui_->row_step_in_pixels })
	{
		QObject::connect(selector, &trakem2_xy_selector::trakem2_xy_changed, this, &displacement_vectors_selector::update_displacement_vectors);
	}
	QObject::connect(ui_->calibration, &calibration_info_selector::calibration_info_changed, this, &displacement_vectors_selector::update_displacement_vectors);
	ui_->path_selector->hide_save(true);
	QObject::connect(ui_->path_selector, &path_load_save_selector::text_changed, this, &displacement_vectors_selector::load_displacement_vector_from_file);
	QObject::connect(ui_->path_selector, &path_load_save_selector::load_button_clicked, this, &displacement_vectors_selector::load_displacement_vector_from_file);

}

void displacement_vectors_selector::load_displacement_vector_from_file()
{
	const auto filename(ui_->path_selector->get_path());

	const auto vector = trakem2_processor::get_vectors_from_xml_file(filename.toStdString());
	if (vector.is_valid())
	{
		set_displacement_vectors(vector);
	}
	else
	{
		const auto text = QString("Can't read info from %1").arg(filename);
		QMessageBox::warning(this, "Invalid File", text, QMessageBox::Ok);
	}
}

trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors displacement_vectors_selector::get_displacement_vectors()
{
	const auto column_step = ui_->column_step_in_pixels->get_trakem2_xy();
	const auto row_step = ui_->row_step_in_pixels->get_trakem2_xy();
	const auto calibration = ui_->calibration->get_calibration_info();
	const trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors vector(column_step, row_step, calibration);
	return vector;
}

void displacement_vectors_selector::set_displacement_vectors(const trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors& vectors)
{
	ui_->column_step_in_pixels->set_trakem2_xy(vectors.column_step_in_pixels);
	ui_->row_step_in_pixels->set_trakem2_xy(vectors.row_step_in_pixels);
	ui_->calibration->set_calibration_info(vectors.calibration_in_stage_microns);
#if _DEBUG
	{
		const auto what_we_set = this->get_displacement_vectors();
		if (!what_we_set.approx_equal(vectors))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

