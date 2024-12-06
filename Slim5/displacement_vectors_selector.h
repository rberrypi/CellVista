#pragma once
#ifndef DISPLACEMENT_SELECTOR_H
#define DISPLACEMENT_SELECTOR_H
#include <QWidget>
#include "trakem2_stitching_structs.h"
namespace Ui {
	class displacement_selector_ui;
}

class displacement_vectors_selector final : public QWidget
{
	Q_OBJECT

		std::unique_ptr<Ui::displacement_selector_ui> ui_;
	void update_displacement_vectors();
	void load_displacement_vector_from_file();
public:
	explicit displacement_vectors_selector(QWidget* parent = nullptr);

	[[nodiscard]] trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors get_displacement_vectors();

public slots:
	void set_displacement_vectors(const trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors& vectors);

signals:
	void displacement_vector_changed(const trakem2_stage_coordinate_to_pixel_mapper::displacement_vectors& vectors);

};

#endif