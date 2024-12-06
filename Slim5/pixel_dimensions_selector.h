#pragma once
#ifndef SCOPE_COMPUTE_DIMENSIONS_SELECTOR_H
#define SCOPE_COMPUTE_DIMENSIONS_SELECTOR_H

#include <QWidget>
#include "scope_compute_settings.h"

namespace Ui
{
	class pixel_dimensions_selector;
}

class pixel_dimensions_selector final : public QWidget
{
	Q_OBJECT

	std::unique_ptr<Ui::pixel_dimensions_selector> ui_;
	void pixel_dimensions_update();	

	public:
	explicit pixel_dimensions_selector(QWidget* parent = Q_NULLPTR);
	[[nodiscard]] pixel_dimensions get_pixel_dimensions() const;

public slots:
	void set_pixel_dimensions(const pixel_dimensions& pixel_dimensions) const;

signals:
	void pixel_dimensions_changed(const pixel_dimensions& pixel_dimensions);
};

#endif
