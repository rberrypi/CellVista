#pragma once
#ifndef TRAKEM2_XY_SELECTOR_H
#define TRAKEM2_XY_SELECTOR_H
#include "trakem2_stitching_structs.h"
#include <QWidget>

namespace Ui {
	class trakem2_xy_selector;
}

class trakem2_xy_selector final : public QWidget
{
	Q_OBJECT
		std::unique_ptr<Ui::trakem2_xy_selector> ui_;
	void update_trakem2_xy();

public:
	explicit trakem2_xy_selector(QWidget* parent = nullptr);
	[[nodiscard]] trakem2_xy get_trakem2_xy() const;

public slots:
	void set_trakem2_xy(const trakem2_xy& trakem2_xy);

signals:
	void trakem2_xy_changed(const trakem2_xy& trakem2_xy);
};

#endif