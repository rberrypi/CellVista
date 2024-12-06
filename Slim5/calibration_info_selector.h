#pragma once
#ifndef CALIBRATION_INFO_SELECTOR_H
#define CALIBRATION_INFO_SELECTOR_H
#include "trakem2_stitching_structs.h"
#include <QWidget>

namespace Ui {
	class calibration_info_selector;
}

class calibration_info_selector final : public QWidget
{
	Q_OBJECT
		std::unique_ptr<Ui::calibration_info_selector> ui_;
	void update_calibration_info_selector();

public:
	explicit calibration_info_selector(QWidget* parent = nullptr);
	[[nodiscard]] calibration_info get_calibration_info() const;

public slots:
	void set_calibration_info(const calibration_info& calibration_info);

signals:
	void calibration_info_changed(const calibration_info& calibration_info);
};

#endif