#pragma once
#ifndef PHASE_SHIFT_EXPOSURE_AND_DELAY_SELECTOR_H
#define PHASE_SHIFT_EXPOSURE_AND_DELAY_SELECTOR_H

#include "phase_shift_exposure_and_delay.h"

namespace Ui {
	class phase_shift_exposure_and_delay_selector;
}
#include <QWidget>
class phase_shift_exposure_and_delay_selector final : public QWidget
{
	Q_OBJECT
		std::unique_ptr<Ui::phase_shift_exposure_and_delay_selector> ui;
	void update_phase_shift_exposure_and_delay();
public:

	explicit phase_shift_exposure_and_delay_selector(QWidget* parent = Q_NULLPTR);
	virtual ~phase_shift_exposure_and_delay_selector();
	[[nodiscard]] phase_shift_exposure_and_delay get_phase_shift_exposure_and_delay() const;
	[[nodiscard]] std::chrono::microseconds get_minimum_exposure_time() const;
public slots:
	void set_id(int id);
	void set_phase_shift_exposure_and_delay(const phase_shift_exposure_and_delay& settings) const;
	void set_minimum_exposure_time(std::chrono::microseconds time);

signals:
	void phase_shift_exposure_and_delay_changed(const phase_shift_exposure_and_delay& settings);
};

#endif 
