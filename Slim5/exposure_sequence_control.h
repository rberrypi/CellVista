#pragma once
#ifndef Q_EXPOSURE_SEQUENCE_CONTROL
#define Q_EXPOSURE_SEQUENCE_CONTROL
#include "phase_processing.h"
#include "phase_shift_exposure_and_delay.h"

// ReSharper disable once CppInconsistentNaming
class QDoubleSpinBox;
class snap_to_min_spinbox;
#include <mutex>
#include <QWidget>
class phase_shift_exposure_and_delay_selector;
class exposure_sequence_control final : public QWidget
{
	Q_OBJECT

	std::vector<phase_shift_exposure_and_delay_selector*> setting_items_;// todo this needs to resize by the number of patterns
	phase_retrieval current_phase_retrieval;
	mutable std::recursive_mutex hack_;
	void update_values();
	void bulk_set_phase_shift_exposure_and_delay(const phase_shift_exposure_and_delay& phase_shift_exposure_and_delay);
public:
	explicit exposure_sequence_control(QWidget* parent = Q_NULLPTR);
	[[nodiscard]] phase_shift_exposures_and_delays get_exposures_and_delays() const;
	[[nodiscard]] std::chrono::microseconds min_time() const;
public slots:
	void set_phase_shift_exposures_and_delays(const phase_shift_exposures_and_delays& settings);
	void current_phase_retrieval_changed(phase_retrieval phase_retrieval);
	void set_minimum_exposure_time(const std::chrono::microseconds& minimum_time);
	void resize_exposures(int patterns);
signals:
	void phase_shift_exposures_and_delays_changed(const phase_shift_exposures_and_delays&);

};


#endif