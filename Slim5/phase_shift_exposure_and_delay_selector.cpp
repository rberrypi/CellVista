#include "stdafx.h"
#include "phase_shift_exposure_and_delay_selector.h"

#include "approx_equals.h"
#include "qli_runtime_error.h"
#include "ui_phase_shift_exposure_and_delay_selector.h"

phase_shift_exposure_and_delay_selector::~phase_shift_exposure_and_delay_selector()= default;

phase_shift_exposure_and_delay_selector::phase_shift_exposure_and_delay_selector(QWidget* parent) : QWidget(parent)
{
	ui = std::make_unique<Ui::phase_shift_exposure_and_delay_selector>();
	ui->setupUi(this);
	QObject::connect(ui->slm_stability, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &phase_shift_exposure_and_delay_selector::update_phase_shift_exposure_and_delay);
	QObject::connect(ui->exposure_time, qOverload<double>(&QDoubleSpinBox::valueChanged), this, &phase_shift_exposure_and_delay_selector::update_phase_shift_exposure_and_delay);
}

void phase_shift_exposure_and_delay_selector::update_phase_shift_exposure_and_delay()
{
	const auto values = get_phase_shift_exposure_and_delay();
	emit phase_shift_exposure_and_delay_changed(values);
}

phase_shift_exposure_and_delay phase_shift_exposure_and_delay_selector::get_phase_shift_exposure_and_delay() const
{
	const auto slm_stability = ms_to_chrono(ui->slm_stability->value());
	const auto exposure_time = ms_to_chrono(ui->exposure_time->value());
	const phase_shift_exposure_and_delay delay(slm_stability, exposure_time);
	return delay;
}

void phase_shift_exposure_and_delay_selector::set_id(const int id)
{
	ui->slm_stability->setToolTip(QString("Mod %1").arg(id));
	ui->slm_stability->setToolTip(QString("Exp %1").arg(id));
}

void phase_shift_exposure_and_delay_selector::set_phase_shift_exposure_and_delay(const phase_shift_exposure_and_delay& settings) const
{
	ui->slm_stability->setValue(to_mili(settings.slm_stability));
	ui->exposure_time->setValue(to_mili(settings.exposure_time));
#if _DEBUG
	{
		const auto what_we_got = get_phase_shift_exposure_and_delay();
		const auto min_time = get_minimum_exposure_time();
		if (!what_we_got.approx_equal(settings, min_time))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void phase_shift_exposure_and_delay_selector::set_minimum_exposure_time(const std::chrono::microseconds time)
{
	const auto as_ms = to_mili(time);
	ui->exposure_time->setMinimum(as_ms);
#if _DEBUG
	{
		const auto what_we_got = get_minimum_exposure_time().count() / 1000;
		const auto what_we_tried_to_set = time.count() / 1000;
		if (!approx_equals(what_we_got, what_we_tried_to_set))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

std::chrono::microseconds phase_shift_exposure_and_delay_selector::get_minimum_exposure_time() const
{
	const auto min_time = ms_to_chrono(ui->exposure_time->minimum());
	return  min_time;
}
