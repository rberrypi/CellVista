#pragma once
#ifndef PHASE_SHIFT_EXPOSURE_AND_DELAY_H
#define PHASE_SHIFT_EXPOSURE_AND_DELAY_H
#include <chrono>
#include "common_limits.h"
#include <boost/container/small_vector.hpp>

struct phase_shift_exposure_and_delay
{
	std::chrono::microseconds slm_stability, exposure_time;
	[[nodiscard]] std::chrono::microseconds duration() const noexcept
	{
		return slm_stability + exposure_time;
	}
	phase_shift_exposure_and_delay(const std::chrono::microseconds& slm_stability, const std::chrono::microseconds& exposure_time) noexcept :slm_stability(slm_stability), exposure_time(exposure_time) {}
	phase_shift_exposure_and_delay() noexcept: phase_shift_exposure_and_delay(std::chrono::microseconds(0), std::chrono::microseconds(0)) {}

	bool operator== (const phase_shift_exposure_and_delay& c) const;

	[[nodiscard]] bool approx_equal(const phase_shift_exposure_and_delay& bb, const std::chrono::microseconds& min_exposure) const;
};
typedef  boost::container::small_vector<phase_shift_exposure_and_delay, typical_psi_patterns> phase_shift_exposures_and_delays;

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(std::chrono::microseconds)
Q_DECLARE_METATYPE(phase_shift_exposures_and_delays)
#endif


#endif