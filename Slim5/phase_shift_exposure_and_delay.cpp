#include "stdafx.h"
#include "phase_shift_exposure_and_delay.h"
#include "approx_equals.h"

bool phase_shift_exposure_and_delay::operator== (const phase_shift_exposure_and_delay& c) const
{
	return slm_stability == c.slm_stability && exposure_time == c.exposure_time && duration() == c.duration();
}

[[nodiscard]] bool phase_shift_exposure_and_delay::approx_equal(const phase_shift_exposure_and_delay& bb, const std::chrono::microseconds& min_exposure) const
{
	return approx_equals(std::max(exposure_time, min_exposure).count(), std::max(bb.exposure_time, min_exposure).count()) && approx_equals(slm_stability.count(), bb.slm_stability.count());
};