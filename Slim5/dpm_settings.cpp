#include "stdafx.h"
#include "fixed_hardware_settings.h"
#include <boost/format.hpp>
#include "ml_shared.h"


bool dpm_settings::is_valid() const noexcept
{
	return dpm_phase_is_valid() && dpm_amp_is_valid();
}

bool dpm_settings::dpm_phase_is_valid() const noexcept
{
	return  dpm_phase_left_column >= 0 && dpm_phase_top_row >= 0 && is_divisible_by_sixteen(dpm_phase_width);
}

bool dpm_settings::dpm_phase_is_complete() const noexcept
{
	return dpm_phase_width > 0 && dpm_phase_is_valid();
}

bool dpm_settings::dpm_amp_is_valid() const noexcept
{
	return dpm_amp_left_column >= 0 && dpm_amp_top_row >= 0 && is_divisible_by_sixteen(dpm_amp_width);
}

bool dpm_settings::dpm_amp_is_complete() const noexcept
{
	return dpm_amp_width > 0 && dpm_amp_is_valid();
}

std::string dpm_settings::get_dpm_string() const noexcept
{
	return boost::str(boost::format("[%05d,%05d,%05d][%05d,%05d,%05d]") % this->dpm_phase_left_column % this->dpm_phase_top_row % this->dpm_phase_width % this->dpm_amp_left_column % this->dpm_amp_top_row % this->dpm_amp_width);
}

bool dpm_settings::fits_in_frame(const frame_size& frame) const
{
	const auto in_range = [](auto start, auto stop, auto value)
	{
		return value >= start && value < stop;
	};
	const auto left_phase = in_range(0, frame.width, dpm_phase_left_column + dpm_phase_width);
	const auto  left_amp = in_range(0, frame.width, dpm_amp_left_column + dpm_amp_width);
	const auto  top_phase = in_range(0, frame.height, dpm_phase_top_row + dpm_phase_width);
	const auto  top_amp = in_range(0, frame.height, dpm_amp_top_row + dpm_amp_width);
	return left_phase && left_amp && top_phase && top_amp;
}

