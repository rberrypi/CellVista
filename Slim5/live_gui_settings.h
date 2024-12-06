#pragma once
#ifndef LIVE_GUI_SETTINGS_H
#define LIVE_GUI_SETTINGS_H
#include "compute_and_scope_state.h"
#include "phase_shift_exposure_and_delay.h"
#include <boost/container/small_vector.hpp>
struct live_gui_settings : compute_and_scope_settings
{

	phase_shift_exposures_and_delays exposures_and_delays;
	static live_gui_settings get_default_live_gui_settings(int scope_channel,const processing_double& settings, bool is_ft);
	live_gui_settings(const compute_and_scope_settings& compute_and_scope_settings, const phase_shift_exposures_and_delays& live_pattern_settings) :compute_and_scope_settings(compute_and_scope_settings), exposures_and_delays(live_pattern_settings) {}
	live_gui_settings() noexcept : live_gui_settings(compute_and_scope_settings(), phase_shift_exposures_and_delays()) {}
	void set_phase_retrieval_and_processing(const processing_quad& quad) noexcept
	{
		static_cast<processing_quad&>(*this) = quad;
	}

	[[nodiscard]] bool is_valid() const;
	void assert_validity() const;

	[[nodiscard]] static live_gui_settings read(const std::string& filename, bool& okay);
	[[nodiscard]] bool write(const std::string& filename) const;
	[[nodiscard]] capture_iterator_view iterator() const;
	void fill_exposure_settings(std::chrono::microseconds exposure);
	[[nodiscard]] bool item_approx_equals(const live_gui_settings& b,const std::chrono::microseconds min_exposure) const noexcept
	{
		const auto predicate = [&](const phase_shift_exposure_and_delay& aa, const phase_shift_exposure_and_delay& bb)
		{
			return aa.approx_equal(bb,min_exposure);
		};
		return std::equal(exposures_and_delays.begin(), exposures_and_delays.end(), b.exposures_and_delays.begin(), b.exposures_and_delays.end(), predicate) && compute_and_scope_settings::item_approx_equals(b);
	}
};
typedef std::array<live_gui_settings, 9> live_channels;


#endif