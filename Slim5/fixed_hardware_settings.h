#pragma once
#ifndef HARDWARE_SETTINGS_H
#define HARDWARE_SETTINGS_H
#include "modulator_configuration.h"
#include "scope_compute_settings.h"

struct fixed_hardware_settings : scope_compute_settings, dpm_settings
{
	fixed_modulator_settings modulator_settings;
	fixed_hardware_settings() = default;
	fixed_hardware_settings(const fixed_modulator_settings& fixed_modulator_settings, const scope_compute_settings& scope_compute_settings, const dpm_settings& dpm_settings) noexcept: scope_compute_settings(scope_compute_settings), dpm_settings(dpm_settings), modulator_settings(fixed_modulator_settings) {
	}
	static fixed_hardware_settings generate_fixed_hardware_settings( slm_mode slm_mode,  int samples_per_pixel,  int slms) ;
	[[nodiscard]] bool item_approx_equals(const fixed_hardware_settings& b) const noexcept
	{
		const auto predicate = [](const per_modulator_saveable_settings& a, const per_modulator_saveable_settings& b)
		{
			return a.item_approx_equals(b);
		};
		const auto modulators_equal = std::equal(modulator_settings.begin(), modulator_settings.end(), b.modulator_settings.begin(), b.modulator_settings.end(), predicate);
		return modulators_equal && scope_compute_settings::item_approx_equals(b)
			&& static_cast<const dpm_settings&>(*this) == b;
	}
	[[nodiscard]] bool operator== (const fixed_hardware_settings& b) const noexcept
	{
		return  static_cast<const scope_compute_settings&>(*this) == b && static_cast<const dpm_settings&>(*this) == b ;
	}
	[[nodiscard]] bool operator!= (const fixed_hardware_settings& b) const noexcept
	{
		return !(*this == b);
	}
	[[nodiscard]] bool is_valid() const noexcept;
};
#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(dpm_settings)
#endif
#endif