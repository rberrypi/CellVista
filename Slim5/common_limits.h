#pragma once
#ifndef COMMON_LIMITS_H
#define COMMON_LIMITS_H

static constexpr auto max_samples_per_pixel = 3;
static constexpr auto typical_psi_patterns = 4;
static constexpr auto typical_calibration_patterns = 511;
static constexpr auto max_slms = 3;
static constexpr auto qt_spin_box_precision = 1;
static constexpr auto pol_two_patterns = 2 * 2;
static constexpr auto pol_psi_patterns = 4 * typical_psi_patterns;
static constexpr auto single_shot = 1;
static constexpr auto pattern_count_from_file = -1;

#endif