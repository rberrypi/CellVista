#pragma once
#ifndef FRAME_META_DATA_H
#define FRAME_META_DATA_H
#include "capture_cycle.h"
#include "phase_shift_exposure_and_delay.h"
#include "scope_compute_settings.h"
#include "render_settings.h"
#include "instrument_configuration.h"

struct frame_meta_data_before_acquire : pixel_dimensions, cycle_position, phase_shift_exposure_and_delay, render_settings, microscope_move_action
{
	phase_processing processing;
	int channel_route_index;
	scope_action action;
	explicit frame_meta_data_before_acquire(const pixel_dimensions& pixel_dimensions, const cycle_position& cycle_position, const phase_shift_exposure_and_delay& phase_shift_exposure_and_delay, const render_settings& render_settings, const microscope_move_action& microscope_move_action, const phase_processing processing, const int channel_route_index, const scope_action scope_action) noexcept:pixel_dimensions(pixel_dimensions), cycle_position(cycle_position), phase_shift_exposure_and_delay(phase_shift_exposure_and_delay), render_settings(render_settings), microscope_move_action(microscope_move_action), processing(processing), channel_route_index(channel_route_index), action(scope_action) {}
	frame_meta_data_before_acquire() noexcept : frame_meta_data_before_acquire(pixel_dimensions(), cycle_position(), phase_shift_exposure_and_delay(), render_settings(), microscope_move_action(), phase_processing::raw_frames, 0, scope_action::capture) {}
	[[nodiscard]] bool is_stop_capture() const noexcept
	{
		return action == scope_action::shutdown_live;
	}
	[[nodiscard]] bool is_valid() const noexcept;
};

struct frame_meta_data : frame_meta_data_before_acquire
{
	std::chrono::microseconds timestamp;
	explicit frame_meta_data(const frame_meta_data_before_acquire& before, const std::chrono::microseconds timestamp) noexcept : frame_meta_data_before_acquire(before), timestamp(timestamp) {}
	frame_meta_data() noexcept : frame_meta_data(frame_meta_data_before_acquire(), std::chrono::microseconds(0)) {}
};
#endif