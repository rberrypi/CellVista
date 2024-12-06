#pragma once
#ifndef CAPTURE_ITEM_H
#define CAPTURE_ITEM_H
#include <QString>
#include "instrument_configuration.h"

struct capture_item_within_layer_hash
{
	int row, column, time;
	capture_item_within_layer_hash(const int row, const int column, const int time) noexcept : row(row), column(column), time(time)
	{

	}
	bool operator<(const capture_item_within_layer_hash& ob) const;
};

struct capture_item_2dt_hash
{

	//f0_i0_ch0_z0_mPhase.xml (note that the postfix doesn't form the hash)
	int roi, repeat, channel_route_index, page, pattern_idx;
	capture_item_2dt_hash(const int roi, const int repeat, const int channel_route_index, const int page, const int pattern_idx) noexcept : roi(roi), repeat(repeat), channel_route_index(channel_route_index), page(page), pattern_idx(pattern_idx)
	{

	}
	bool operator<(const capture_item_2dt_hash& ob) const;
};

struct roi_name
{
	int roi, time, repeat;
	int row, column, page;
	explicit roi_name(const  int roi_number, const  int time_point, const  int repeat, const  int column, const  int row, const  int page) noexcept :
		roi(roi_number), time(time_point), repeat(repeat), row(row), column(column), page(page) {}

	roi_name() noexcept : roi_name(0, 0, 0, 0, 0, 0) {}
};

struct scope_delays
{
	std::chrono::microseconds stage_move_delay, roi_move_delay;
	explicit scope_delays(const std::chrono::microseconds stage_move, const std::chrono::microseconds roi_delay) noexcept : stage_move_delay(stage_move), roi_move_delay(roi_delay)
	{
	}
	scope_delays() noexcept : scope_delays(std::chrono::microseconds(0), std::chrono::microseconds(0)) {}

	[[nodiscard]] std::chrono::microseconds total_move_delay() const noexcept
	{
		return stage_move_delay + roi_move_delay;
	}

};

struct capture_item final : roi_name, scope_delays, scope_location_xyz
{
	scope_action action;
	bool sync_io;
	int channel_route_index;

	explicit capture_item(const roi_name& n, const scope_delays& del,
		const scope_location_xyz& loc, const int channel_route_index, const bool sync_io = false, const scope_action& a = scope_action::capture)
		noexcept : roi_name(n), scope_delays(del), scope_location_xyz(loc), action(a), sync_io(sync_io), channel_route_index(channel_route_index) {}

	capture_item() noexcept : action(scope_action::capture), sync_io(false), channel_route_index(0) {}

	[[nodiscard]] capture_item_2dt_hash get_2dt_hash(const int pattern_idx) const noexcept
	{
		return capture_item_2dt_hash(roi, repeat, channel_route_index, page, pattern_idx);
	}

	[[nodiscard]] capture_item_within_layer_hash get_within_layer_hash() const noexcept
	{
		return capture_item_within_layer_hash(row, column, time);
	}
};
enum class file_kind { image, roi, stats };

#ifdef QT_DLL
Q_DECLARE_METATYPE(capture_item)
Q_DECLARE_METATYPE(scope_delays)
#endif

#endif