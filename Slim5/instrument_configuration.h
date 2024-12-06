#pragma once
#ifndef INSTRUMENT_CONFIGURATION_H
#define INSTRUMENT_CONFIGURATION_H
#include <boost/container/small_vector.hpp>
#include "approx_equals.h"

struct scope_location_xy
{
	constexpr static float null() noexcept
	{
		return std::numeric_limits<float>::quiet_NaN();
	}
	float x, y;
	scope_location_xy(const float x, const float y) noexcept : x(x), y(y) {}
	scope_location_xy() noexcept : scope_location_xy(null(), null()) {}
	[[nodiscard]] bool operator== (const scope_location_xy& b) const noexcept
	{
		return x == b.x && y == b.y;
	}
	[[nodiscard]] bool operator!= (const scope_location_xy& b) const noexcept
	{
		return !(*this == b);
	}
};

struct scope_location_xyz : scope_location_xy
{
	float z;
	explicit scope_location_xyz() noexcept : scope_location_xyz(null(), null(), null()) {}//teh c++11
	explicit scope_location_xyz(const float x, const float y, const float z) noexcept : scope_location_xy(x, y), z(z)
	{
	}

	[[nodiscard]] bool is_valid() const;
};

struct condenser_position
{
	float nac;
	int nac_position;
	[[nodicsard]] static int invalid_nac_position() noexcept
	{
		return (-1);
	}

	[[nodiscard]] bool condenser_moves() const noexcept
	{
		return nac_position != invalid_nac_position();
	}
	condenser_position(const float nac, const int position) noexcept : nac(nac), nac_position(position) {}
	condenser_position() noexcept : condenser_position(0.0f, condenser_position::invalid_nac_position()) {}

	[[nodiscard]] bool item_approx_equals(const condenser_position& b) const noexcept
	{
		return approx_equals(nac, b.nac) && nac_position == b.nac_position;
	}
	[[nodiscard]] bool operator== (const condenser_position& b) const noexcept
	{
		return nac == b.nac && nac_position == b.nac_position;
	}
	[[nodiscard]] bool operator!= (const condenser_position& b) const noexcept
	{
		return !(*this == b);
	}
};

struct microscope_light_path : condenser_position
{
	int scope_channel, light_path; // internal indices
	[[nodiscard]] bool operator== (const microscope_light_path& b) const noexcept
	{
		return scope_channel == b.scope_channel && light_path == b.light_path && static_cast<const condenser_position&>(*this) == b;
	}

	[[nodiscard]] bool item_approx_equals(const microscope_light_path& b) const noexcept
	{
		return scope_channel == b.scope_channel && light_path == b.light_path && condenser_position::item_approx_equals(b);
	}
	microscope_light_path(const int scope_channel, const int light_path, const  condenser_position& position) noexcept : condenser_position(position), scope_channel(scope_channel), light_path(light_path) {}
	microscope_light_path() noexcept : microscope_light_path(0, 0, condenser_position()) {}
};


struct microscope_state : scope_location_xyz, microscope_light_path
{
	explicit microscope_state(const scope_location_xyz& location, const microscope_light_path& light_path) noexcept :
		scope_location_xyz(location), microscope_light_path(light_path) { }

	microscope_state() noexcept : microscope_state(scope_location_xyz(), microscope_light_path()) {}
};

struct microscope_move_action : microscope_state
{
	std::chrono::microseconds stage_move_delay;
	explicit microscope_move_action(const microscope_state& loc, const std::chrono::microseconds move_delay) noexcept :
		microscope_state(loc), stage_move_delay(move_delay)
	{
	}
	microscope_move_action() noexcept : microscope_move_action(microscope_state(), std::chrono::microseconds(0)) {}
};

typedef boost::container::small_vector<int, 3> fl_channel_index_list;
enum class scope_action { capture, focus, set_bg_for_this_channel, shutdown_live };

#ifdef QT_CORE_LIB
#include <QMetaType>
Q_DECLARE_METATYPE(condenser_position)
Q_DECLARE_METATYPE(fl_channel_index_list)
#endif

#endif