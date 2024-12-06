#pragma once
#ifndef VIRTUAL_SCOPE_H
#define VIRTUAL_SCOPE_H

#include "scope.h"

class microscope_xy_drive_virtual final : public scope_xy_drive
{
public:
	microscope_xy_drive_virtual()
	{
		xy_current_ = scope_location_xy(0, 0);
		common_post_constructor();
	}
	constexpr static auto communication_time = ms_to_chrono(12);
	constexpr static auto stage_warm_up = ms_to_chrono(100);
	constexpr static auto milliseconds_per_micron = ms_to_chrono(0.125);
	void move_to_xy_internal(const scope_location_xy& new_location) override
	{
		windows_sleep(communication_time);
		windows_sleep(stage_warm_up);
		const auto distance = hypot(xy_current_.x - new_location.x, xy_current_.y - new_location.y);
		windows_sleep(distance * milliseconds_per_micron);
	}

	scope_limit_xy get_stage_xy_limits_internal() override
	{
		windows_sleep(communication_time);
		const auto side = 50000;
		QRectF rect(0, 0, side, side);
		rect.moveCenter(QPointF(0, 0));
		return scope_limit_xy(rect);
	}

	scope_location_xy get_position_xy_internal() override
	{
		windows_sleep(communication_time);
		return xy_current_;
	}

	void print_settings(std::ostream&) override
	{
		//
	}
};

class microscope_z_drive_virtual final : public scope_z_drive
{
public:
	constexpr static auto communication_time = ms_to_chrono(5);
	constexpr static auto move_warmup = ms_to_chrono(100);
	constexpr static auto milliseconds_per_micron = ms_to_chrono(1);
	void move_to_z_internal(const float z) override
	{
		windows_sleep(communication_time);
		windows_sleep(move_warmup);
		const auto distance = abs(z - this->z_current_);
		windows_sleep(distance * milliseconds_per_micron);
	}

	scope_limit_z get_z_drive_limits_internal() override
	{
		windows_sleep(communication_time);
		return{ -1000, 1000 };
	}

	float get_position_z_internal() override
	{
		windows_sleep(communication_time);
		return z_current_;
	}
	microscope_z_drive_virtual()
	{
		z_current_ = 0;
	}

	void print_settings(std::ostream&) noexcept override 
	{
		//
	}
};

class microscope_channel_drive_virtual final : public scope_channel_drive
{
public:
	constexpr static auto communication_time = ms_to_chrono(7);
	constexpr static auto channel_switch_time = ms_to_chrono(1000);
	constexpr static auto condenser_switch_warm_up = ms_to_chrono(50);
	constexpr static auto condenser_position_switch_time = ms_to_chrono(500);
	constexpr static auto condenser_switch_speed_per_na = ms_to_chrono(1818);

	microscope_channel_drive_virtual()
	{
		has_nac = true;
		has_light_path = true;
		channel_names.push_back(channel_off_str);
		channel_names.push_back(channel_phase_str);
		channel_names.emplace_back("FITC");
		channel_names.emplace_back("TRITC");
		channel_names.emplace_back("GABI");
		channel_names.emplace_back("CY3");
		channel_names.emplace_back("DAPI");
		light_path_names.emplace_back("Left");
		light_path_names.emplace_back("Right");
		light_path_names.emplace_back("Bottom");
		light_path_names.emplace_back("80/20");
		condenser_names.emplace_back("BF");
		condenser_names.emplace_back("PC1");
		condenser_names.emplace_back("DICIV");
		common_post_constructor();
	}

protected:
	void move_to_channel_internal(const int new_channel) override
	{
		windows_sleep(communication_time);
		const auto change = this->current_light_path_.scope_channel != new_channel;
		if (change)
		{
			windows_sleep(channel_switch_time);
		}
	}

	void print_settings(std::ostream&) noexcept override
	{
		//
	}

	int get_channel_internal() override
	{
		windows_sleep(communication_time);
		return current_light_path_.scope_channel;
	}

	int get_light_path_internal() override
	{
		windows_sleep(communication_time);
		return current_light_path_.light_path;
	}

	void toggle_lights(const bool enable) override
	{
		move_to_channel(enable ? phase_channel_idx : off_channel_idx);
	}

	void move_condenser_internal(const condenser_position& position) override
	{
		windows_sleep(communication_time);
		const auto condenser_switched = this->current_light_path_.nac_position != position.nac_position;
		if (position.condenser_moves() && condenser_switched)
		{
			windows_sleep(condenser_position_switch_time);
		}
		const auto nac_motion = abs(position.nac - this->current_light_path_.nac);
		windows_sleep(nac_motion * condenser_switch_speed_per_na);
	}

	condenser_position get_condenser_internal() override
	{
		windows_sleep(communication_time);
		return static_cast<condenser_position>(current_light_path_);
	}

	void move_to_light_path_internal(const int new_position) override
	{
		windows_sleep(communication_time);
		const auto change = this->current_light_path_.light_path != new_position;
		if (change)
		{
			windows_sleep(condenser_position_switch_time);
		}
	}
	condenser_nac_limits get_condenser_na_limit_internal() override
	{
		windows_sleep(communication_time);
		return condenser_nac_limits(0.09, 0.75, true);
	}
};


#endif