#pragma once
#ifndef OLYMPUS_SCOPE_H
#define OLYMPUS_SCOPE_H

#include "scope.h"
class microscope_xy_drive_olympus final : public scope_xy_drive
{
	static constexpr auto micro_to_pdu = 100;
public:
	microscope_xy_drive_olympus();

	void move_to_xy_internal(const scope_location_xy&) override;

	scope_limit_xy get_stage_xy_limits_internal() override;

	scope_location_xy get_position_xy_internal() override;

	void print_settings(std::ostream&) override
	{
		//
	}
};

class microscope_z_drive_olympus final : public scope_z_drive
{
	static constexpr auto micro_to_pdu = 100;
public:
	void move_to_z_internal(const float z) override;

	scope_limit_z get_z_drive_limits_internal() override;

	float get_position_z_internal() override;
	microscope_z_drive_olympus();

	void print_settings(std::ostream&) override
	{
		//
	}
};

class microscope_channel_drive_olympus final : public scope_channel_drive
{
	struct olympus_channel_state
	{
		bool tl_shutter;
		bool rl_shutter;
		int reflector_turret;
	};
	std::vector<olympus_channel_state> channel_states;
	olympus_channel_state current_state;
	void move_to_state(const olympus_channel_state& channel_state);
public:
	microscope_channel_drive_olympus();

protected:
	void move_to_channel_internal(int) override;

	void print_settings(std::ostream&) override
	{
		//
	}

	int get_channel_internal() override
	{
		//use whatever is stored
		return current_light_path_.scope_channel;
	}

	int get_light_path_internal() override
	{
		return current_light_path_.light_path;
	}

	void toggle_lights(const bool enable) override
	{
		move_to_channel(enable ? phase_channel_idx : off_channel_idx);
	}

	void move_condenser_internal(const condenser_position&) override
	{

	}

	condenser_position get_condenser_internal() override
	{
		return static_cast<condenser_position>(current_light_path_);
	}

	void move_to_light_path_internal(int) override
	{
		//
	}
	condenser_nac_limits get_condenser_na_limit_internal() override
	{
		return condenser_nac_limits(0.09, 0.75, true);
	}
};


#endif