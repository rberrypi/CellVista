#pragma once
#ifndef NIKON_DEVICE2_H
#define NIKON_DEVICE2_H

#include "scope.h"
struct nikon_devices2;

class microscope_z_drive_nikon2 final : public scope_z_drive
{
protected:
	void move_to_z_internal(float z) override;
	float get_position_z_internal() override;
	scope_limit_z get_z_drive_limits_internal() override;
	focus_system_status get_focus_system_internal()override;

public:
	microscope_z_drive_nikon2();
	virtual ~microscope_z_drive_nikon2() = default;
	void print_settings(std::ostream& input) noexcept override;
};

class microscope_xy_drive_nikon2 final : public scope_xy_drive
{
protected:
	void move_to_xy_internal(const scope_location_xy& xy) override;
	scope_location_xy get_position_xy_internal() override;
public:
	microscope_xy_drive_nikon2();
	scope_limit_xy get_stage_xy_limits_internal() override;
	virtual ~microscope_xy_drive_nikon2() = default;
	void print_settings(std::ostream& input)  override;
};

class microscope_channel_drive_nikon2 final : public scope_channel_drive
{
	static void set_fl_position(int channel_idx);
protected:
	void move_to_channel_internal(int channel_idx) override;
	void move_to_light_path_internal(int light_path_idx) override
	{
		Q_UNUSED(light_path_idx);
	}
	int get_channel_internal() override;
	int get_light_path_internal() override
	{
		return current_light_path_.light_path;
	}

public:
	microscope_channel_drive_nikon2();
	void toggle_lights(bool enable) override;
	virtual ~microscope_channel_drive_nikon2() = default;
	void print_settings(std::ostream& input) noexcept override;
};

#endif