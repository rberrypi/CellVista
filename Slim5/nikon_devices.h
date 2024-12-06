#pragma once
#ifndef NIKON_DEVICE_H
#define NIKON_DEVICE_H

#include "scope.h"
struct nikon_devices;

class microscope_z_drive_nikon final : public scope_z_drive
{
	float ztomicrons_;
protected:
	void move_to_z_internal(float z) override;
	float get_position_z_internal() override;
	scope_limit_z get_z_drive_limits_internal() override;
	focus_system_status get_focus_system_internal()override;

public:
	microscope_z_drive_nikon();
	virtual ~microscope_z_drive_nikon() = default;
	void print_settings(std::ostream& input) noexcept override;
};

class microscope_xy_drive_nikon final : public scope_xy_drive
{
	float stomicrons_;
protected:
	void move_to_xy_internal(const scope_location_xy& xy) override;
	[[nodiscard]] scope_location_xy get_position_xy_internal() override;
public:
	microscope_xy_drive_nikon();
	[[nodiscard]] scope_limit_xy get_stage_xy_limits_internal() override;
	virtual ~microscope_xy_drive_nikon() = default;
	void print_settings(std::ostream& input)  override;
};

class microscope_channel_drive_nikon final : public scope_channel_drive
{
	 void set_fl_position(int chan_internal);
	static std::vector<std::string> get_rl_channel_names();
	static const int dummy_channels_offset = 2;
	static const int changer_off = 1, changer_on = 2;

protected:
	void move_to_channel_internal(int channel_idx) override;
	void move_to_light_path_internal(int) override;
	int get_channel_internal() override;
	int get_light_path_internal() override;
	static void toggle_tl(bool enable);
	static void toggle_rl(bool enable);
public:
	void toggle_lights(bool enable) override;
	microscope_channel_drive_nikon();
	virtual ~microscope_channel_drive_nikon() = default;
	void print_settings(std::ostream& input) noexcept override;
protected:
	void move_condenser_internal(const condenser_position&) override
	{
		//not supported
	}
	condenser_position get_condenser_internal() override
	{
		return condenser_position();
	}
	condenser_nac_limits get_condenser_na_limit_internal() override
	{
		return condenser_nac_limits();
	}
};

#endif