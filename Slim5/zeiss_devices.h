#pragma once
#ifndef MTB_DEVICE
#define MTB_DEVICE
// DLLs from this are installed via the MTB 2011 RDK into the so-called "GAC" (see MTB manual)

#include "scope.h"

class mtb_devices;

class scope_z_drive_zeiss final : public scope_z_drive
{
protected:
	void move_to_z_internal(float z) override;
	float get_position_z_internal() override;
	scope_limit_z get_z_drive_limits_internal() override;
public:
	scope_z_drive_zeiss();
	virtual ~scope_z_drive_zeiss() = default;
	void print_settings(std::ostream& input) noexcept override;
};

class scope_xy_drive_zeiss final : public scope_xy_drive
{
protected:
	void move_to_xy_internal(const scope_location_xy& xy) override;
	scope_location_xy get_position_xy_internal() override;
public:
	scope_xy_drive_zeiss();
	scope_limit_xy get_stage_xy_limits_internal() override;
	virtual ~scope_xy_drive_zeiss() = default;
	void print_settings(std::ostream& input)  override;
};

struct foobar_zeiss_light_path_position
{
	//this is really fucked, we should have our light path contain a variable number of switcher elements but we don't because there isn't time and most of our obstructions are caused by Gabi 
	int side_port_idx;
	int tl_lamp_switch_idx;
	std::string name;

	[[nodiscard]] bool approx_equals(const foobar_zeiss_light_path_position& position) const
	{
		return this->side_port_idx == position.side_port_idx && this->tl_lamp_switch_idx == position.tl_lamp_switch_idx;
	}
	foobar_zeiss_light_path_position(const int side_port_idx, const int tl_lamp_switch_idx, const std::string& name = "") : side_port_idx(side_port_idx), tl_lamp_switch_idx(tl_lamp_switch_idx), name(name)
	{

	}
};

class scope_channel_drive_zeiss final : public scope_channel_drive
{
	static void toggle_tl(bool enable);
	static void toggle_rl(bool enable);
	void set_cube_position(int chan_internal, bool is_tl) const;
	std::vector<foobar_zeiss_light_path_position> light_path_index_to_position;
	static const int dummy_channels_offset = 2;
	static const int changer_off = 1, changer_on = 2;
	condenser_nac_limits get_condenser_na_limit_internal() override;

protected:
	void move_to_channel_internal(int channel_idx) override;
	void move_to_light_path_internal(int light_path_idx) override;
	int get_channel_internal() override;
	int get_light_path_internal() override;
	void move_condenser_internal(const condenser_position& position) override;
	condenser_position get_condenser_internal() override;

public:
	void toggle_lights(bool enable) override;
	scope_channel_drive_zeiss();
	virtual ~scope_channel_drive_zeiss() = default;
	void print_settings(std::ostream& input) noexcept override;
};


#endif