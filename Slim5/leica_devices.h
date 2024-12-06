#pragma once
#ifndef LEICA_DEVICES_H
#define LEICA_DEVICES_H
#include "scope.h"

struct leica_contrast_amalgamatrix;
struct leica_xy_drive;
struct leica_drive;

class microscope_z_drive_leica final : public scope_z_drive
{
	std::unique_ptr<leica_drive> zd_;
protected:
	void move_to_z_internal(float z) override;
	float get_position_z_internal() override;
	scope_limit_z get_z_drive_limits_internal() override;
public:
	microscope_z_drive_leica();
	void print_settings(std::ostream& input) noexcept override ;
};

class microscope_xy_drive_leica final : public scope_xy_drive
{
	std::unique_ptr<leica_xy_drive> xyd_;
protected:
	void move_to_xy_internal(const scope_location_xy& xy) override;
	scope_location_xy get_position_xy_internal() override;
public:
	microscope_xy_drive_leica();
	scope_limit_xy get_stage_xy_limits_internal() override;
	void print_settings(std::ostream& input)  override;
};

class microscope_channel_drive_leica final : public scope_channel_drive
{
	std::unique_ptr<leica_contrast_amalgamatrix> cmd_;

protected:
	void move_to_channel_internal(int channel_idx) override;
	void move_to_light_path_internal(int channel_idx) override;
	int get_light_path_internal() override;
	int get_channel_internal() override;

	void move_condenser_internal(const condenser_position& position) override;
	condenser_position get_condenser_internal() override;
	condenser_nac_limits get_condenser_na_limit_internal() override;

public:
	void toggle_lights(bool enable) override;
	microscope_channel_drive_leica();
	virtual ~microscope_channel_drive_leica();
	void print_settings(std::ostream& input) noexcept override;
};

#endif