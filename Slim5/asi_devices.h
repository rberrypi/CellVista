#pragma once
#ifndef ASI_DEVICE_H
#define ASI_DEVICE_H
#include "scope.h"

class microscope_z_drive_asi final : public scope_z_drive
{
	float micron_to_internal_z_;
protected:
	void move_to_z_internal(float z) override;
	float get_position_z_internal() override;
	scope_limit_z get_z_drive_limits_internal() override;
public:
	microscope_z_drive_asi();
	virtual ~microscope_z_drive_asi() = default;
};

class microscope_xy_drive_asi final : public scope_xy_drive
{
	float micron_to_internal_x_, micron_to_internal_y_;
protected:
	void move_to_xy_internal(const scope_location_xy& xy) override;
	scope_location_xy get_position_xy_internal() override;
public:
	microscope_xy_drive_asi();
	scope_limit_xy get_stage_xy_limits_internal() override;
};


#endif