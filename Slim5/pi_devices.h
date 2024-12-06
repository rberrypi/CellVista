#pragma once
#ifndef PI_DEVICES_H
#define PI_DEVICES_H
#include "scope.h"
class microscope_z_drive_pi final : public scope_z_drive
{
protected:
	void move_to_z_internal(float z) override;
	float get_position_z_internal() override;
	scope_limit_z get_z_drive_limits_internal() override;
	focus_system_status get_focus_system_internal()override;
	int usb_id_;
public:
	microscope_z_drive_pi();
	virtual ~microscope_z_drive_pi();
	void print_settings(std::ostream& input) noexcept override;
};
#endif