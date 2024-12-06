#include "stdafx.h"
#include "device_factory.h"
//Defaults
#include "virtual_slm.h"
#include "scope.h"
#include <iostream>
#include "virtual_camera_settings.h"
//
const auto io_settings_filename = "io_settings.json";

device_factory::~device_factory()
{
	Q_UNUSED(io_settings::write(io_settings_filename));
}

device_factory::device_factory(const virtual_camera_type psi_type) : camera_holder(psi_type,get_slm_count())
{
	scope = std::make_unique<microscope>(nullptr);
	scope->chan_drive->has_light_path = cameras.size() > 1 ? scope->chan_drive->has_light_path : false;
	std::cout << "Scope can switch light path? " << (scope->chan_drive->has_light_path ? "Yep" : "Nope") << std::endl;
	static_cast<io_settings&>(*this) = io_settings(io_settings_filename);
}

std::unique_ptr<device_factory> D;