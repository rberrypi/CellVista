#pragma once
#ifndef DEVICE_FACTORY_H
#define DEVICE_FACTORY_H
#include "virtual_camera_shared.h"
#include "slm_holder.h"
#include "camera_holder.h"
#include "background_update_functors.h"
class microscope;
//This guy contains the programs global state, typical data flow is:
// Write: GUI->DeviceFactory->Hardware
// Read:  <-DeviceFactory
//There was a dark time when Qt was incompatible with CUDA, so we used this shim. Yes, it would be good to replace.

#include "acquisition.h"
class device_factory final : public io_settings, public slm_holder, public camera_holder
{
	//todo move compute engine to here?
public:
	std::unique_ptr<microscope> scope;
	acquisition route;// do not create new routes, these structs can be pretty big when loaded
	[[nodiscard]] background_update_functors get_background_update_functors();
	explicit device_factory(virtual_camera_type psi_type);//build initial channel settings
	~device_factory();
};

extern std::unique_ptr<device_factory> D;
// ReSharper disable once CppInconsistentNaming
#endif