#pragma once
#ifndef SAVE_POSITION_H
#define SAVE_POSITION_H


#include <boost/core/noncopyable.hpp>
#include "slm_holder.h"
#include "slm_device.h"
#include "compute_and_scope_state.h"

class camera_device;
class slm_holder;
class microscope;

struct save_position_scope : boost::noncopyable
{
	microscope* scope;
	const microscope_state pos_scope;
	explicit save_position_scope(microscope* scope);
	~save_position_scope();
};

struct camera_state_pair
{
	camera_device* camera;
	camera_config camera_config;
	std::chrono::microseconds exposure;
	camera_state_pair() :camera(nullptr), exposure(ms_to_chrono(0)) {}
};

struct save_position_cameras : boost::noncopyable
{
	std::vector<camera_state_pair> configs;
	explicit save_position_cameras(const std::vector<camera_device*>& cameras);
	~save_position_cameras();
};

struct save_slm_positions : boost::noncopyable
{
	const slm_states state;
	slm_holder* slms;
	explicit save_slm_positions(slm_holder* slms);
	~save_slm_positions();
};

struct save_device_state final : save_position_cameras, save_slm_positions, save_position_scope
{
	explicit save_device_state(const std::vector<camera_device*>& cameras, slm_holder* slms, microscope* scope) :
		save_position_cameras(cameras), save_slm_positions(slms), save_position_scope(scope)
	{

	}
};
#endif