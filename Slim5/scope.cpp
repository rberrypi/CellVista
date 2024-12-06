#include "stdafx.h"
#include "scope.h"
#include <future>
#include "virtual_scope_devices.h"
#include <QTimer>
#include <iostream>
#include <boost/container/static_vector.hpp>
#include "qli_runtime_error.h"
// ReSharper disable CppUnusedIncludeDirective
#if ( BODY_TYPE==BODY_TYPE_NIKON) || (STAGE_TYPE==STAGE_TYPE_NIKON)
#include "nikon_devices.h"
#endif
#if ( BODY_TYPE==BODY_TYPE_ZEISS) || (STAGE_TYPE==STAGE_TYPE_ZEISS)
#include "zeiss_devices.h"
#endif
#if ( BODY_TYPE==BODY_TYPE_LEICA) || (STAGE_TYPE==STAGE_TYPE_LEICA)
#include "leica_devices.h"
#endif
#if ( BODY_TYPE==BODY_TYPE_ASI) || (STAGE_TYPE==STAGE_TYPE_ASI)
#include "asi_devices.h"
#endif
#if ( BODY_TYPE==BODY_TYPE_NIKON2) || (STAGE_TYPE==STAGE_TYPE_NIKON2)
#include "nikon_devices2.h"
#endif
#if ( BODY_TYPE==BODY_TYPE_NIKON2) || (STAGE_TYPE==STAGE_TYPE_NIKON2)
#include "nikon_devices2.h"
#endif
#if ( BODY_TYPE==BODY_TYPE_PI_Z) 
#include "pi_devices.h"
#endif
// ReSharper restore CppUnusedIncludeDirective

std::unique_ptr<scope_z_drive> microscope::get_microscope_z_drive()
{
	try
	{
#if BODY_TYPE == BODY_TYPE_VIRTUAL
		return std::make_unique<microscope_z_drive_virtual>();
#elif BODY_TYPE == BODY_TYPE_ZEISS
		return std::make_unique<scope_z_drive_zeiss>();
#elif BODY_TYPE == BODY_TYPE_NIKON
		return std::make_unique<microscope_z_drive_nikon>();
#elif BODY_TYPE == BODY_TYPE_LEICA
		return new microscope_z_drive_leica();
#elif BODY_TYPE == BODY_TYPE_ASI
		return new microscope_z_drive_asi();
#elif BODY_TYPE == BODY_TYPE_NIKON2
		return new microscope_z_drive_nikon2();
#elif BODY_TYPE == BODY_TYPE_PI_Z
		return new microscope_z_drive_pi();
#endif
	}
	catch (...)
	{
	}
	return std::make_unique<microscope_z_drive_virtual>();
}

std::unique_ptr<scope_xy_drive> microscope::get_scope_xy_drive()
{
	try
	{
#if STAGE_TYPE == STAGE_TYPE_VIRTUAL
		return std::make_unique<microscope_xy_drive_virtual>();
#elif STAGE_TYPE == STAGE_TYPE_ZEISS
		return std::make_unique<scope_xy_drive_zeiss>();
#elif STAGE_TYPE  == STAGE_TYPE_NIKON
		return std::make_unique<microscope_xy_drive_nikon>();
#elif STAGE_TYPE == STAGE_TYPE_LEICA
		return new microscope_xy_drive_leica();
#elif STAGE_TYPE == STAGE_TYPE_ASI
		return new microscope_xy_drive_asi();
#elif STAGE_TYPE == STAGE_TYPE_NIKON2
		return new microscope_xy_drive_nikon2();
#endif
	}
	catch (...)
	{
	}
	return std::make_unique<microscope_xy_drive_virtual>();
}

std::unique_ptr<scope_channel_drive> microscope::get_scope_channel_drive()
{
	try
	{
#if BODY_TYPE == BODY_TYPE_VIRTUAL
		return std::make_unique<microscope_channel_drive_virtual>();
#elif BODY_TYPE == BODY_TYPE_ZEISS
		return std::make_unique<scope_channel_drive_zeiss>();		
#elif BODY_TYPE == BODY_TYPE_NIKON
		return std::make_unique<microscope_channel_drive_nikon>();
#elif BODY_TYPE == BODY_TYPE_LEICA
		return new microscope_channel_drive_leica();
#elif BODY_TYPE == BODY_TYPE_ASI
		return new microscope_channel_drive_virtual();
#elif BODY_TYPE == BODY_TYPE_NIKON2
		return new microscope_channel_drive_nikon2();
#elif BODY_TYPE == BODY_TYPE_PI_Z
		return new microscope_channel_drive_virtual();
#endif
	}
	catch (...)
	{
		return std::make_unique<microscope_channel_drive_virtual>();
	}
}

microscope::microscope(QObject* parent) :
	QObject(parent), chan_drive(get_scope_channel_drive()), xy_drive(get_scope_xy_drive()), z_drive(get_microscope_z_drive())
{
	qRegisterMetaType<scope_location_xyz>("scope_location_xyz");
}

microscope::~microscope()
{
	std::cout << "Disconnecting Microscope..." << std::endl;
}

void microscope::print_settings(std::ostream& input) const
{
	chan_drive->print_settings(input);
	xy_drive->print_settings(input);
	z_drive->print_settings(input);
}

void microscope::static_async_launcher(const scope_async_actions& input, const bool async)
{

	if (input.size() == 1 && !async)
	{
		input.front()();
	}
	else
	{
		boost::container::static_vector<std::future<void>, 4> run_these;
		for (const auto& func : input)
		{
			run_these.push_back(std::async(std::launch::async, [func] {func(); }));
		}
		if (!async)
		{
			for (auto& item: run_these)
			{
				item.wait();
			}
		}
	}
}

void microscope::wait_for_pfs(const bool async, const bool stage_moving) const
{
	const auto focus_system_engaged = z_drive->get_focus_system_status(false) != scope_z_drive::focus_system_status::off;
	if (!async && stage_moving && focus_system_engaged)
	{
		//no change for 30 ticks means we're stable
		const auto burn_out_time = ms_to_chrono(2000);//two seconds
		const auto start = timestamp();
		auto stability_ticks = 0;
		const auto stability_ticks_required = 30;
		auto total_ticks = 0;
		//scope_z_drive::focus_system_status status;
		auto failed = false;
		// ReSharper disable once CppEntityAssignedButNoRead
		scope_z_drive::focus_system_status focus_system_status;
		while (true)
		{
			const auto is_stable = (focus_system_status = z_drive->get_focus_system_status(true)) != scope_z_drive::focus_system_status::moving;
			windows_sleep(ms_to_chrono(1));
			if (is_stable)
			{
				stability_ticks = stability_ticks + 1;
			}
			if (stability_ticks > stability_ticks_required)
			{
				break;
			}
			if (timestamp() - start > burn_out_time)
			{
				failed = true;
				break;
			}
			//
			total_ticks = total_ticks + 1;
		}
		if (failed)
		{
			std::cout << "PFS failed to stabilize" << std::endl;
		}
#if _DEBUG
		std::cout << "PFS waited for " << total_ticks << " ticks status is " << static_cast<int>(focus_system_status) << std::endl;
#endif
	}
}

void microscope::move_to(const microscope_move_action& pos, const bool async)
{
	scope_async_actions items_to_run;
	constexpr auto refresh = false;
	const auto condenser_moving = chan_drive->get_condenser(refresh) != static_cast<const condenser_position&>(pos);
	if (condenser_moving)
	{
		const auto move_function = [&] {chan_drive->move_condenser(pos); };
		items_to_run.push_back(move_function);
	}
	auto stage_moving = xy_drive->get_position_xy(refresh) != pos;
	if (stage_moving)
	{
		const auto move_function = [&] {xy_drive->move_to_xy(pos); };
		items_to_run.push_back(move_function);
	}
	const auto z_moving = z_drive->get_position_z(refresh) != pos.z;
	if (z_moving)
	{
		const auto move_function = [&] {z_drive->move_to_z(pos.z); };
		items_to_run.push_back(move_function);
	}
	// a delay is applied only when the stage moves, this compensates for things like shaking (which cause slightly different positions to be reported)
	const auto channel_moving = chan_drive->get_channel(refresh) != pos.scope_channel;
	const auto light_path_moving = chan_drive->get_light_path(refresh) != pos.light_path;
	if (light_path_moving || channel_moving || stage_moving && pos.stage_move_delay.count() > 0)
	{
		const auto move_function = [&] {
			if (stage_moving)
			{
				auto move_delay = pos.stage_move_delay;
				if (move_delay > shutter_time)
				{
					std::cout << "Stage move timeout " << move_delay.count() << " us" << std::endl;
					const auto start = timestamp();
					chan_drive->move_to_channel(scope_channel_drive_settings::off_channel_idx);
					chan_drive->move_to_light_path(pos.light_path);
					const auto duration = timestamp() - start;
					const auto duration_left = move_delay - duration;
					move_delay = std::max(ms_to_chrono(0), duration_left);
				}
				windows_sleep(move_delay);
			}
			chan_drive->move_to_channel(pos.scope_channel);
			chan_drive->move_to_light_path(pos.light_path);
		};
		items_to_run.push_back(move_function);
	}
	static_async_launcher(items_to_run, async);
	wait_for_pfs(async, stage_moving);
}

void microscope::move_to(const scope_location_xyz& pos, const bool async)
{
	scope_async_actions items_to_run;
	const auto stage_moving = xy_drive->get_position_xy(false) != pos;
	if (stage_moving)
	{
		const auto move_function = [&] {xy_drive->move_to_xy(pos); };
		items_to_run.push_back(move_function);
	}
	const auto z_moving = z_drive->get_position_z(false) != pos.z;
	if (z_moving)
	{
		const auto move_function = [&] {z_drive->move_to_z(pos.z); };
		items_to_run.push_back(move_function);
	}
	static_async_launcher(items_to_run, async);
	wait_for_pfs(async, stage_moving);
}

void microscope::move_light_path(const microscope_light_path& light_path, const bool async)
{
	constexpr auto refresh = false;
	scope_async_actions items_to_run;
	{
		const auto current_channel_idx = chan_drive->get_channel(refresh);
		const auto channel_moving = current_channel_idx != light_path.scope_channel;
		if (channel_moving)
		{
			const auto move_function = [&]
			{
				chan_drive->move_to_channel(light_path.scope_channel);
			};
			items_to_run.push_back(move_function);
		}}
	const auto light_path_moving = chan_drive->get_light_path(refresh) != light_path.light_path;
	if (light_path_moving)
	{
		const auto move_function = [&] {chan_drive->move_to_light_path(light_path.light_path); };
		items_to_run.push_back(move_function);
	}
	{
		const auto condenser_moving = chan_drive->get_condenser(refresh) != light_path;
		if (condenser_moving)
		{
			const auto move_function = [&] {chan_drive->move_condenser(light_path); };
			items_to_run.push_back(move_function);
		}
	}
	static_async_launcher(items_to_run, async);
}

microscope_state microscope::get_state(const bool refresh) const
{
	const auto xy = xy_drive->get_position_xy(refresh);
	const auto z = z_drive->get_position_z(refresh);
	const auto chan = chan_drive->get_channel(refresh);
	const auto light_path = chan_drive->get_light_path(refresh);
	const auto cond = chan_drive->get_condenser(refresh);
	return  microscope_state(scope_location_xyz(xy.x, xy.y, z), microscope_light_path(chan, light_path, cond));
}

QStringList microscope::get_light_path_names() const
{
	static auto return_me = [&] {
		QStringList values;
		for (const auto& name : chan_drive->light_path_names)
		{
			values << QString::fromStdString(name);
		}
		if (values.empty())
		{
			values << microscope::unused;
		}
		return values;
	}();
	return return_me;
}

QStringList microscope::get_condenser_settings_names() const
{
	static auto return_me = [&] {
		QStringList values;
		for (const auto& name : chan_drive->condenser_names)
		{
			values << QString::fromStdString(name);
		}
		if (values.empty())
		{
			values << microscope::unused;
		}
		return values;
	}();
	return return_me;
}

QStringList microscope::get_channel_settings_names() const
{
	static auto return_me = [&] {
		QStringList values;
		for (const auto& name : chan_drive->channel_names)
		{
			values << QString::fromStdString(name);
		}
		if (values.empty())
		{
			values << microscope::unused;
		}
		return values;
	}();
	return return_me;
}

const std::string scope_channel_drive::channel_off_str = "Off";
const std::string scope_channel_drive::channel_phase_str = "Phase";

const std::chrono::microseconds microscope::shutter_time = ms_to_chrono(4000);

void scope_xy_drive::move_to_xy(const scope_location_xy& xy)
{
	const auto position_changed = xy != get_position_xy(false);
	const auto point_inside = point_inside_range(xy);
	if (position_changed && point_inside)
	{
		move_to_xy_internal(xy);
		xy_current_ = xy;
	}
}

bool scope_limit_xy::point_inside_range(const scope_location_xy& pos) const
{
	if (!valid)
	{
		return true;
	}
	const auto inside = [](const auto value, const auto min, const auto max)
	{
		return (value >= min) && (value <= max);
	};
	const auto x_min = xy.left();
	const auto x_max = xy.right();
	const auto y_min = xy.top();
	const auto y_max = xy.bottom();
	const auto x_inside = inside(pos.x, x_min, x_max);
	const auto y_inside = inside(pos.y, y_min, y_max);
	return x_inside && y_inside;
}

void scope_xy_drive::common_post_constructor()
{
	xy_current_ = get_position_xy_internal();
	xy_limits_ = get_stage_xy_limits_internal();
}

void scope_z_drive::common_post_constructor()
{
	zee_limits_ = get_z_drive_limits_internal();
	z_current_ = get_position_z_internal();
}

void scope_z_drive::move_to_z(const float z)
{
	constexpr auto refresh = false;
	const auto position_changed = z != get_position_z(refresh);
	const auto pfs_off = get_focus_system_status(refresh) == focus_system_status::off;
	const auto point_inside_grid = point_inside_range(z);
	if (position_changed && point_inside_grid && pfs_off)
	{
		move_to_z_internal(z);
		z_current_ = z;
	}
}

void scope_channel_drive::move_condenser(const condenser_position& position)
{
	const auto moves = position.condenser_moves();
	const auto inside_limits = nac_inside_limits(position.nac);
	const auto do_move = moves && inside_limits && position.nac_position < condenser_names.size();
	if (do_move)
	{
		move_condenser_internal(position);
	}
	static_cast<condenser_position&>(current_light_path_) = position;
}

void scope_channel_drive::move_to_channel(const int channel_idx)
{
	//some checking for valid channel idx
	if (channel_idx != current_light_path_.scope_channel)
	{
		move_to_channel_internal(channel_idx);
		current_light_path_.scope_channel = channel_idx;
	}
}

void scope_channel_drive::move_to_light_path(const int light_path_idx)
{
	if (has_light_path)
	{
		if (light_path_idx != current_light_path_.light_path)
		{
			move_to_light_path_internal(light_path_idx);
			current_light_path_.light_path = light_path_idx;
#if _DEBUG
			{
				const auto where_we_moved = get_light_path_internal();
				if (where_we_moved != light_path_idx)
				{
					qli_runtime_error("Something Wrong");
				}
			}
#endif
		}
	}
}

void scope_channel_drive::common_post_constructor()
{
	has_light_path = !light_path_names.empty();
#if KILL_CONDENSER_NAC_CONTROL!=0
	this->has_nac = false;
#endif
#if KILL_PORT_SWITCHER!=0
	this->has_light_path = false;
#endif	
	nac_limits_ = get_condenser_na_limit_internal();
	current_light_path_.light_path = get_light_path_internal();
	static_cast<condenser_position&>(current_light_path_) = get_condenser_internal();
	const auto first_name = channel_names[0];
	const auto second_name = channel_names[1];
	if (first_name != channel_off_str || second_name != channel_phase_str)
	{
		qli_runtime_error("Channel drive built without required channels, that's no good");
	}
}