#include "stdafx.h"
#if (BODY_TYPE==BODY_TYPE_OLYMPUS) || (STAGE_TYPE==STAGE_TYPE_OLYMPUS) || (BUILD_ALL_DEVICES_TARGETS)
#include "olympus_scopes.h"
#include "time_slice.h"
#include <boost/core/noncopyable.hpp>
#include <gt.h>
#include <iostream>
#include <mutex>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

#define OLYMPUS_SAFE_CALL( err ) olympus_safe_call(err, __FILE__, __LINE__ )
template<typename T>
T olympus_safe_call(T value, const char* file, int line)
{
	if (value == NULL)
	{
		std::cout << "Olympus Error: " << file << ":" << line << std::endl;
		//call last error
	}
	return value;
}

int error_handler()
{
	return 1;
}

std::mutex protect_notification;
std::condition_variable wait_for_notification;
bool consumed_notification = false;
//todo callback body function

int	CALLBACK CommandCallback(
	ULONG		MsgId,			// Callback ID.
	ULONG		wParam,			// Callback parameter, it depends on callback event.
	ULONG		lParam,			// Callback parameter, it depends on callback event.
	PVOID		pv,				// Callback parameter, it depends on callback event.
	PVOID		pContext,		// This contains information on this call back function.
	PVOID		pCaller			// This is the pointer specified by a user in the requirement function.
)
{
	try
	{
		{
			std::unique_lock lk(protect_notification);
			consumed_notification = true;
		}
		wait_for_notification.notify_one();
		return	0;
	}
	catch (...)
	{
		return  error_handler();
	}
}

int	CALLBACK NotifyCallback(
	ULONG		MsgId,			// Callback ID.
	ULONG		wParam,			// Callback parameter, it depends on callback event.
	ULONG		lParam,			// Callback parameter, it depends on callback event.
	PVOID		pv,				// Callback parameter, it depends on callback event.
	PVOID		pContext,		// This contains information on this call back function.
	PVOID		pCaller			// This is the pointer specified by a user in the requirement function.
)
{
	try
	{
		{
			std::unique_lock lk(protect_notification);
			consumed_notification = true;
		}
		wait_for_notification.notify_one();
		return	0;
	}
	catch (...)
	{
		return error_handler();
	}
}

int	CALLBACK ErrorCallback(
	ULONG		MsgId,			// Callback ID.
	ULONG		wParam,			// Callback parameter, it depends on callback event.
	ULONG		lParam,			// Callback parameter, it depends on callback event.
	PVOID		pv,				// Callback parameter, it depends on callback event.
	PVOID		pContext,		// This contains information on this call back function.
	PVOID		pCaller			// This is the pointer specified by a user in the requirement function.
)
{
	try
	{
		{
			std::unique_lock lk(protect_notification);
			consumed_notification = true;
		}
		wait_for_notification.notify_one();
		return	0;
	}
	catch (...)
	{
		return error_handler();
	}
}

struct command_result_pair
{
	std::string command;
	std::string required_result;
};
struct olympus_implementation : private boost::noncopyable
{
	fn_MSL_PM_SendCommand pfn_SendCommand;
	void* first_interface;
	olympus_implementation()
	{
		try
		{
			const auto m_hMod = OLYMPUS_SAFE_CALL(LoadLibrary(L"msl_pm.dll"));
			auto pfn_InifInterface = reinterpret_cast<fn_MSL_PM_Initialize>(OLYMPUS_SAFE_CALL(GetProcAddress(m_hMod, "MSL_PM_Initialize")));
			auto pfn_EnumInterface = reinterpret_cast<fn_MSL_PM_EnumInterface>(OLYMPUS_SAFE_CALL(GetProcAddress(m_hMod, "MSL_PM_EnumInterface")));
			auto pfn_GetInterfaceInfo = reinterpret_cast<fn_MSL_PM_GetInterfaceInfo>(OLYMPUS_SAFE_CALL(GetProcAddress(m_hMod, "MSL_PM_GetInterfaceInfo")));
			auto pfn_OpenInterface = reinterpret_cast<fn_MSL_PM_OpenInterface>(OLYMPUS_SAFE_CALL(GetProcAddress(m_hMod, "MSL_PM_OpenInterface")));
			auto pfn_CloseInterface = reinterpret_cast<fn_MSL_PM_CloseInterface>(OLYMPUS_SAFE_CALL(GetProcAddress(m_hMod, "MSL_PM_CloseInterface")));
			pfn_SendCommand = reinterpret_cast<fn_MSL_PM_SendCommand>(OLYMPUS_SAFE_CALL(GetProcAddress(m_hMod, "MSL_PM_SendCommand")));
			auto pfn_RegisterCallback = reinterpret_cast<fn_MSL_PM_RegisterCallback>(OLYMPUS_SAFE_CALL(GetProcAddress(m_hMod, "MSL_PM_RegisterCallback")));
			//
			auto init_ret = pfn_InifInterface();
			const auto interfaces = pfn_EnumInterface();
			std::vector<void*> if_data;
			for (auto i = 0; i < interfaces; ++i)
			{
				void* ptr;
				const auto interface_info_code = pfn_GetInterfaceInfo(i, &ptr);
				if_data.push_back(ptr);
			}
			first_interface = if_data.front();
			const auto interface_open_success = pfn_OpenInterface(first_interface);
			if (!interface_open_success)
			{
				throw std::runtime_error("Failed to Open Olympus Interface");
			}
			MDK_MSL_CMD command;
			const auto call_back_registration_sucess = pfn_RegisterCallback(first_interface, CommandCallback, NotifyCallback, ErrorCallback, &command);
			{
				safe_issue_command({ "L 1,1" ,"L +" });
				safe_issue_command({ "L?" ,"L 1" });
				safe_issue_command({ "DG 1,1","DG +" });
				safe_issue_command({ "EN6 1,1","EN6 +" });
				safe_issue_command({ "EN5 1","EN5 +" });
				safe_issue_command({ "DG 0,1","DG +" });
				safe_issue_command({ "OPE 0","OPE +" });

			}
		}
		catch (...)
		{
			error_handler();
		}
	}
	std::mutex one_command_at_a_time;
	std::string issue_command(const std::string& command_text)
	{
			std::unique_lock lk2(one_command_at_a_time);
#if 0
			time_slice ts(command_text);
#endif
			std::unique_lock lk(protect_notification);
			MDK_MSL_CMD command;
			std::memset(&command, 0x00, sizeof(MDK_MSL_CMD));
			const auto full_command = command_text + "\r\n";
			const auto len = full_command.size();
			const auto s = full_command.c_str();
			std::memcpy(command.m_Cmd, (LPCSTR)s, len + 1);
			command.m_CmdSize = len;
			command.m_Callback = CommandCallback;
			command.m_Context = NULL;		// this pointer passed by pv
			command.m_Timeout = 30 * 1000;	// (ms)
			command.m_Sync = FALSE;
			command.m_Command = TRUE;		// TRUE: Command , FALSE: it means QUERY form ('?').
			consumed_notification = false;
			const auto success = pfn_SendCommand(first_interface, &command);
			wait_for_notification.wait(lk, [&] {
				return consumed_notification;
			});
			std::string to_return(command.m_RspSize, 'X');
			std::copy(command.m_Rsp, command.m_Rsp + command.m_RspSize, to_return.begin());
			//std::cout << "Sent:" << command_text << " " << to_return << std::endl;
			return to_return;
	}
	void safe_issue_command(const command_result_pair& pair)
	{
		const auto result = issue_command(pair.command);
		const auto match = result == pair.required_result;
		if (!match && !result.empty())
		{
			std::cout << "Warning: Sent:" << pair.command << " " << " Received:" << result << " Expected:" << pair.required_result << std::endl;
		}
	};
	~olympus_implementation()
	{
		safe_issue_command({ "OPE 1","OPE +" });
		safe_issue_command({ "L 0,0","L +" });
	}
};

std::unique_ptr<olympus_implementation> impl;

microscope_xy_drive_olympus::microscope_xy_drive_olympus()
{
	xy_current_ = scope_location_xy(0, 0);
	//
	if (!impl)
	{
		impl = std::make_unique<olympus_implementation>();
	}
	common_post_constructor();
}

scope_location_xy microscope_xy_drive_olympus::get_position_xy_internal()
{
	const auto result = impl->issue_command("XYP?");
	float x_pdu, y_pdu;
	std::sscanf(result.c_str(), "XYP %f,%f", &x_pdu, &y_pdu);
	auto xy = scope_location_xy(x_pdu / micro_to_pdu, y_pdu / micro_to_pdu);
	return xy;
}

scope_limit_xy microscope_xy_drive_olympus::get_stage_xy_limits_internal()
{
	scope_limit_xy xy;
	float left, right, top, bottom;
	{
		const auto result = impl->issue_command("XRANGE?");
		std::sscanf(result.c_str(), "XRANGE %f,%f", &left, &right);
	}
	{
		const auto result = impl->issue_command("YRANGE?");
		std::sscanf(result.c_str(), "YRANGE %f,%f", &bottom, &top);
	}
	const auto left_fixed = left / micro_to_pdu;
	xy.xy.setLeft(left_fixed);
	const auto right_fixed = right / micro_to_pdu;
	xy.xy.setRight(right_fixed);
	const auto top_fixed = top / micro_to_pdu;
	xy.xy.setTop(top_fixed);
	const auto bot_fixed = bottom / micro_to_pdu;
	xy.xy.setBottom(bot_fixed);
	xy.xy = xy.xy.normalized();
	xy.valid = true;
	return xy;
}

void microscope_xy_drive_olympus::move_to_xy_internal(const scope_location_xy& xy)
{
	const int x = xy.x * micro_to_pdu;
	const int y = xy.y * micro_to_pdu;
	const auto values = boost::format("XYG %d,%d") % x % y;
	impl->safe_issue_command({ values.str() ,"XYG +" });
}

void microscope_z_drive_olympus::move_to_z_internal(const float z)
{
	const int z_pdu = z * micro_to_pdu;
	const auto values = boost::format("FG %d") % z_pdu;
	impl->safe_issue_command({ values.str() ,"FG +" });
	z_current_ = z;
}

scope_limit_z microscope_z_drive_olympus::get_z_drive_limits_internal()
{
	constexpr auto hard_coded_microns = 3215;
	return { 0, hard_coded_microns };
}

float microscope_z_drive_olympus::get_position_z_internal()
{
	const auto result = impl->issue_command("FP?");
	float in_pdu = 0;
	std::sscanf(result.c_str(), "FP %f", &in_pdu);
	const auto value = in_pdu / micro_to_pdu;
	return value;
}

microscope_z_drive_olympus::microscope_z_drive_olympus()
{
	z_current_ = 0;
	if (!impl)
	{
		impl = std::make_unique<olympus_implementation>();
	}
	common_post_constructor();
}

void microscope_channel_drive_olympus::move_to_state(const olympus_channel_state& channel_state)
{
	{
		const auto move_tl = channel_state.tl_shutter != this->current_state.tl_shutter;
		if (move_tl)
		{
			const auto command = channel_state.tl_shutter ? "DSH 0" : "DSH 1";
			impl->safe_issue_command({ command,"DSH +" });
			this->current_state.tl_shutter = channel_state.tl_shutter;
		}
	}
	//
	{
		const auto move_rl = channel_state.rl_shutter != this->current_state.rl_shutter;
		if (move_rl)
		{
			const auto command = channel_state.rl_shutter ? "ESH1 0" : "ESH1 1";
			impl->safe_issue_command({ command,"ESH1 +" });
			this->current_state.rl_shutter = channel_state.rl_shutter;
		}
	}
	//
	{
		const auto move_reflector = channel_state.reflector_turret != (-1) && channel_state.reflector_turret != this->current_state.reflector_turret;
		if (move_reflector)
		{
			const auto command = "MU1 " + std::to_string(channel_state.reflector_turret);
			impl->safe_issue_command({ command,"MU1 +" });
			this->current_state.reflector_turret = channel_state.reflector_turret;
		}
	}
}

microscope_channel_drive_olympus::microscope_channel_drive_olympus()
{
	if (!impl)
	{
		impl = std::make_unique<olympus_implementation>();
	}
	channel_names.push_back(channel_off_str);
	channel_states.push_back({ false,false,-1 });
	const auto magic_pc_position = 6;
	channel_names.push_back(channel_phase_str);
	channel_states.push_back({ true,false,magic_pc_position });

	for (auto i = 0; i < 9; ++i)
	{
		const auto internal_idx = i + 1;
		const auto query_command = "GMU1 " + std::to_string(internal_idx);
		auto result = impl->issue_command(query_command);
		std::vector<std::string> results;
		boost::split(results, result, [](const char c) {return c == ','; });
		if (results.size() != 4)
		{
			break;
		}
		const auto& name = results.at(2);
		channel_names.push_back(name);
		channel_states.push_back({ false,true,internal_idx });
	}
	light_path_names.emplace_back("Not Implemented");
	condenser_names.emplace_back("Not Implemented");
	this->current_state = olympus_channel_state({ true,true,-2 });
	move_to_state({ false,false,magic_pc_position });
	common_post_constructor();
}

void microscope_channel_drive_olympus::move_to_channel_internal(int idx)
{
	const auto& state = channel_states.at(idx);
	move_to_state(state);
}

#endif