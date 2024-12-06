#include "stdafx.h"
#if (BODY_TYPE==BODY_TYPE_NIKON) || (STAGE_TYPE==STAGE_TYPE_NIKON) || (BUILD_ALL_DEVICES_TARGETS)
#include <boost/core/noncopyable.hpp>
#include "nikon_devices.h"
#import <NikonTi.dll> named_guids
#import <MipParam2.dll> named_guids
#include <atlbase.h>
#include "time_slice.h"
#include "atlstr.h"
#include <future>
#include <QTimer>
#include <iostream>
#include <sstream>
#include "qli_runtime_error.h"
#include "time_slice.h"
#include "com_persistent_device.h"
void nikon_error(_com_error* e, const char* file, const int line)
{
	//
	CString message, description;
	const auto lower = static_cast<HRESULT>(TISCOPELib::TISCOPE_DEVICE_ERROR_BASE);
	const auto upper = static_cast<HRESULT>(TISCOPELib::TISCOPE_DATABASE_ERROR_BASE);
	const auto code = e->Error();
	if (e && (code >= lower && code <= upper))
	{
		BSTR msg;
		auto info = e->ErrorInfo();
		if (info)
		{
			info->GetDescription(&msg);
			info->Release();
			message = CString(msg);
		}
	}
	// other com errors
	else
	{
		message = e->ErrorMessage();
		const BSTR msg = e->Description();
		description = CString(msg);
	}
	const auto message_a = CStringA(message);
	const auto description_a = CStringA(description);
	std::stringstream ss;
	ss << "NIKON Error: " << message_a << ":" << description_a << " " << file << ":" << line << " CODE" << code << std::endl;;
}

#define NIKON_ERROR(err) nikon_error(err,__FILE__,__LINE__)
#define NIKONTRY try{
#define NIKONCATCH }catch (_com_error e){NIKON_ERROR(&e);}

struct nikon_devices final : boost::noncopyable
{

	TISCOPELib::INikonTiPtr NikonTiPtr;
	TISCOPELib::INikonTiDevice* NikonTiDevice;
	TISCOPELib::INosepiece* Nosepiece;
	TISCOPELib::ICondenserCassette* CondenserCassette;
	TISCOPELib::IFilterBlockCassette* FilterBlockCassette1;
	TISCOPELib::IFilterBlockCassette* FilterBlockCassette2;
	TISCOPELib::IBarrierFilterWheel* BarrierFilterWheel;
	TISCOPELib::IExcitationFilterWheel* ExcitationFilterWheel;
	TISCOPELib::ILightPathDrive* LightPathDrive;
	TISCOPELib::IZDrive* ZDrive;
	TISCOPELib::IPiezoZDrive* PiezoZDrive;
	TISCOPELib::IXDrive* XDrive;
	TISCOPELib::ITIRF* TIRF;
	TISCOPELib::IPFS* PFS;
	TISCOPELib::IDiaLamp* DiaLamp;
	TISCOPELib::IFiberIlluminator* FiberIlluminator;
	TISCOPELib::IDiaShutter* DiaShutter;
	TISCOPELib::IEpiShutter* EpiShutter;
	TISCOPELib::IAuxShutter* AuxShutter;
	TISCOPELib::IMainController* MainController;
	TISCOPELib::IRemoteController* RemoteController;
	TISCOPELib::IErgoController* ErgoController;
	TISCOPELib::IDSC* DSC1;
	TISCOPELib::IDSC* DSC2;
	TISCOPELib::IAnalyzer* Analyzer;
	TISCOPELib::IYDrive* YDrive;
	std::unique_ptr<com_persistent_device> intensilight;
	nikon_devices()
	{
		CoUninitialize();//https://bugreports.qt-project.org/browse/QTBUG-40632
		const auto a = CoInitializeEx(nullptr, COINIT_SPEED_OVER_MEMORY | COINIT_MULTITHREADED);
		if (a != S_OK)
		{
			const std::string error_msg = "Can't initialize COM on Nikon microscope";
			qli_runtime_error(error_msg);
		}
		{
			std::cout << "Nikon Initialized Status: " << a << std::endl;
		}
		NIKONTRY

			NikonTiPtr.CreateInstance(TISCOPELib::CLSID_NikonTi);

		//make sure these guys are called because they initialize the hardware
		// ReSharper disable CppDeclaratorNeverUsed

		volatile auto NikonTiDevice_okay = NikonTiPtr->get_Device(&NikonTiDevice);
		//
		volatile auto Nosepiece_okay = NikonTiPtr->get_Nosepiece(&Nosepiece);
		const bool Nosepiece_has = Nosepiece->GetIsMounted();//has
		//
		volatile auto CondenserCassette_okay = NikonTiPtr->get_CondenserCassette(&CondenserCassette);
		const bool CondenserCassette_has = CondenserCassette->GetIsMounted();
		//
		volatile auto FilterBlockCassette1_okay = NikonTiPtr->get_FilterBlockCassette1(&FilterBlockCassette1);
		const bool FilterBlockCassette1_has = FilterBlockCassette1->GetIsMounted();//has
		//
		volatile auto FilterBlockCassette2_okay = NikonTiPtr->get_FilterBlockCassette2(&FilterBlockCassette2);
		const bool FilterBlockCassette2_has = FilterBlockCassette2->GetIsMounted();
		//
		volatile auto BarrierFilterWheel_okay = NikonTiPtr->get_BarrierFilterWheel(&BarrierFilterWheel);
		const bool BarrierFilterWheel_has = BarrierFilterWheel->GetIsMounted();
		//
		volatile auto ExcitationFilterWheel_okay = NikonTiPtr->get_ExcitationFilterWheel(&ExcitationFilterWheel);
		const bool ExcitationFilterWheel_has = ExcitationFilterWheel->GetIsMounted();
		//
		volatile auto LightPathDrive_okay = NikonTiPtr->get_LightPathDrive(&LightPathDrive);
		const bool LightPathDrive_has = LightPathDrive->GetIsMounted();//has
		//
		volatile auto ZDrive_okay = NikonTiPtr->get_ZDrive(&ZDrive);
		const bool ZDrive_has = ZDrive->GetIsMounted();//has
		//
		volatile auto ZPiezoZDrive_okay = NikonTiPtr->get_PiezoZDrive(&PiezoZDrive);
		const bool PiezoZDrive_has = PiezoZDrive->GetIsMounted();
		//
		volatile auto XDrive_okay = NikonTiPtr->get_XDrive(&XDrive);
		const bool XDrive_has = XDrive->GetIsMounted();//has
		//
		volatile auto XTIRF_okay = NikonTiPtr->get_TIRF(&TIRF);
		const bool TIRF_has = TIRF->GetIsMounted();
		//
		volatile auto PFS_okay = NikonTiPtr->get_PFS(&PFS);
		const bool PFS_has = PFS->GetIsMounted();//has
		//
		volatile auto DiaLamp_okay = NikonTiPtr->get_DiaLamp(&DiaLamp);
		const bool DiaLamp_has = DiaLamp->GetIsMounted();//has
		//
		volatile auto FiberIlluminator_okay = NikonTiPtr->get_FiberIlluminator(&FiberIlluminator);
		const bool FiberIlluminator_has = FiberIlluminator->GetIsMounted();
		//
		volatile auto DiaShutter_okay = NikonTiPtr->get_DiaShutter(&DiaShutter);
		const bool DiaShutter_has = DiaShutter->GetIsMounted();
		//
		volatile auto EpiShutter_okay = NikonTiPtr->get_EpiShutter(&EpiShutter);
		const bool EpiShutter_has = EpiShutter->GetIsMounted();
		//
		volatile auto AuxShutter_okay = NikonTiPtr->get_AuxShutter(&AuxShutter);
		const bool AuxShutter_has = AuxShutter->GetIsMounted();
		//
		volatile auto MainController_okay = NikonTiPtr->get_MainController(&MainController);
		const bool MainController_has = MainController->GetIsMounted();//has
		//
		volatile auto RemoteController_okay = NikonTiPtr->get_RemoteController(&RemoteController);
		const bool RemoteController_has = RemoteController->GetIsMounted();
		//
		volatile auto ErgoController_okay = NikonTiPtr->get_ErgoController(&ErgoController);
		const bool ErgoController_has = ErgoController->GetIsMounted();
		//
		volatile auto DSC1_okay = NikonTiPtr->get_DSC1(&DSC1);
		const bool DSC1_has = DSC1->GetIsMounted();
		//
		volatile auto DSC2_okay = NikonTiPtr->get_DSC2(&DSC2);
		const bool DSC2_has = DSC2->GetIsMounted();
		//
		volatile auto Analyzer_okay = NikonTiPtr->get_Analyzer(&Analyzer);
		const bool Analyzer_has = Analyzer->GetIsMounted();
		//
		volatile auto YDrive_okay = NikonTiPtr->get_YDrive(&YDrive);
		const bool YDrive_has = YDrive->GetIsMounted();//has
		const auto here = 0;
		//com_persistent_device(std::string  device_name, int preferred_baud_rate, int com_port_number, std::string  terminator, const std::string& full_path_to_arduino_binary)
		try
		{
			intensilight = std::make_unique<com_persistent_device>("intensilight", CBR_9600, com_persistent_device::com_number_unspecified, "\r");
		}
		catch (...)
		{
			intensilight = nullptr;
		}
		//MIPPARAMLib::IMipParameterPtr pfs_sink = pfs_system->IsEnabled;
		//pfs_sink->AdviseAsync;
		NIKONCATCH
	}
	~nikon_devices()
	{
		//maybe fill out more of these later
		YDrive->Release();
		Analyzer->Release();
		DSC2->Release();
		DSC1->Release();
		ErgoController->Release();
		RemoteController->Release();
		MainController->Release();
		AuxShutter->Release();
		EpiShutter->Release();
		DiaShutter->Release();
		FiberIlluminator->Release();
		DiaLamp->Release();
		PFS->Release();
		TIRF->Release();
		XDrive->Release();
		PiezoZDrive->Release();
		ZDrive->Release();
		LightPathDrive->Release();
		ExcitationFilterWheel->Release();
		BarrierFilterWheel->Release();
		FilterBlockCassette2->Release();
		FilterBlockCassette1->Release();
		CondenserCassette->Release();
		Nosepiece->Release();
		//NikonTiDevice->Release();
		NikonTiPtr.Release();
		CoUninitialize();
	}
};

nikon_devices device_handle;
void microscope_z_drive_nikon::move_to_z_internal(const float z)
{
	NIKONTRY
		const long to_internal = z / ztomicrons_;
	const auto try_move = [&](auto pos) {
		try
		{
			device_handle.ZDrive->MoveAbsolute(pos);
			return true;
		}
		catch (...)
		{
			return false;
		}
	};
	const auto retry_efforts = 1000;
	auto attempt = 0;
	for (; attempt <= retry_efforts; ++attempt)
	{
		const auto actual_position = to_internal + attempt * 10;
		if (try_move(actual_position))
		{
			break;
		}
		std::cout << "Failed to move to " << z << " retrying to move to " << actual_position * ztomicrons_ << std::endl;
		windows_sleep(ms_to_chrono(1));
	}
	if (attempt > retry_efforts / 2)
	{
		std::cout << "Failed to find a valid z position, giving up" << std::endl;
	}
	NIKONCATCH
}

float microscope_z_drive_nikon::get_position_z_internal()
{
	NIKONTRY
		long zl = 0;
	device_handle.ZDrive->get__Position(&zl);
	const auto z = zl * ztomicrons_;
	return z;
	NIKONCATCH
		return qQNaN();
}

scope_z_drive::focus_system_status microscope_z_drive_nikon::get_focus_system_internal()
{
	auto status = focus_system_status::off;
	NIKONTRY
		//MIPPARAMLib::IMipParameterPtr pfs_sink = device_handle.pfs_system->IsEnabled;
		//auto is_enabled = long(pfs_sink) == TISCOPELib::StatusTrue;
		const long status_code = device_handle.PFS->GetStatus();
	switch (status_code)
	{
	case 4: status = focus_system_status::moving; break;
	case 5: status = focus_system_status::settled; break;
	default:
		status = focus_system_status::off;
	}
	NIKONCATCH
		return status;
}

scope_limit_z microscope_z_drive_nikon::get_z_drive_limits_internal()
{
	NIKONTRY
		const MIPPARAMLib::IMipParameterPtr zspeedcontrol = device_handle.ZDrive->Position;//wrong units oh well
	const auto zmin = zspeedcontrol->RangeLowerLimit.lVal * ztomicrons_;
	const auto zmax = zspeedcontrol->RangeHigherLimit.lVal * ztomicrons_;
	return scope_limit_z(zmin, zmax);
	NIKONCATCH
		return{};
}

microscope_z_drive_nikon::microscope_z_drive_nikon()
{
	{
		NIKONTRY
			const long res = device_handle.ZDrive->GetResolution();
		const auto unit = device_handle.ZDrive->GetUnit();
		const auto comp = lstrcmpi(L"nm", unit);
		ztomicrons_ = res / 1000.0f;
		NIKONCATCH
	}
	has_focus_system_ = long(device_handle.PFS->IsMounted) == TISCOPELib::StatusTrue;
	common_post_constructor();
}

void microscope_z_drive_nikon::print_settings(std::ostream&) noexcept
{
	//
}

void microscope_xy_drive_nikon::move_to_xy_internal(const scope_location_xy& xy)
{
	NIKONTRY
		const long x_move = xy.x / stomicrons_;
	const long y_move = xy.y / stomicrons_;
	const auto f1 = std::async(std::launch::async, &TISCOPELib::IPositionAccessory::raw_MoveAbsolute, device_handle.XDrive, x_move);
	const auto f2 = std::async(std::launch::async, &TISCOPELib::IPositionAccessory::raw_MoveAbsolute, device_handle.YDrive, y_move);
	f1.wait();
	f2.wait();
	NIKONCATCH
}

scope_location_xy microscope_xy_drive_nikon::get_position_xy_internal()
{
	NIKONTRY
		long xl = 0, yl = 0;
	device_handle.XDrive->get__Position(&xl);
	device_handle.YDrive->get__Position(&yl);
	const auto x = xl * stomicrons_;
	const auto y = yl * stomicrons_;
	return{ x, y };
	NIKONCATCH
		return{};
}

microscope_xy_drive_nikon::microscope_xy_drive_nikon()
{
	{
		NIKONTRY
			const long foo = device_handle.XDrive->GetResolution();
		const auto unit = device_handle.XDrive->GetUnit();
		const auto comp = lstrcmpi(L"nm", unit);
		stomicrons_ = foo / 1000.0f;
		NIKONCATCH
	}
	common_post_constructor();
}

scope_limit_xy microscope_xy_drive_nikon::get_stage_xy_limits_internal()
{

	NIKONTRY
		const MIPPARAMLib::IMipParameterPtr xspeedcontrol = device_handle.XDrive->Position;//wrong units oh well
	const MIPPARAMLib::IMipParameterPtr yspeedcontrol = device_handle.YDrive->Position;//wrong units oh well
	const auto l = xspeedcontrol->RangeLowerLimit.lVal * stomicrons_;//negative
	const auto r = xspeedcontrol->RangeHigherLimit.lVal * stomicrons_;
	const auto b = yspeedcontrol->RangeLowerLimit.lVal * stomicrons_;//negative
	const auto t = yspeedcontrol->RangeHigherLimit.lVal * stomicrons_;
	const auto wid = r - l;
	const auto hei = t - b;
	const auto rect = QRectF(l, b, wid, hei);
	return scope_limit_xy(rect);
	NIKONCATCH
		return{};
}

void microscope_xy_drive_nikon::print_settings(std::ostream&)
{
	//
}

microscope_channel_drive_nikon::microscope_channel_drive_nikon()
{
	NIKONTRY
		channel_names.push_back(scope_channel_drive::channel_off_str);
	channel_names.push_back(scope_channel_drive::channel_phase_str);
	auto names = get_rl_channel_names();
	//So, if you do 0 it closes, if you do 1 it goes to "unused channel" and if you do 3 it goes to channel #1 (start from 1)
	channel_names.insert(std::end(channel_names), std::begin(names), std::end(names));
	light_path_names = { "E100","L100","R100","L80" };//todo get in a better way
	current_light_path_.scope_channel = get_channel_internal();
	if (phase_channel_alias == invalid_phase_channel)
	{
		phase_channel_alias = 4;//note that these alias start at 0
	}
	common_post_constructor();
	NIKONCATCH
}

void microscope_channel_drive_nikon::print_settings(std::ostream&) noexcept
{
	//
}

std::vector<std::string> microscope_channel_drive_nikon::get_rl_channel_names()
{
	std::vector<std::string> names;
	NIKONTRY
		const auto blocks = device_handle.FilterBlockCassette1->GetFilterBlocks();
	const auto last_position = blocks->Count;
	for (auto i = 0; i < last_position; i++)
	{
		const auto internal_number = i + 1;
		const LPWSTR fruitful = blocks->Item[internal_number]->Name;
		// ReSharper disable once CppMsExtDoubleUserConversionInCopyInit
		const std::string converted = CW2A(fruitful);
		//std::string name = CT2A(fruitful);
		names.push_back(converted);
	}
	NIKONCATCH
		return names;
}

void microscope_channel_drive_nikon::set_fl_position(const int chan_internal)
{
	const auto enable_transmitted_light = chan_internal == phase_channel_alias;
	toggle_tl(enable_transmitted_light);
	toggle_rl(!enable_transmitted_light);
	device_handle.FilterBlockCassette1->Position = chan_internal + 1;
}

void microscope_channel_drive_nikon::move_to_light_path_internal(const int channel_idx)
{
	NIKONTRY
		device_handle.LightPathDrive->Position = channel_idx + 1;
	NIKONCATCH
}

void microscope_channel_drive_nikon::move_to_channel_internal(const int channel_idx)
{
	switch (channel_idx)
	{
	case 0: toggle_lights(false); break;
	case 1: set_fl_position(phase_channel_alias); break;
	default: set_fl_position(channel_idx - dummy_channels_offset); break;
	}
}

int microscope_channel_drive_nikon::get_light_path_internal()
{
	const int value = device_handle.LightPathDrive->Position;
	return value - 1;//broken
}

int microscope_channel_drive_nikon::get_channel_internal()
{
	//auto channel = device_handle->FL_Wheel->Position;
	//return channel + dummy_channels_offset;
	return current_light_path_.scope_channel;//broken
}

void microscope_channel_drive_nikon::toggle_rl(const bool enable)
{
	const auto& handle = device_handle.intensilight;
	if (handle)
	{
		const auto message = enable ? "fSXC1" : "fSXC2";
		handle->com_send(message);
	}
}


void microscope_channel_drive_nikon::toggle_tl(const bool enable)
{
	NIKONTRY
	{
		// device_handle.DiaLamp->IsControlled = TISCOPELib::EnumStatus::StatusTrue; // why do we have this line here?
		// const auto get_old_value = []
		// {
		// 	// return device_handle.DiaLamp->Value;
		// 	return device_handle.DiaLamp->Get_Value();
		// };
		//
		// static auto old_value = get_old_value();
		// // upper is always 24
		// static auto upper = [&]
		// {
		// 	// return device_handle.DiaLamp->GetUpperLimit();
		// 	return device_handle.DiaLamp->Get_UpperLimit();
		// }();
		// // lower is always 0
		// static auto lower = [&]
		// {
		// 	// return device_handle.DiaLamp->GetLowerLimit();
		// 	return device_handle.DiaLamp->Get_LowerLimit();
		// }();
		// // I suspect during different t in scanning, the old value will become 0
		// // then when the next t starts, upper / 2 can cause saturation
		// static auto not_zero_old_value = (old_value == 0) ? upper / 2 : old_value;
		
		static auto upper = device_handle.DiaLamp->Get_UpperLimit();
		static auto lower = device_handle.DiaLamp->Get_LowerLimit();
		// static auto old_value = device_handle.DiaLamp->Get_Value(); //only executed once
		static auto old_value = 2 * upper / 3;
		// std::cout << "Before check : old value is " << old_value << ";" << std::endl;
		// if (old_value == lower)
		// {
		// 	old_value = upper / 3;
		// }
		// std::cout << "After check : old value is " << old_value << ";" << std::endl;
		
		if (enable)
		{

			device_handle.DiaLamp->IsOn = TISCOPELib::EnumStatus::StatusTrue;
			// old value validity (not low) is already checked before if statement
			device_handle.DiaLamp->Put_Value(old_value); //this one works, it just calls put__Value()
			std::cout << "Turning ON TL with old value = " << old_value << ";" << std::endl;
			// std::cout << "Turning ON TL with lower value = " << lower << ";" << std::endl;
			// std::cout << "Turning ON TL with upper value = " << upper << ";" << std::endl;
			// #if _DEBUG
			// 		{
			// 			const auto what_we_got = get_old_value();
			// 			if (what_we_got != old_value)
			// 			{
			// 				qli_runtime_error("Nikon TL Lamp Mismatch");
			// 			}
			// 		}
			// #endif
				}
		else
		{
			// old_value = device_handle.DiaLamp->Get_Value();
			// // old_value = get_old_value();
			// if (old_value != 0)
			// {
			// 	not_zero_old_value = old_value;
			// }
			device_handle.DiaLamp->IsOn = TISCOPELib::EnumStatus::StatusFalse;
			std::cout << "Turning OFF TL with old value read as " << old_value << ";" << std::endl;
			// std::cout << "Turning OFF TL with lower value = " << lower << ";" << std::endl;
			// std::cout << "Turning OFF TL with upper value = " << upper << ";" << std::endl;
			// const auto status = device_handle.DiaLamp->IsOn;
			// std::cout << "Turning OFF TL with status = " << status.boolVal << std::endl;
		}
		//
	}
	NIKONCATCH
	//this function is supposed to be synchronous but somehow isn't
	const auto one_second = ms_to_chrono(1000);
	windows_sleep(one_second);
}

void microscope_channel_drive_nikon::toggle_lights(const bool enable)
{
	toggle_tl(enable);
	toggle_rl(enable);
}

#endif