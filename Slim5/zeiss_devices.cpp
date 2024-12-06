#include "stdafx.h"
//few of these need to be outside for linking etc
#if ((BODY_TYPE==BODY_TYPE_ZEISS)|| (STAGE_TYPE==STAGE_TYPE_ZEISS)) || BUILD_ALL_DEVICES_TARGETS
#include <iostream>
#include <memory>
#include "zeiss_devices.h"
#include "qli_runtime_error.h"
#include <atlbase.h>
#include <atlstr.h>
#import "MTBApi.tlb" named_guids //raw_interfaces_only, could put this somehwere else maybe?speeds up compile time
#include <future>
#include <boost/noncopyable.hpp>

const auto um = L"µm";
const auto na = L"NA";
std::vector<std::string> all_the_elements = { "MTBFocus",
"MTBCondenserFocus",
"MTBStage",
"MTBStageAxisX",
"MTBStageAxisY",
"MTBFocusStabilizer",
"MTBFocusStabilizer2",
"MTBFocusStabilizerLiveCellScanner",
"MTBAutoFocus",
"MTBPiezoFocus",
"MTBPiezoFocusCan",
"MTBReflectorChanger",
"MTBReflectorChanger2",
"MTBObjectiveChanger",
"MTBRLVirtualFilterChanger",
"MTBRLFilterChanger1",
"MTBRLFilterChanger2",
"MTBTLVirtualFilterChanger",
"MTBTLFilterChanger1",
"MTBTLFilterChanger2",
"MTBFLAttenuator",
"MTBRLExtFilterChanger",
"MTBExtFilterwheel_UNKNOWN",
"MTBExtFilterwheel_RL_Mic",
"MTBExtFilterwheel_RL_Lamp",
"MTBExtFilterwheel_TL",
"MTBExtFilterwheel_CAM_Left",
"MTBExtFilterwheel_CAM_Right",
"MTBExtFilterwheel_CAM_Top",
"MTBLaserModuleFilterChanger",
"MTBCondenserContrastChanger",
"MTBDICChanger",
"MTBTLDICChanger",
"MTBCondenserFrontLensChanger",
"MTBOptovarChanger",
"MTBTubeCameraMagnificationChanger",
"MTBMagnificationTube",
"MTBCameraAdapterMagnificationChanger",
"MTBRLLampChanger",
"MTBTLLampChanger",
"MTBInfinitySpacePortChanger",
"MTBSideportChanger",
"MTBBaseportChanger",
"MTBRLRearportChanger",
"MTBIntermediateTube",
"MTBObservationModeChanger",
"MTBDoubleCameraAdapterChanger",
"MTB2TVCamerasChanger",
"MTB2TVVisCamChanger",
"MTBTubeShutter",
"MTBRLShutter",
"MTBTLShutter",
"MTBLaserModuleShutter",
"MTBHXP120Shutter",
"MTBExtShutter_UNKNOWN",
"MTBExtShutter_RL",
"MTBExtShutter_RL_LEFT",
"MTBExtShutter_RL_RIGHT",
"MTBExtShutter_TL",
"MTBExtShutter_TL_LEFT",
"MTBExtShutter_TL_RIGHT",
"MTBFLLEDShutter",
"MTBRLTLSwitch",
"MTBTLPortLSMChanger",
"MTBRLPortLSMChanger",
"MTBTubeVisCamChanger",
"MTBRLApertureStopChanger",
"MTBRLFieldStop",
"MTBTLFieldStop",
"MTBRLApertureStop",
"MTBTLApertureStop",
"MTBDICAzimuth",
"MTBDICShift",
"MTBTLDICShift",
"MTBTIRFAngle",
"MTBTLHalogenLamp",
"MTBRLHalogenLamp",
"MTBReoRLLamp",
"MTBReoTLLamp",
"MTBReoFlLED",
"MTBReoFlLED2",
"MTBExtLamp",
"MTBFiberLight1",
"MTBFiberLight2",
"MTBFiberLight3",
"MTBHXPLamp",
"MTBFluoLamp",
"MTBOtherLamp",
"MTBOtherLamp_MTBTLLampPort",
"MTBOtherLamp_MTBTLLampChanger_left",
"MTBOtherLamp_MTBTLLampChanger_right",
"MTBOtherLamp_MTBRLLampPort",
"MTBOtherLamp_MTBRLLampChanger_left",
"MTBOtherLamp_MTBRLLampChanger_right",
"MTBOtherLamp_MTBHXP120Shutter",
"MTBZoom",
"MTBLightZoom",
"MTBCoupledZoom",
"MTBRLApoTome",
"MTBRLApoTomeFocus",
"MTBRLApoTomePhase",
"MTBRLApoTomeGridChanger",
"MTBEyePiece",
"MTBObject",
"MTBMicroscopeManager",
"MTBCamera_***",
"MTBCameraAdapter_MTBSideportChanger_Left",
"MTBCameraAdapter_MTBSideportChanger_Left_Straight",
"MTBCameraAdapter_MTBSideportChanger_Left_Deflected",
"MTBCameraAdapter_MTBSideportChanger_Right",
"MTBCameraAdapter_MTBSideportChanger_Right_Straight",
"MTBCameraAdapter_MTBSideportChanger_Right_Deflected",
"MTBCameraAdapter_MTBTube_Cameraport",
"MTBCameraAdapter_MTBTube_Cameraport_Straight",
"MTBCameraAdapter_MTBTube_Cameraport_Deflected",
"MTBCameraAdapter_MTBTube_Top",
"MTBCameraAdapter_MTBTube_Left",
"MTBCameraAdapter_MTBBaseportChanger_Frontport",
"MTBCameraAdapter_MTBBaseportChanger_Frontport_Straight",
"MTBCameraAdapter_MTBBaseportChanger_Frontport_Deflected",
"MTBCameraAdapter_MTBBaseportChanger_Baseport",
"MTBCameraAdapter_MTBBaseportChanger_Baseport_Straight",
"MTBCameraAdapter_MTBBaseportChanger_Baseport_Deflected",
"MTBCameraAdapter_MTB2TVCamerasChanger_Back",
"MTBCameraAdapter_MTB2TVCamerasChanger_Top",
"MTBCameraAdapter_MTBInfinitySpacePortChanger_Cameraport",
"MTBCameraAdapter_MTBInfinitySpacePortChanger_Cameraport_Straight",
"MTBCameraAdapter_MTBInfinitySpacePortChanger_Cameraport_Deflected",
"MTBCameraAdapter_MTBIntermediateTube_Left",
"MTBCameraAdapter_MTBIntermediateTube_Left_Straight",
"MTBCameraAdapter_MTBIntermediateTube_Left_Deflected",
"MTBCameraAdapter_MTBIntermediateTube_Right",
"MTBCameraAdapter_MTBIntermediateTube_Right_Straight",
"MTBCameraAdapter_MTBIntermediateTube_Right_Deflected",
"MTBCameraAdapter_MTBCSU_Left",
"MTBCameraAdapter_MTBCSU_Top",
"MTBFLLEDController",
"MTBFLLEDCombiner1234",
"MTBFLLEDCombiner12",
"MTBFLLEDCombiner34",
"MTBLED1",
"MTBLED2",
"MTBLED3",
"MTBLED4",
"MTBCSUDiskSpeed",
"MTBCSUShutter",
"MTBCSULeftEmissionFilterChanger",
"MTBCSUTopEmissionFilterChanger",
"MTBCSUDichroicChanger",
"MTBCSUPortSwitch",
"MTBDL450ContrastChanger",
"MTBDL450BestModeOnOffChanger",
"MTBDL450Lamp",
"MTBPulseGenerator",
"MTBBatchCommander",
"MTBRLSutterLambdaDGController",
"MTBRLSutterLambdaDGExcitationChanger",
"MTBRLSutterLambdaDGShutter",
"MTBRLSutterLambdaDGTransmissionChanger",
"MTBFLLEDFilterChanger" };
using namespace MTBApi;
//All in one for compile time improvement
#define MTB_ERROR(err) mtb_display_error(err,__FILE__,__LINE__)
void mtb_display_error(const _com_error& e, const char* file, const int line)
{
	CString message;
	if (e.Error() >= 0x80041000)
	{
		BSTR msg;
		auto info = e.ErrorInfo();
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
		message = e.ErrorMessage();
	}
	const auto filtered = CStringA(message);
	std::cout << "MTB Error: " << filtered << " " << line << ":" << file << " " << std::endl;
	throw e;
}
bool mtb_hardware_test(const std::function<void(void)>& function)
{
	try
	{
		function();
		return true;
	}
	catch (...)
	{
		return false;
	}
}



#define MTBTRY try{
#define MTBCATCH }catch (const _com_error& e){MTB_ERROR(e);}

static std::vector<std::string> get_changer_names(IMTBChangerPtr device_handle)
{
	std::vector<std::string> names;
	MTBTRY
		if (device_handle)
		{
			const auto num = device_handle->GetElementCount();
			for (auto i = 0; i < num; ++i)
			{
				auto element = static_cast<IMTBIdentPtr>(device_handle->GetElement(i));
				const auto name = element->GetName();
				names.push_back(std::string(name));
			}
		}
	MTBCATCH
		if (names.empty())
		{
			names.push_back("");
		}
	return names;
}

class mtb_devices final : boost::noncopyable  // NOLINT(cppcoreguidelines-special-member-functions)
{

	CComBSTR mtb_id_;
	IMTBConnectionPtr mtb_mtb_connection_{};
	IMTBRootPtr mtb_root_{};
	IMTBServoPtr mtb_tl_aperture_stop_{};
	IMTBLampChangerPtr mtb_tl_lamp_changer_{};
	IMTBSideportChangerPtr mtb_sideport_changer_{};
	IMTBShutterPtr mtb_tl_shutter_{};
	IMTBContrastChangerPtr  mtb_condenser_contrast_changer_{};
	IMTBStagePtr mtb_stage_{};
	IMTBFocusPtr mtb_focus_{};
	IMTBShutterPtr mtb_shutter_tl_{};
	IMTBShutterPtr mtb_shutter_rl_{};
	IMTBReflectorChangerPtr mtb_shutter_rl_changer_{};
	IMTBMicroscopeManagerPtr mtb_microscope_manger_{};
	//constructor fills self with devices:
	void fix_lock(const IMTBBasePtr& unlock_me) const
	{
		MTBTRY
			const bool is_locked = static_cast<IMTBBasePtr>(unlock_me)->IsLocked;
		if (is_locked)
		{
			static_cast<IMTBBasePtr>(unlock_me)->Unlock(static_cast<BSTR>(mtb_id_));
		}
		MTBCATCH
	}
	//decltype(MTB_Root->getcompensationonentFullConfig(in))
	auto make_and_check(_bstr_t in) const -> decltype(mtb_root_->GetComponentFullConfig(in))
	{
		auto answer = mtb_root_->GetComponentFullConfig(in);
		if (!answer)
		{
			std::cout << "Zeiss: could not get " << in << std::endl;
			return nullptr;
		}
		MTBApi::MTBAccessibility accessibility;
		answer->get_Accessibility(&accessibility);
		std::cout << "Zeiss Accessibility " << accessibility << std::endl;
		const auto motorization = answer->GetMotorization();
		std::cout << "Zeiss Motorization " << motorization << std::endl;
		// ReSharper disable CppMsExtBindingRValueToLvalueReference
		fix_lock(static_cast<IMTBBasePtr>(answer));

		// ReSharper restore CppMsExtBindingRValueToLvalueReference
		std::cout << "Zeiss Made " << in << std::endl;
		return answer;
	}
public:
	const MTBCmdSetModes mtb_move_mode_acq{}, mtb_move_mode_live{}, mtb_move_mode_acq_z{};
	IMTBChangerPtr side_port_changer() const
	{
		return mtb_sideport_changer_ ? static_cast<IMTBChangerPtr>(mtb_sideport_changer_) : nullptr;
	}
	IMTBChangerPtr lamp_changer() const
	{
		return mtb_tl_lamp_changer_ ? static_cast<IMTBChangerPtr>(mtb_tl_lamp_changer_) : nullptr;
	}
	IMTBContinualPtr tl_condenser() const
	{
		return mtb_tl_aperture_stop_ ? static_cast<IMTBServoPtr>(mtb_tl_aperture_stop_) : nullptr;
	}
	IMTBAxisPtr x_axis_axis() const
	{
		return mtb_stage_ ? static_cast<IMTBAxisPtr>(mtb_stage_->XAxis) : nullptr;
	}
	IMTBAxisPtr y_axis_axis() const
	{
		return mtb_stage_ ? static_cast<IMTBAxisPtr>(mtb_stage_->YAxis) : nullptr;
	}
	IMTBAxisPtr z_axis_axis() const
	{
		return static_cast<IMTBAxisPtr>(mtb_focus_);
	}
	IMTBContinualSpeedPtr x_axis_speed() const
	{
		return mtb_stage_ ? static_cast<IMTBContinualSpeedPtr>(mtb_stage_->XAxis) : nullptr;
	}
	IMTBContinualSpeedPtr y_axis_speed() const
	{
		return mtb_stage_ ? static_cast<IMTBContinualSpeedPtr>(mtb_stage_->YAxis) : nullptr;
	}
	IMTBContinualSpeedPtr z_axis_speed() const
	{
		return static_cast<IMTBContinualSpeedPtr>(mtb_focus_);
	}
	IMTBContinualPtr z_axis_continual() const
	{
		return static_cast<IMTBContinualPtr>(mtb_focus_);
	}
	IMTBChangerPtr rl_shutter() const
	{
		return static_cast<IMTBChangerPtr>(mtb_shutter_rl_);
	}
	IMTBChangerPtr tl_shutter() const
	{
		return static_cast<IMTBChangerPtr>(mtb_shutter_tl_);
	}
	IMTBChangerPtr shutter_rl_changer() const
	{
		return static_cast<IMTBChangerPtr>(mtb_shutter_rl_changer_);
	}
	IMTBChangerPtr condenser_changer() const
	{
		return static_cast<IMTBChangerPtr>(mtb_condenser_contrast_changer_);
	}
	IMTBStagePtr stage() const
	{
		return mtb_stage_;
	}
	IMTBRootPtr root() const
	{
		return mtb_root_;
	}
	IMTBMicroscopeManagerPtr manager() const
	{
		return mtb_microscope_manger_;
	}
	mtb_devices() :
		mtb_move_mode_acq(static_cast<MTBCmdSetModes>(MTBCmdSetModes_Synchronous | MTBCmdSetModes_BidirectionalBacklashSmart | MTBCmdSetModes_Smooth)),
		mtb_move_mode_live(MTBCmdSetModes_Synchronous),
		mtb_move_mode_acq_z(MTBCmdSetModes_Synchronous)
	{
		{
			const auto ret = CoInitializeEx(nullptr, COINIT_SPEED_OVER_MEMORY | COINIT_MULTITHREADED);
			std::cout << "COM Status " << (ret == S_OK ? "Okay" : "Maybe Okay") << std::endl;
			//assert(a == S_OK);//oh well//
		}
		MTBTRY
			mtb_mtb_connection_ = IMTBConnectionPtr(CLSID_MTBConnection);
		mtb_mtb_connection_->Login("en", &mtb_id_);
		mtb_root_ = static_cast<IUnknown*>(mtb_mtb_connection_->GetRoot(static_cast<BSTR>(mtb_id_)));
#if _DEBUG
		{
			for (const auto element : all_the_elements)
			{
				const _bstr_t temp(element.c_str());
				const auto answer = mtb_root_->GetComponentFullConfig(temp);
				if (!answer)
				{
					continue;
				}
				std::cout << "Detected " << element << std::endl;
			}
		}
#endif
		// ReSharper disable once CppMsExtBindingRValueToLvalueReference
		fix_lock(static_cast<IMTBBasePtr>(mtb_root_));//ie we have exclusive use of the microscope?
		MTBCATCH
			//Focus
			MTBTRY
			mtb_focus_ = make_and_check("MTBFocus");
		mtb_hardware_test([&] {Q_UNUSED(z_axis_continual()->GetPosition(um)); });
		MTBCATCH
			MTBTRY
			mtb_stage_ = make_and_check("MTBStage");
		MTBCATCH
			MTBTRY
			mtb_shutter_tl_ = make_and_check("MTBTLShutter");
		MTBCATCH
			MTBTRY
			mtb_shutter_rl_ = make_and_check("MTBRLShutter");
		MTBCATCH
			MTBTRY
			mtb_shutter_rl_changer_ = make_and_check("MTBReflectorChanger");
		const auto test = [&] {Q_UNUSED(shutter_rl_changer()->GetPosition()); };
		mtb_shutter_rl_changer_ = mtb_hardware_test(test) ? mtb_shutter_rl_changer_ : nullptr;
		MTBCATCH
			MTBTRY
			mtb_microscope_manger_ = make_and_check("MTBMicroscopeManager");
		MTBCATCH
			MTBTRY
			mtb_tl_aperture_stop_ = make_and_check("MTBTLApertureStop");
		const auto test = [&] {Q_UNUSED(tl_condenser()->GetPosition(na)); };
		mtb_tl_aperture_stop_ = mtb_hardware_test(test) ? mtb_tl_aperture_stop_ : nullptr;
		MTBCATCH
			MTBTRY
			mtb_sideport_changer_ = make_and_check("MTBSideportChanger");
		MTBCATCH
			MTBTRY
			mtb_tl_lamp_changer_ = make_and_check("MTBTLLampChanger");
		MTBCATCH
			MTBTRY
			mtb_condenser_contrast_changer_ = make_and_check("MTBCondenserContrastChanger");
		MTBCATCH
			std::cout << "Done Setting Up" << std::endl;
	}
	~mtb_devices()
	{
		//fuck these need to be custom destructors
		//note these gotta be in the reverse order of the other devices
		if (mtb_shutter_rl_changer_) mtb_shutter_rl_changer_.Release();
		if (mtb_microscope_manger_) mtb_microscope_manger_.Release();
		if (mtb_shutter_rl_) mtb_shutter_rl_.Release();
		if (mtb_shutter_tl_) mtb_shutter_tl_.Release();
		if (mtb_focus_) mtb_focus_.Release();
		if (mtb_stage_) mtb_stage_.Release();
		if (mtb_condenser_contrast_changer_) mtb_condenser_contrast_changer_.Release();
		if (mtb_tl_shutter_) mtb_tl_shutter_.Release();
		if (mtb_sideport_changer_) mtb_sideport_changer_.Release();
		if (mtb_tl_lamp_changer_) mtb_tl_lamp_changer_.Release();
		if (mtb_tl_aperture_stop_) mtb_tl_aperture_stop_.Release();
		if (mtb_root_) mtb_root_.Release();
		//
		mtb_mtb_connection_->Logout(static_cast<BSTR>(mtb_id_));
		CoUninitialize();//this might be un_needed and bad, I'm not 100% sure what happens when you call this multiple times. actually I'm pretty sure what happens, and it ain't good..
	}
};

std::unique_ptr<mtb_devices> device_handle;
//

void scope_z_drive_zeiss::move_to_z_internal(const float z)
{
	MTBTRY
		device_handle->z_axis_continual()->SetPosition_2(z, um, device_handle->mtb_move_mode_acq_z);
	MTBCATCH
}

float scope_z_drive_zeiss::get_position_z_internal()
{
	MTBTRY
		return 	device_handle->z_axis_continual()->GetPosition(um);
	MTBCATCH
		return qQNaN();
}

scope_limit_z scope_z_drive_zeiss::get_z_drive_limits_internal()
{
	const auto zmin = device_handle->z_axis_axis() ? device_handle->z_axis_axis()->GetSWLimit(false, um) : -1;
	const auto zmax = device_handle->z_axis_axis() ? device_handle->z_axis_axis()->GetSWLimit(true, um) : 1;
	return scope_limit_z(zmin, zmax);
}

scope_z_drive_zeiss::scope_z_drive_zeiss()
{
	if (device_handle == nullptr)
	{
		device_handle.reset(new mtb_devices);
	}
	common_post_constructor();
}

void scope_z_drive_zeiss::print_settings(std::ostream&) noexcept
{
	//
}

void scope_xy_drive_zeiss::move_to_xy_internal(const scope_location_xy& xy)
{
	const auto loc_x = static_cast<double>(xy.x);
	const auto loc_y = static_cast<double>(xy.y);
	MTBTRY
		device_handle->stage()->SetPosition_4("MTBStage", loc_x, loc_y, um, device_handle->mtb_move_mode_acq);
	MTBCATCH
}

scope_location_xy scope_xy_drive_zeiss::get_position_xy_internal()
{
	MTBTRY
		double x, y;
	device_handle->stage()->GetPosition(&x, &y, um);
	return{ static_cast<float>(x), static_cast<float>(y) };
	MTBCATCH
		return{};
}

scope_limit_xy scope_xy_drive_zeiss::get_stage_xy_limits_internal()
{
	MTBTRY
		const auto l = device_handle->x_axis_axis()->GetSWLimit(false, um);
	const auto r = device_handle->x_axis_axis()->GetSWLimit(true, um);
	const auto  b = device_handle->y_axis_axis()->GetSWLimit(false, um);
	const auto  t = device_handle->y_axis_axis()->GetSWLimit(true, um);
	const auto  wid = r - l;
	const auto  hei = t - b;
	const auto rect = QRectF(l, b, wid, hei);
	static const auto s = 350000;// get this from looking at MTB test, for the old MAC 5000 stages?
	const auto alternative = QRectF(-s, -s, s * 2, s * 2);
	const auto valid_rect = !rect.isEmpty();
	return scope_limit_xy(valid_rect ? rect : alternative);
	MTBCATCH
		return{};
}

void scope_xy_drive_zeiss::print_settings(std::ostream&)
{
	//
}

scope_xy_drive_zeiss::scope_xy_drive_zeiss()
{
	if (device_handle == nullptr)
	{
		device_handle.reset(new mtb_devices);
	}
	common_post_constructor();
}

void scope_channel_drive_zeiss::toggle_rl(const bool enable)
{
	MTBTRY
		auto handle = device_handle->rl_shutter();
	if (handle)
	{
		const auto pos = enable ? changer_on : changer_off;
		const auto mode = MTBCmdSetModes_Synchronous;
		handle->SetPosition_2(pos, mode);
	}
	MTBCATCH
}

void scope_channel_drive_zeiss::toggle_tl(const bool enable)
{
	MTBTRY
		auto handle = device_handle->tl_shutter();
	if (handle)
	{
		const auto pos = enable ? changer_on : changer_off;
		const auto mode = MTBCmdSetModes_Synchronous;
		handle->SetPosition_2(pos, mode);
	}
	MTBCATCH
}

void scope_channel_drive_zeiss::toggle_lights(bool enable)
{
	const auto tl = std::async(std::launch::async, &scope_channel_drive_zeiss::toggle_tl, enable);
	const auto rl = std::async(std::launch::async, &scope_channel_drive_zeiss::toggle_rl, enable);
	tl.wait();
	rl.wait();
}

scope_channel_drive_zeiss::scope_channel_drive_zeiss()
{
	if (device_handle == nullptr)
	{
		device_handle.reset(new mtb_devices);
	}
	channel_names.push_back(channel_off_str);
	channel_names.push_back(channel_phase_str);
	//So, if you do 0 it closes, if you do 1 it goes to "unused channel" and if you do 3 it goes to channel #1 (start from 1)
	{
		const auto ptr = device_handle->shutter_rl_changer();
		auto rl_channels = get_changer_names(ptr);
		channel_names.insert(std::end(channel_names), std::begin(rl_channels), std::end(rl_channels));
	}
	condenser_names = get_changer_names(device_handle->condenser_changer());
	{
		const auto side_ports = get_changer_names(device_handle->side_port_changer());
		const auto tl_ports = get_changer_names(device_handle->lamp_changer());
		//manual loop unrolling is sad
		for (auto side_port_idx = 0; side_port_idx < side_ports.size(); ++side_port_idx)
		{
			for (auto tl_ports_idx = 0; tl_ports_idx < tl_ports.size(); ++tl_ports_idx)
			{
				const auto tl_name = tl_ports.at(tl_ports_idx);
				auto name = side_ports.at(side_port_idx);
				if (!tl_name.empty())
				{
					name = name + " + " + tl_name;
				}
				const auto zeiss_native_idx = [](auto idx) {return idx + 1; };
				foobar_zeiss_light_path_position position(zeiss_native_idx(side_port_idx), zeiss_native_idx(tl_ports_idx), name);
				light_path_index_to_position.push_back(position);
			}
		}
		const auto get_name = [](const foobar_zeiss_light_path_position& positions) {return positions.name; };
		std::transform(light_path_index_to_position.begin(), light_path_index_to_position.end(), std::back_inserter(light_path_names), get_name);
	}
	has_nac = device_handle->tl_condenser() != nullptr;
	has_light_path = device_handle->side_port_changer() != nullptr;
	current_light_path_.scope_channel = get_channel_internal();
	current_light_path_.light_path = get_light_path_internal();
	static_cast<condenser_position&>(current_light_path_) = get_condenser_internal();
	if (phase_channel_alias == invalid_phase_channel)
	{
		phase_channel_alias = 4;//note that these alias start at 0
	}
	common_post_constructor();
}

void scope_channel_drive_zeiss::move_to_light_path_internal(const int light_path_idx)
{
	MTBTRY
		if (light_path_idx >= light_path_index_to_position.size())
		{
			return;
		}
	const auto position = light_path_index_to_position.at(light_path_idx);
	const auto move_if_exists = [](const IMTBChangerPtr& handle, auto native_idx)
	{
		if (handle)
		{
			const auto mode = MTBCmdSetModes_Synchronous;
			handle->SetPosition_2(native_idx, mode);
		}
	};
	move_if_exists(device_handle->side_port_changer(), position.side_port_idx);
	move_if_exists(device_handle->lamp_changer(), position.tl_lamp_switch_idx);
	MTBCATCH
}

_bstr_t zeiss_condenser_units;
condenser_nac_limits scope_channel_drive_zeiss::get_condenser_na_limit_internal()
{
	auto alias = device_handle->tl_condenser();
	if (alias)
	{
		zeiss_condenser_units = alias->GetPositionUnit(0);//rads NA
		const auto max_position = alias->GetMaxPosition(zeiss_condenser_units);
		const auto min_position = alias->GetMinPosition(zeiss_condenser_units);
		return condenser_nac_limits(min_position, max_position, true);
	}
	return condenser_nac_limits();
}

void scope_channel_drive_zeiss::move_to_channel_internal(const int channel_idx)
{
	//So, if you do 0 it closes, if you do 1 it goes to "unused channel" and if you do 3 it goes to channel #1 (start from 1)
	switch (channel_idx)
	{
	case 0: toggle_lights(false); break;
	case 1: set_cube_position(phase_channel_alias, is_transmission); break;
	default: set_cube_position(channel_idx - dummy_channels_offset, false); break;
	}
}

void scope_channel_drive_zeiss::set_cube_position(int chan_internal, bool is_tl) const
{
	MTBTRY
		auto switch_cube = [chan_internal] {
		auto handle = device_handle->shutter_rl_changer();
		const auto mode = MTBCmdSetModes_Synchronous;
		handle->SetPosition_2(chan_internal + 1, mode);
	};


	//So, Zeiss scope's will try to match the contrast manager, so the TL will open switching the filter cube
	//So we switch the cube, and then switch the TL. This is the only way.
	auto cube = std::async(std::launch::async, switch_cube);
	cube.wait();
	auto tl = std::async(std::launch::async, &scope_channel_drive_zeiss::toggle_tl, is_tl);
	auto rl = std::async(std::launch::async, &scope_channel_drive_zeiss::toggle_rl, !is_tl);
	tl.wait();
	rl.wait();

	MTBCATCH
}

int scope_channel_drive_zeiss::get_channel_internal()
{
	MTBTRY
		const auto get_position = [&](const auto& shutter) {
		short value = (-1);
		if (shutter)
		{
			shutter->get_Position(&value);
		}
		return value;
	};
	const auto tl = get_position(device_handle->tl_shutter());
	const auto rl = get_position(device_handle->rl_shutter());
	auto cube = get_position(device_handle->shutter_rl_changer());
	cube = cube - 1;//to our zero convention
	if (tl == changer_off && rl == changer_off)
	{
		return off_channel_idx;
	}
	if (tl == changer_on && cube == phase_channel_alias)
	{
		return phase_channel_idx;
	}
	return cube + dummy_channels_offset;
	MTBCATCH
		return invalid_phase_channel;
}

int scope_channel_drive_zeiss::get_light_path_internal()
{
	MTBTRY
		if (!light_path_index_to_position.empty())
		{
			const auto get_channel_native_position = [](const IMTBChangerPtr& handle)
			{
				short position = 0;
				if (!handle)
				{
					return position;
				}
				handle->get_Position(&position);
				return position;
			};
			//secret O(N)
			const auto get_position = [&]
			{
				const auto side_port_changer = get_channel_native_position(device_handle->side_port_changer());
				const auto lamp_changer = get_channel_native_position(device_handle->lamp_changer());
				if (lamp_changer == 0)
				{
					std::cout << "Zeiss Light Path jammed, not reading correct tl light port. Defaulting to left (LED) until further change." << std::endl;
					return foobar_zeiss_light_path_position(side_port_changer, 1);
				} else
				{
					return foobar_zeiss_light_path_position(side_port_changer, lamp_changer);					
				}
			};
			const auto extra_attempts = 1;
			for (auto i = 0; i < extra_attempts; ++i)
			{
				const auto position = get_position();
				const auto functor = [&](const foobar_zeiss_light_path_position& light_path_position)
				{
					return position.approx_equals(light_path_position);
				};
				const auto element = std::find_if(light_path_index_to_position.begin(), light_path_index_to_position.end(), functor);
				if (element != light_path_index_to_position.end())
				{
					const auto idx = std::distance(light_path_index_to_position.begin(), element);
					return idx;
				}
				windows_sleep(ms_to_chrono(1000 * 10));
			}
			std::cout << "Zeiss light path is jammed! Shouldn't be here" << std::endl;
			auto volatile here = 0;
		}
	MTBCATCH
		return current_light_path_.light_path;
}

void scope_channel_drive_zeiss::move_condenser_internal(const condenser_position& position)
{
	const auto nac_control = device_handle->tl_condenser();
	if (nac_control)
	{
		nac_control->SetPosition_2(position.nac, zeiss_condenser_units, device_handle->mtb_move_mode_acq);
	}
	const auto condenser = device_handle->condenser_changer();
	if (condenser)
	{
		const auto mode = MTBCmdSetModes_Synchronous;
		condenser->SetPosition_2(position.nac_position + 1, mode);
	}
}

condenser_position scope_channel_drive_zeiss::get_condenser_internal()
{
	auto nac_control = device_handle->tl_condenser();
	if (nac_control)
	{
		current_light_path_.nac = nac_control->GetPosition(na);
	}
	const auto condenser = device_handle->condenser_changer();
	{
		short position = 0;
		condenser->get_Position(&position);
		position = position - 1;//zeiss convention? WTF DID I WRITE WHY THE FUCK IS THIS INDIRECTION HANDLED AT THIS LEVEL? 
		current_light_path_.nac_position = position;
	}
	return static_cast<condenser_position&>(current_light_path_);
}

void scope_channel_drive_zeiss::print_settings(std::ostream&) noexcept
{
	//
}

#endif