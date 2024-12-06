#include "stdafx.h"
#if (BODY_TYPE==BODY_TYPE_NIKON2) || (STAGE_TYPE==STAGE_TYPE_NIKON2) || (BUILD_ALL_DEVICES_TARGETS)
#include "nikon_devices2.h"
#include <iostream>
#include <memory>
#include <boost/core/noncopyable.hpp>
#include <sstream>
#include "atlstr.h"
#include "new_mic_sdk2.h"
#include "qli_runtime_error.h"
//#include "new_mic_sdk2_DedicatedCommand.h"

#pragma comment(lib, "Ti2_Mic_Driver.lib")
#define NIKON_SAFE_CALL(err) nikon_safe_call(err,__FILE__,__LINE__)
std::unordered_map<lx_result, std::string> error_labels = {
{ LX_OK ,"LX_OK"},
{ LX_ERR_UNEXPECTED ,"LX_ERR_UNEXPECTED" },
{ LX_ERR_NOTIMPL ,"LX_ERR_NOTIMPL" },
{ LX_ERR_OUTOFMEMORY ,"LX_ERR_OUTOFMEMORY" },
{ LX_ERR_INVALIDARG ,"LX_ERR_INVALIDARG" },
{ LX_ERR_NOINTERFACE ,"LX_ERR_NOINTERFACE" },
{ LX_ERR_POINTER ,"LX_ERR_POINTER" },
{ LX_ERR_HANDLE ,"LX_ERR_HANDLE" },
{ LX_ERR_ABORT ,"LX_ERR_ABORT" },
{ LX_ERR_FAIL ,"LX_ERR_FAIL" },
{ LX_ERR_ACCESSDENIED ,"LX_ERR_ACCESSDENIED" }
};
inline void nikon_safe_call(const lx_result err, const char* file, const int line)
{
	if (err != LX_OK)
	{
		const auto name = error_labels.at(err);
		std::stringstream error_msg;
		error_msg << err << " @" << line << ":" << file << ": <-" << name;
		const auto error_msg_str = error_msg.str();
		qli_runtime_error(error_msg_str);
	}
}
std::unique_ptr<nikon_devices2> ptr;

struct nikon_devices2 final : private boost::noncopyable
{
	MIC_MetaData meta_data;
	nikon_devices2()
	{
		lx_uint32 ui_device_count = 0;
		lx_int32* ppi_device_type_list = nullptr;
		NIKON_SAFE_CALL(MIC_GetDeviceList(ui_device_count, &ppi_device_type_list));
		lx_uint64 ui_connected_accessory_mask = 0;
		const lx_uint32 ui_err_msg_max_size = 255;
		lx_wchar  pwsz_err_msg[256] = { 0 };
		if (ui_device_count == 0) {
			// Connect to Simulator.
			NIKON_SAFE_CALL(MIC_SimulatorOpen(0, ui_connected_accessory_mask, ui_err_msg_max_size, pwsz_err_msg));
		}
		else {
			NIKON_SAFE_CALL(MIC_Open(ui_device_count - 1, ui_connected_accessory_mask, ui_err_msg_max_size, pwsz_err_msg));
			//throw here?
		}
		meta_data.uiMetaDataUsageMask = MIC_METADATA_MASK_FULL;
		NIKON_SAFE_CALL(MIC_MetadataGet(meta_data));

	}
	~nikon_devices2()
	{
		NIKON_SAFE_CALL(MIC_Close());
	}
};

void microscope_z_drive_nikon2::move_to_z_internal(const float z)
{
	MIC_Data s_in_data, s_out_data;
	const auto mask = MIC_DATA_MASK_ZPOSITION;
	s_in_data.uiDataUsageMask = mask;
	MIC_Convert_Phys2Dev(mask, z, s_in_data.iZPOSITION);
	s_in_data.iZPOSITIONSpeed = 1;
	s_in_data.iZPOSITIONTolerance = 9;
	NIKON_SAFE_CALL(MIC_DataSet(s_in_data, s_out_data, false));
}

float microscope_z_drive_nikon2::get_position_z_internal()
{
	MIC_Data sdata;
	const auto mask = MIC_DATA_MASK_ZPOSITION;
	sdata.uiDataUsageMask = mask;
	NIKON_SAFE_CALL(MIC_DataGet(sdata));
	double phys_value;
	NIKON_SAFE_CALL(MIC_Convert_Dev2Phys(mask, sdata.iZPOSITION, phys_value));
	return phys_value;
}

scope_limit_z microscope_z_drive_nikon2::get_z_drive_limits_internal()
{
	const auto phys_values_z = ptr->meta_data.iZDrive_RangePhys;
	double min_z, max_z;
	NIKON_SAFE_CALL(MIC_Convert_Dev2Phys(MIC_DATA_MASK_ZPOSITION, phys_values_z[0], min_z));
	NIKON_SAFE_CALL(MIC_Convert_Dev2Phys(MIC_DATA_MASK_ZPOSITION, phys_values_z[1], max_z));
	return scope_limit_z(min_z, max_z);
}

scope_z_drive::focus_system_status  microscope_z_drive_nikon2::get_focus_system_internal()
{
	return scope_z_drive::focus_system_status::off;
}


microscope_z_drive_nikon2::microscope_z_drive_nikon2()
{
	if (!ptr)
	{
		ptr = std::make_unique<nikon_devices2>();
	}
	common_post_constructor();
}

void microscope_z_drive_nikon2::print_settings(std::ostream&) noexcept
{
	//not enough time to implement
}

void microscope_xy_drive_nikon2::move_to_xy_internal(const scope_location_xy& xy)
{
	MIC_Data s_in_data, s_out_data;
	const auto mask = MIC_DATA_MASK_XPOSITION | MIC_DATA_MASK_YPOSITION;
	s_in_data.uiDataUsageMask = mask;
	NIKON_SAFE_CALL(MIC_Convert_Phys2Dev(MIC_DATA_MASK_XPOSITION, xy.x, s_in_data.iXPOSITION));
	s_in_data.iXPOSITIONSpeed = 1;
	s_in_data.iXPOSITIONTolerance = 9;
	NIKON_SAFE_CALL(MIC_Convert_Phys2Dev(MIC_DATA_MASK_YPOSITION, xy.y, s_in_data.iYPOSITION));
	s_in_data.iYPOSITIONSpeed = 1;
	s_in_data.iYPOSITIONTolerance = 9;
	NIKON_SAFE_CALL(MIC_DataSet(s_in_data, s_out_data, false));//<-Breaks on out of bounds
}
//TEST LIMITS ON Z DRIVE

void microscope_xy_drive_nikon2::print_settings(std::ostream&)
{
	//not implemented
}

scope_location_xy microscope_xy_drive_nikon2::get_position_xy_internal()
{
	const auto mask = MIC_DATA_MASK_XPOSITION | MIC_DATA_MASK_YPOSITION;
	MIC_Data sdata;
	sdata.uiDataUsageMask = mask;
	NIKON_SAFE_CALL(MIC_DataGet(sdata));
	double phys_value_x;
	NIKON_SAFE_CALL(MIC_Convert_Dev2Phys(MIC_DATA_MASK_XPOSITION, sdata.iXPOSITION, phys_value_x));
	double phys_value_y;
	NIKON_SAFE_CALL(MIC_Convert_Dev2Phys(MIC_DATA_MASK_YPOSITION, sdata.iYPOSITION, phys_value_y));
	return scope_location_xy(phys_value_x, phys_value_y);
}

microscope_xy_drive_nikon2::microscope_xy_drive_nikon2()
{
	if (!ptr)
	{
		ptr = std::make_unique<nikon_devices2>();
	}
	common_post_constructor();
}

scope_limit_xy microscope_xy_drive_nikon2::get_stage_xy_limits_internal()
{
	const auto phys_values_x = ptr->meta_data.iXYStage_XRangePhys;
	double min_x, max_x;
	NIKON_SAFE_CALL(MIC_Convert_Dev2Phys(MIC_DATA_MASK_XPOSITION, phys_values_x[0], min_x));
	NIKON_SAFE_CALL(MIC_Convert_Dev2Phys(MIC_DATA_MASK_XPOSITION, phys_values_x[1], max_x));
	//
	const auto phys_values_y = ptr->meta_data.iXYStage_YRangePhys;
	double min_y, max_y;
	NIKON_SAFE_CALL(MIC_Convert_Dev2Phys(MIC_DATA_MASK_YPOSITION, phys_values_y[0], min_y));
	NIKON_SAFE_CALL(MIC_Convert_Dev2Phys(MIC_DATA_MASK_YPOSITION, phys_values_y[1], max_y));
	//(qreal left, qreal top, qreal width, qreal height)
	const QPointF top_left(min_x, min_y);
	const QPointF bottom_right(max_x, max_y);
	QRectF rect(top_left, bottom_right);
	const auto bounds = scope_limit_xy(rect.normalized());// screw it
	return bounds;
}

microscope_channel_drive_nikon2::microscope_channel_drive_nikon2()
{
	if (!ptr)
	{
		ptr = std::make_unique<nikon_devices2>();
	}
	channel_names.push_back(scope_channel_drive::channel_off_str);
	channel_names.push_back(scope_channel_drive::channel_phase_str);
	for (auto&& channel : ptr->meta_data.sTurret1Filter)
	{
		// ReSharper disable once CppMsExtDoubleUserConversionInCopyInit
		const std::string converted = CW2A(reinterpret_cast<wchar_t*>(channel.wsShortName));
		//std::cout << converted << std::endl;
		channel_names.push_back(converted);
	}
	common_post_constructor();
}

void microscope_channel_drive_nikon2::set_fl_position(const int channel_idx)
{
	MIC_Data s_in_data, s_out_data;
	s_in_data.uiDataUsageMask = MIC_ACCESSORY_MASK_TURRET1;
	s_in_data.iTURRET1POS = channel_idx;
	NIKON_SAFE_CALL(MIC_DataSet(s_in_data, s_out_data, false));//<-Breaks on out of bounds
}

void microscope_channel_drive_nikon2::move_to_channel_internal(const int channel_idx)
{
	switch (channel_idx)
	{
	case 0: toggle_lights(false); break;
	case 1: set_fl_position(phase_channel_alias); break;
	default: set_fl_position(channel_idx - 2); break;
	}
}

int microscope_channel_drive_nikon2::get_channel_internal()
{
	return current_light_path_.scope_channel;//broken
}

void  microscope_channel_drive_nikon2::toggle_lights(bool)
{
	//not implemented

}

void microscope_channel_drive_nikon2::print_settings(std::ostream&) noexcept
{
	//not implemented
}

#endif
