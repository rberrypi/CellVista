#include "stdafx.h"
#include "camera_device.h"

const camera_chroma_setting::camera_chroma_settings_map camera_chroma_setting::settings =
{
	{camera_chroma::monochrome,{"Monochrome",demosaic_mode::no_processing}},
	{camera_chroma::forced_color,{"ForcedColor",demosaic_mode::no_processing}},
	{camera_chroma::optional_color,{"OptionalColor",demosaic_mode::rggb_14_native}},
	{camera_chroma::optional_polarization,{"OptionalPolarizer",demosaic_mode::polarization_0_45_90_135}}
};