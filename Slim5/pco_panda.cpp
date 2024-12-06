#include "stdafx.h"
#include <stdafx.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

#if CAMERA_PRESENT_PCO_PANDA == CAMERA_PRESENT || BUILD_ALL_DEVICES_TARGETS

#include <pco_panda.h>

pco_panda::pco_panda(int camera_id, QObject* parent) :
	pco(
		camera_device_features( // features associated with the PCO panda family
			false, // panda does not support burst mode. see section 2.12.3.4
			false, // the camera does not seem to have async mode
			false, // the panda family does not have active cooling
			false, // this is not a virtual camera
			camera_contrast_features(
				camera_chroma::monochrome, // This camera can support both color and monochrome
											// currently we are only using monochrome
				demosaic_mode::no_processing, // we do not want to do any pose demosaic processing
				{ 0, 65535 } // the display range is just some arbitrary value. it can be changed in the GUI
			)
		),
		1, // number of buffers to allocate in camera internal memory
		camera_id,
		parent
	)
{
	init();
}


pco_panda::~pco_panda() {}


void pco_panda::initialize_aois() {
	WORD x0, y0, _garbage_;
	PCO_ERR_CHK(PCO_GetROI(cam_handle, &x0, &y0, &_garbage_, &_garbage_)); // refer to section 2.5.6 of the SDK manual

	PCO_ERR_CHK(PCO_GetSizes(cam_handle, &XResAct, &YResAct, &XResMax, &YResMax));

	aois.emplace_back(camera_aoi(2048, 2048, x0, y0));
	aois.emplace_back(camera_aoi(1024, 1024, x0, y0));
	aois.emplace_back(camera_aoi(512, 512, x0, y0));
	aois.emplace_back(camera_aoi(256, 256, x0, y0));

	aois.at(0).re_center_and_fixup(XResMax, YResMax);
}

#endif