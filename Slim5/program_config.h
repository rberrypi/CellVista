#pragma once
#ifndef PROGRAM_CONFIG_H
#define PROGRAM_CONFIG_H

#define INCLUDE_ML												0

#define HIDE_AUTOMATED_SCANNING									0   //1 = Hide it
#define BUILD_ALL_DEVICES_TARGETS								0

#define SLM_NOT_PRESENT											0
#define SLM_PRESENT												1
#define SLM_PRESENT_VIRTUAL										SLM_NOT_PRESENT
#define SLM_PRESENT_BNS											SLM_NOT_PRESENT
#define SLM_PRESENT_BNS_ANCIENT									SLM_NOT_PRESENT
#define SLM_PRESENT_THORLABSCOM									SLM_NOT_PRESENT
#define SLM_PRESENT_MONITOR										SLM_NOT_PRESENT
#define SLM_PRESENT_ARDUINOCOM									SLM_NOT_PRESENT
#define SLM_PRESENT_MEADOWLARK_RETARDER							SLM_NOT_PRESENT
#define SLM_PRESENT_MEADOWLARK_HS_RETARDER						SLM_NOT_PRESENT
#define SLM_PRESENT_THORLABS_EOM								SLM_NOT_PRESENT

#define BODY_TYPE_VIRTUAL										1
#define BODY_TYPE_ZEISS											2
#define BODY_TYPE_NIKON											3
#define BODY_TYPE_LEICA											4
#define BODY_TYPE_ASI											5
#define BODY_TYPE_NIKON2										6
#define BODY_TYPE_PI_Z											7
#define BODY_TYPE_OLYMPUS										8

#define STAGE_TYPE_VIRTUAL										1
#define STAGE_TYPE_ZEISS										2
#define STAGE_TYPE_NIKON										3
#define STAGE_TYPE_LEICA										4
#define STAGE_TYPE_ASI											5
#define STAGE_TYPE_NIKON2										6
#define STAGE_TYPE_OLYMPUS										7

// List of cameras that are supported
#define CAMERA_NOT_PRESENT										0
#define CAMERA_PRESENT											1
#define CAMERA_PRESENT_VIRTUAL_PSI								CAMERA_NOT_PRESENT
#define CAMERA_PRESENT_ANDOR									CAMERA_NOT_PRESENT
#define CAMERA_PRESENT_HAMAMATSU								CAMERA_NOT_PRESENT
#define CAMERA_PRESENT_ZEISSMR									CAMERA_NOT_PRESENT
#define CAMERA_PRESENT_FLYCAPTURE								CAMERA_NOT_PRESENT
#define CAMERA_PRESENT_SPINRAKER								CAMERA_NOT_PRESENT
#define CAMERA_PRESENT_BSI										CAMERA_NOT_PRESENT
#define CAMERA_PRESENT_PCO_PANDA								CAMERA_NOT_PRESENT
#define CAMERA_PRESENT_PCO_EDGE									CAMERA_NOT_PRESENT

#define KILL_CONDENSER_NAC_CONTROL								1		// 1 = disabled
#define KILL_PORT_SWITCHER										0		// 1 = disabled
#define HIDE_LIGHT_PATH_WIDGET									(~(KILL_CONDENSER_NAC_CONTROL &&  KILL_PORT_SWITCHER))

#define REMOVE_SLIM											1
#define REMOVE_HRSLIM										1
#define REMOVE_GLIM											0
#define REMOVE_IGLIM										0
#define REMOVE_FPM											1
#define REMOVE_CUSTOM										0
#define REMOVE_POL_PSI										1
#define REMOVE_POL_DPM										1
#define REMOVE_DPM											1
#define REMOVE_POL											1


#if 0 //virtual virtual (Debug the UI only)
#define SLM_PRESENT_BNS											SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_VIRTUAL
#define STAGE_TYPE												STAGE_TYPE_VIRTUAL
#define CAMERA_PRESENT_PCO_PANDA								CAMERA_PRESENT
#endif

#if 1 // Fucking GLIM
#define SLM_PRESENT_VIRTUAL										SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_ZEISS
#define STAGE_TYPE												STAGE_TYPE_ZEISS
#define CAMERA_PRESENT_VIRTUAL_PSI								CAMERA_PRESENT
#endif

#if 0 // fuck lilboi and pco
#define SLM_PRESENT_BNS											SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_ZEISS
#define STAGE_TYPE												STAGE_TYPE_ZEISS
#define CAMERA_PRESENT_PCO_PANDA									CAMERA_PRESENT
#endif

#if 0 // zeiss camera test 
#define SLM_PRESENT_VIRTUAL										SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_VIRTUAL
#define STAGE_TYPE												BODY_TYPE_VIRTUAL
#define CAMERA_PRESENT_HAMAMATSU								CAMERA_PRESENT
#endif

#if	0 //virtual virtual virtual virtual virtual (Darkfield)
#define SLM_PRESENT_VIRTUAL										SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_VIRTUAL
#define STAGE_TYPE												STAGE_TYPE_VIRTUAL
#define CAMERA_PRESENT_VIRTUAL_PSI								CAMERA_PRESENT
#define EXTRA_VIRTUAL_CAMERA
#define EXTRA_VIRTUAL_SLM
#endif

#if	0 //virtual virtual virtual virtual virtual 
#define SLM_PRESENT_MONITOR										SLM_PRESENT
//#define SLM_PRESENT_THORLABSCOM									SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_VIRTUAL
#define STAGE_TYPE												STAGE_TYPE_VIRTUAL
#define CAMERA_PRESENT_ANDOR									CAMERA_PRESENT
#define EXTRA_VIRTUAL_CAMERA
//#define EXTRA_VIRTUAL_SLM
#endif

#if	0 //Thorlabs DVI virtual virtual Andor (Darkfield)
#define SLM_PRESENT_MONITOR										SLM_PRESENT
#define SLM_PRESENT_THORLABSCOM									SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_VIRTUAL
#define STAGE_TYPE												STAGE_TYPE_VIRTUAL
#define CAMERA_PRESENT_ANDOR									CAMERA_PRESENT
#endif

#if 0 //BNS NIKON NIKON ANDOR (BI 3410 Zeiss)
#define SLM_PRESENT_THORLABSCOM									SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_ZEISS
#define STAGE_TYPE												STAGE_TYPE_ZEISS
#define CAMERA_PRESENT_HAMAMATSU								CAMERA_PRESENT
#endif

#if 0 //BNS Virtual Virtual Andor (BI 3436, ReflectionQDIC)
#define SLM_PRESENT_MEADOWLARK_RETARDER							SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_VIRTUAL
#define STAGE_TYPE												STAGE_TYPE_VIRTUAL
#define CAMERA_PRESENT_ANDOR									CAMERA_PRESENT
#endif

#if 0 //BNS NIKON NIKON ANDOR (BI 3438 Nikon)
#define SLM_PRESENT_BNS											SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_NIKON
#define STAGE_TYPE												STAGE_TYPE_NIKON
#define CAMERA_PRESENT_ANDOR									CAMERA_PRESENT
#define CAMERA_PRESENT_ZEISSMR									CAMERA_PRESENT
#endif

#if 0 //BNS NIKON NIKON HAMAMATSU (BI 3438 Nikon)
#define SLM_PRESENT_BNS											SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_NIKON
#define STAGE_TYPE												STAGE_TYPE_NIKON
#define CAMERA_PRESENT_ANDOR									CAMERA_PRESENT
#define CAMERA_PRESENT_ZEISSMR									CAMERA_PRESENT
#endif

#if 0 //BNS Virtual Virtual Andor (BI 3436, DPM & SLIM)
#define SLM_PRESENT_BNS											SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_ZEISS
#define STAGE_TYPE												STAGE_TYPE_ZEISS
#define CAMERA_PRESENT_HAMAMATSU								CAMERA_PRESENT
#define CAMERA_PRESENT_ANDOR									CAMERA_PRESENT
#endif

#if 0 //BNS Virtual Virtual Andor (BI 3436, SLIM only)
#define SLM_PRESENT_BNS											SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_ZEISS
#define STAGE_TYPE												STAGE_TYPE_ZEISS
#define CAMERA_PRESENT_HAMAMATSU								CAMERA_PRESENT
#endif

#if 0 //BNS Virtual Virtual Andor (BI 3410, LCVR Speed Test)
#define SLM_PRESENT_THORLABSCOM									SLM_PRESENT
#define BODY_TYPE												BODY_TYPE_ZEISS
#define STAGE_TYPE												STAGE_TYPE_ZEISS
#define CAMERA_PRESENT_HAMAMATSU								CAMERA_PRESENT
#endif

#endif


#ifndef LOGGER_INFO
#define LOGGER_INFO(x) std::cout << "[INFO]["<< std::this_thread::get_id <<"]: " << __FILE__ << ":" << __LINE__ << " -> " << x << "\n" 
#define LOGGER_WARN(x) std::cout << "[WARN]["<< std::this_thread::get_id <<"]: " << __FILE__ << ":" << __LINE__ << " -> " << x << "\n"
#define LOGGER_ERR(x) std::cout << "[ERROR]["<< std::this_thread::get_id <<"]: " << __FILE__ << ":" << __LINE__ << " -> " << x << "\n"
#define LOGGER_ENDL std::endl
#endif
