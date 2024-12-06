#pragma once
#pragma warning( push )  
#pragma warning(disable : 4996)
#pragma warning(disable : 4005)

//Windows 7 target

#if 0
#define VC_EXTRALEAN        // Exclude rarely-used stuff from Windows headers
#include <afxwin.h>//This will include windows.h because
#include <afxeNOWINSTYLESxt.h> // MFC extensions
#else
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
//#define NOGDICAPMASKS (Need CP, for example CP_THREAD_ACP)
//#define NOVIRTUALKEYCODES (cimg needs this)
//#define NOWINMESSAGES (WM_QUIT)
//#define NOWINSTYLES (cimg)
#define NOMENUS
//#define NOICONS IDI_APPLICATION
#define NOKEYSTATES
#define NOSYSCOMMANDS
#define NORASTEROPS
//#define NOSHOWWINDOW (cimg)
#define NOEMRESOURCE
#define NOATOM
#define NOCLIPBOARD
#define NOCOLOR
//#define NOCTLMGR Also from cimg (sad)
#define NODRAWTEXT
//#define NOGDI
#define NOKERNEL
//#define NOUSER (MessageBox is in user ??)
// #define NONLS (MultiByteToWideChar function)
//#define NOMB (we actually used message box)
#define NOMEMMGR
#define NOMETAFILE
#define NOMINMAX
//#define NOMSG (we actually used message box)
#define NOOPENFILE
#define NOSCROLL
#define NOSERVICE
#define NOSOUND
//#define NOTEXTMETRIC (apparently need this, for example LPTEXTMETRICA)
#define NOCOMM
#define NOKANJI
#define NOHELP
#define NOPROFILER
#define NODEFERWINDOWPOS
#define NOMCX

#include <Windows.h>
#endif
#undef NOMINMAX
//
//
#include "windows_sleep.h"
#include "chrono_converters.h"
#include "program_config.h"

#define IS_NIKON_APARTMENT_BUG (((BODY_TYPE==BODY_TYPE_NIKON) || (STAGE_TYPE==STAGE_TYPE_NIKON)))
#define HAS_LEICA (((BODY_TYPE==BODY_TYPE_LEICA) || (STAGE_TYPE==STAGE_TYPE_LEICA)))

#ifdef _DEBUG 
#define SIMULATE_OVERFLOW (0)
#else
#define SIMULATE_OVERFLOW (0)
#endif


#pragma warning( pop )   