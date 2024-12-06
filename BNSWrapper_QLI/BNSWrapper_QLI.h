#pragma once
#ifndef BNSWRAPPER_QLI_H
#define BNSWRAPPER_QLI_H
// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the BNSWRAPPER_QLI_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// BNSWRAPPER_QLI_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef BNSWRAPPER_QLI_EXPORTS
#define BNSWRAPPER_QLI_API __declspec(dllexport)
#else
#define BNSWRAPPER_QLI_API __declspec(dllimport)
#endif

// This class is exported from the BNSWrapper_QLI.dll
class BNSWRAPPER_QLI_API CBNSWrapper_QLI {
	void* handle;// oh god, what I have done
public:
	CBNSWrapper_QLI();
	~CBNSWrapper_QLI();
	void set_pattern( unsigned char* in) const;
};

#endif