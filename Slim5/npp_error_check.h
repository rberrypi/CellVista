#pragma once
#ifndef NPP_ERROR_CHECK_H
#define NPP_ERROR_CHECK_H
#include <npp.h>
#define NPP_SAFE_CALL( err ) npp_safe_call(err, __FILE__, __LINE__ )
void npp_safe_call(NppStatus err, const char* file, int line);

#endif