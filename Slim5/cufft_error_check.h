#pragma once
#ifndef CUFFT_Error_Check_H
#define CUFFT_ERROR_CHECK_H
#include <cufft.h>
#define CUFFT_SAFE_CALL( err ) cufft_safe_call( err, __FILE__, __LINE__ )
void cufft_safe_call(cufftResult_t err, const char* file, int line);
#endif