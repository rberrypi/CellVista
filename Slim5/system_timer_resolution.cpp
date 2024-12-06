#include "stdafx.h"
#include "internal_windows_timer_helpers.h"
#include <bcrypt.h>

#include "qli_runtime_error.h"
#pragma comment(lib, "ntdll.lib")
#define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)

system_timer_resolution get_system_resolution()
{
	static auto nt_query_timer_resolution = reinterpret_cast<NTSTATUS(__stdcall*)(PULONG, PULONG, PULONG)>(GetProcAddress(GetModuleHandle(L"ntdll.dll"), "NtQueryTimerResolution"));
	ULONG minimum_resolution, maximum_resolution, actual_resolution;
	const auto result = nt_query_timer_resolution(&minimum_resolution, &maximum_resolution, &actual_resolution);
	if (!NT_SUCCESS(result))
	{
		qli_runtime_error("Some kind of error");
	}
	const std::chrono::nanoseconds min_resolution(minimum_resolution * 100);
	const std::chrono::nanoseconds max_resolution(maximum_resolution * 100);
	const std::chrono::nanoseconds act_resolution(actual_resolution * 100);
	const system_timer_resolution resolution = { min_resolution,max_resolution,act_resolution };
	return resolution;
}

void set_actual_resolution(const ULONG hundred_nano_second_increments, const system_timer_resolution&)
{
	static auto nt_set_timer_resolution = reinterpret_cast<NTSTATUS(__stdcall*)(ULONG, BOOLEAN, PULONG)>(GetProcAddress(GetModuleHandle(L"ntdll.dll"), "NtSetTimerResolution"));
	const BOOLEAN set = TRUE;
	ULONG current_resolution;
	const auto result = nt_set_timer_resolution(hundred_nano_second_increments, set, &current_resolution);
	if (!NT_SUCCESS(result))
	{
		qli_runtime_error("Some kind of error");
	}
}