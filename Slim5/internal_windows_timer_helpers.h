#pragma once
#ifndef INTERNAL_WINDOWS_TIMER_HELPERS_H
#define INTERNAL_WINDOWS_TIMER_HELPERS_H
#include <chrono>
struct system_timer_resolution final
{
	std::chrono::nanoseconds min, max, actual;
	system_timer_resolution(const std::chrono::nanoseconds& min, const std::chrono::nanoseconds& max, const std::chrono::nanoseconds& actual) noexcept: min(min), max(max), actual(actual) {}
};

//wrapper for call NtQueryTimerResolution
[[nodiscard]] system_timer_resolution get_system_resolution();

//NtSetTimerResolution
void set_actual_resolution(ULONG hundred_nano_second_increments, const system_timer_resolution& current_values = get_system_resolution());


#endif
