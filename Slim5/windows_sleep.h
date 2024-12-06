#pragma once
#ifndef WINDOWS_SLEEP_H
#define WINDOWS_SLEEP_H
#include <thread>
template<typename T> void windows_sleep(T time_to_sleep) noexcept
{
	const auto milliseconds_double = to_mili(time_to_sleep);
	if (milliseconds_double > 0)
	{
		// ReSharper disable CppDeprecatedEntity
		// 		const auto milliseconds = static_cast<unsigned long>(std::ceil(milliseconds_double));
		//_sleep(milliseconds);
		// ReSharper restore CppDeprecatedEntity
		//Old sleep behavior changed (old behavior had _sleep actually sleep for the right time), no std::apparently does this
		std::this_thread::sleep_for(time_to_sleep);
	}
}
#endif