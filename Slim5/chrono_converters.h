#pragma once
#ifndef CHRONO_CONVERTERS_H
#define CHRONO_CONVERTERS_H

#include <chrono>
#include <iomanip>

template<typename T, typename V> void display_time(T& os, V ns)
{
	auto fill = os.fill();
	os.fill('0');
	const auto d = std::chrono::duration_cast<std::chrono::duration<int, std::ratio<86400>>>(ns);
	ns -= d;
	const auto h = std::chrono::duration_cast<std::chrono::hours>(ns);
	ns -= h;
	const auto m = std::chrono::duration_cast<std::chrono::minutes>(ns);
	ns -= m;
	const auto s = std::chrono::duration_cast<std::chrono::seconds>(ns);
	ns -= s;
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(ns);
	ns -= ms;
	const auto us = std::chrono::duration_cast<std::chrono::microseconds>(ns);
	//ns -= us;
	os << std::setw(2) << d.count() << "d:"
		<< std::setw(2) << h.count() << "h:"
		<< std::setw(2) << m.count() << "m:"
		<< std::setw(2) << s.count() << "s:"
		<< std::setw(3) << ms.count() << "ms:"
		<< std::setw(3) << us.count() << "us";
	os.fill(fill);
}

template<typename T> double to_mili(T chrono_time) noexcept
{
	return std::chrono::duration_cast<std::chrono::duration<double, std::milli >> (chrono_time).count();
}

inline constexpr std::chrono::microseconds ms_to_chrono(const double time_in_ms) noexcept
{
	constexpr auto ms = 1000;
	const auto micro_seconds = static_cast<long long>(time_in_ms * ms);
	return std::chrono::microseconds(micro_seconds);
}

__forceinline constexpr std::chrono::microseconds ns_to_us_chrono(const double time_in_ns) noexcept {
	return std::chrono::microseconds(int64_t(time_in_ns * 0.001));
}

std::chrono::microseconds timestamp() noexcept;

#endif