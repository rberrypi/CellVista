#pragma once
#ifndef APPORX_EQUALS_H
#define APPORX_EQUALS_H
template<typename T> [[nodiscard]] bool exactly_equals(T p1, T p2) noexcept
{
	if (_isnan(p1) &&_isnan(p2))
	{
		return true;
	}
	return p1 == p2;
}

template<typename T>
[[nodiscard]] bool approx_equals(T p1, T p2) noexcept
{
	if (exactly_equals(p1, p2))
	{
		return true;
	}
	const auto precision = std::pow(static_cast<T>(10), -1);
	const auto equal = std::abs(p1 - p2) < precision;
	return equal;
}
#endif