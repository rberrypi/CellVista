#ifndef ML_SHARED_H
#define ML_SHARED_H

[[nodiscard]] inline bool is_divisible_by_sixteen(const int value) noexcept
{
	return (value % 16) == 0;
}

[[nodiscard]] inline bool is_divisible_by_sixteen_nonzero(const int value) noexcept
{
	return (value != 0) & ((value % 16) == 0);
}

#endif