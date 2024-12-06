#include "stdafx.h"

std::chrono::microseconds timestamp() noexcept
{
	const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
	return microseconds;
}

const char* logic_error = "Logic Error";
