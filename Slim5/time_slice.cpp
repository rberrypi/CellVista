#include "stdafx.h"
#include "time_slice.h"
#include <atomic>
#include <iostream>
#include <utility>

time_slice::time_slice(std::string  name, const bool silent) noexcept: silent_(silent), start_(timestamp()), name_(std::move(name))
{
	if (!silent)
	{
		fancy_fence();//don't optimize me out!
	}
}


time_slice::~time_slice()
{
	if (!silent_)
	{
		fancy_fence();
		const auto elapsed = timestamp() - start_;
		std::cout << name_ << " ";
		display_time(std::cout, elapsed);
		std::cout << std::endl;
	}
}

void time_slice::fancy_fence() noexcept
{
	std::atomic_signal_fence(std::memory_order_seq_cst);
}