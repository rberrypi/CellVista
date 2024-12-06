#include "stdafx.h"
#include "time_guarantee.h"
#include <atomic>
void time_guarantee::fancy_fence()
{
	std::atomic_signal_fence(std::memory_order_seq_cst);
}

time_guarantee::time_guarantee(const std::chrono::microseconds milliseconds) : milliseconds_(milliseconds), start_(timestamp())
{
	fancy_fence();//don't optimize me out!
}

time_guarantee::~time_guarantee()
{
	fancy_fence();
	if (milliseconds_.count() > 0)
	{
		const auto elapsed = timestamp() - start_;
		if (elapsed < milliseconds_)
		{
			const auto left = milliseconds_ - elapsed;
			windows_sleep(left);
		}
	}

}