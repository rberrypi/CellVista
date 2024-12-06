#include "stdafx.h"
#include "ensure_time_has_elapsed_barrier.h"
ensure_time_has_elapsed_barrier::ensure_time_has_elapsed_barrier() : start(0), barrier_to_ensure(0),
optional_value_to_check(0)
{
}


void ensure_time_has_elapsed_barrier::set_barrier(const std::chrono::microseconds& set_to_time, const  int optional_value)
{
	std::lock_guard<std::mutex> lk(protect_barrier);
	optional_value_to_check = optional_value;
	barrier_to_ensure = set_to_time;
	start = timestamp();
}

void ensure_time_has_elapsed_barrier::wait_for_barrier_to_pass(const int optional_value)
{
	std::lock_guard<std::mutex> lk(protect_barrier);
	if (barrier_to_ensure.count() > 0)
	{
		const auto now = timestamp();
		const auto wait_until_this_time = start + barrier_to_ensure;
		if (now < wait_until_this_time)
		{
			const auto time_waited = now - start;
			const auto time_left = barrier_to_ensure - time_waited;
			windows_sleep(time_left);
		}
#if 0
		else
		{

			const auto time_between_calls = now - start;
			std::cout << "Time Between Calls " << to_mili(time_between_calls) << std::endl;
		}
#endif
	}
}
