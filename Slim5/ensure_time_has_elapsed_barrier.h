#pragma once
#ifndef ENSURE_TIME_HAS_ELAPSED_BARRIER_H
#define ENSURE_TIME_HAS_ELAPSED_BARRIER_H
#include <chrono>
#include <mutex>
class ensure_time_has_elapsed_barrier
{
	std::chrono::microseconds start, barrier_to_ensure;
	int optional_value_to_check;
	std::mutex protect_barrier;
public:
	ensure_time_has_elapsed_barrier();
	void set_barrier(const std::chrono::microseconds& set_to_time, int optional_value);
	void wait_for_barrier_to_pass(int optional_value);
};

#endif