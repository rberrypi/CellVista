#pragma once
#ifndef THREAD_PRIORITY_H
#define THREAD_PRIORITY_H
#include "time_priority.h"
struct thread_priority final : time_priority
{
	thread_priority(thread_priority const&) = delete;
	thread_priority& operator =(thread_priority const&) = delete;
	explicit thread_priority(int core = -1);
};
#endif