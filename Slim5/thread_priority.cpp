#include "stdafx.h"
#include "thread_priority.h"

#include "qli_runtime_error.h"

thread_priority::thread_priority(const int core)
{
	//int cpu = 0+1;
	if (core > -1)
	{
		const auto mask = static_cast<DWORD_PTR>(1) << core;//Result of shift cast to a larger size
		const auto ret = SetThreadAffinityMask(GetCurrentThread(), mask);
		if (!ret)
		{
			qli_runtime_error("Error setting CPU thread affinity");
		}
	}
	auto* pri = OpenProcess(PROCESS_ALL_ACCESS, true, GetCurrentProcessId());//might not work outside of a debug enviroement
	//DWORD oldPC = GetPriorityClass(pri);
	auto okay = SetPriorityClass(pri, REALTIME_PRIORITY_CLASS);
	if (!okay)
	{
		qli_runtime_error("Error setting process priority");
	}
	//DWORD newPC = GetPriorityClass(pri);
	auto* th = GetCurrentThread();
	//DWORD oldHPC = GetThreadPriority(th);
	okay = SetThreadPriority(th, THREAD_PRIORITY_TIME_CRITICAL);
	if (!okay)
	{
		qli_runtime_error("Error setting thread priority");
	}
	//DWORD newHPC = GetThreadPriority(th);
}
