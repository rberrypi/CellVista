#pragma once
#ifndef TIME_PRIORITY_H
#define TIME_PRIORITY_H
#include <boost/noncopyable.hpp>
struct time_priority : private boost::noncopyable
{
	static const int p = 1;
	time_priority(time_priority const&) = delete;
	time_priority& operator=(time_priority const&) = delete;
	time_priority();
};
#endif