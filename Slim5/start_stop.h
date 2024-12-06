#pragma once
#ifndef START_STOP
#define START_STOP
#include <functional>
#include <boost/core/noncopyable.hpp>

class start_stop final : boost::noncopyable
{
	const std::function< void() > stop_;
public:
	start_stop(const std::function< void() >& start, const std::function< void() >& stop) : stop_(stop)
	{
		start();
	}
	~start_stop()
	{
		stop_();
	}
};
#endif