#pragma once
#ifndef TIME_GUARANTEE_H
#define TIME_GUARANTEE_H
#include <chrono>
#include <boost/noncopyable.hpp>

class time_guarantee final : boost::noncopyable
{
	const std::chrono::microseconds milliseconds_, start_;
	static void fancy_fence();
public:
	explicit time_guarantee(std::chrono::microseconds milliseconds);
	~time_guarantee();
};
#endif