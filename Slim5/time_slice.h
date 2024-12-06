#pragma once
#ifndef TIME_SLICE_H
#define TIME_SLICE_H
#include <string>
#include <chrono>
#include <boost/noncopyable.hpp>

class time_slice : boost::noncopyable
{
	const bool silent_;
	const std::chrono::microseconds start_;
	const std::string name_;
	static inline void fancy_fence() noexcept;
public:
	explicit time_slice(std::string  name, bool silent = false) noexcept;
	~time_slice();

};

#endif