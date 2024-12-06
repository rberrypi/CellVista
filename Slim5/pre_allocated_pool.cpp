#include "stdafx.h"
#include "pre_allocated_pool.h"

#include <algorithm>
#include <iostream>

#include "qli_runtime_error.h"
#include "time_slice.h"

pre_allocated_pool::pre_allocated_pool(const size_t total_size) : total_size_(total_size), front_pointer_(0), back_pointer_(0)
{
	if (total_size < 64)
	{
		qli_invalid_arguments();
	}
	std::cout << "Initializing Memory Buffer" << std::endl;
	time_slice ts("Took:");
	bulk_ = static_cast<unsigned char*>(std::calloc(total_size, sizeof(unsigned char)));
	reset_queue();
	if (bulk_ == nullptr)
	{
		const auto* msg = "Allocation Failed";
		qli_runtime_error(msg);
	}
	std::memset(bulk_, 255, total_size);
}

void pre_allocated_pool::reset_queue()
{
	std::unique_lock<std::mutex> lk(buffer_modify_);
	back_pointer_ = 0;
	front_pointer_ = 64;
}

pre_allocated_pool::~pre_allocated_pool()
{
	delete bulk_;
}

unsigned char* pre_allocated_pool::get(const size_t bytes)
{
	std::unique_lock<std::mutex> lk(buffer_modify_);
	auto new_front_pointer = front_pointer_ + bytes;
	if (new_front_pointer >= total_size_)
	{
		new_front_pointer = 0 + bytes;
		front_pointer_ = 0;//dirty hack
	}
	//todo ensure avx memory alignment!!
	const auto forward_jump = back_pointer_ >= front_pointer_ && back_pointer_ <= new_front_pointer;
	if (forward_jump)
	{
		const auto* const msg = "Ran out of buffer, try reducing acquisition rate";
		std::cout << msg << std::endl;
		return nullptr;
	}
	auto* const return_me = &bulk_[front_pointer_];
	front_pointer_ = new_front_pointer;
	return return_me;
}

void pre_allocated_pool::put_back(const size_t bytes)
{
	std::unique_lock<std::mutex> lk(buffer_modify_);
	back_pointer_ = back_pointer_ + bytes;
	if (back_pointer_ >= total_size_)
	{
		back_pointer_ = 0;
	}
}

size_t pre_allocated_pool::get_bytes_ram_available()
{
	MEMORYSTATUSEX status;
	status.dwLength = sizeof status;
	GlobalMemoryStatusEx(&status);
	// ReSharper disable once CppLocalVariableMayBeConst
	auto gigs = static_cast<int>((status.ullAvailPhys / (1024 * 1024 * 1024)));
	//auto gigs = getGigabytesRam();
#if SIMULATE_OVERFLOW==1
	gigs = 1;
#else
	const auto ram_requirement = 5;
	if (gigs < ram_requirement)
	{
		const auto message =  "Error: failed to allocate memory buffer" + std::to_string( ram_requirement) + " gigabytes is required";
		qli_runtime_error(message);
	}
#endif
#if _DEBUG
	gigs = std::min(4, gigs);
#endif
	const auto gigabyte = 1073741824ull;
	const size_t eighty_percent = 0.70 * gigs;
	const size_t leave_ten_free = std::max(gigs - 10, 0);
	const auto one = static_cast<size_t>(1);
	// ReSharper disable once CppInitializedValueIsAlwaysRewritten
	// ReSharper disable CppLocalVariableMayBeConst
	auto gig_numbers = std::max(std::max(eighty_percent, leave_ten_free), one);
	// ReSharper restore CppLocalVariableMayBeConst
#ifdef _DEBUG
	gig_numbers = 3;//for development, w/e
#endif
	gig_numbers = std::min(static_cast<size_t>(24), gig_numbers);
	std::cout << "Allocating " << gig_numbers << " Gigabytes" << std::endl;
	return gig_numbers * gigabyte;
}
