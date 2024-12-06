#pragma once
#ifndef PRE_ALLOCATED_POOL_H
#define PRE_ALLOCATED_POOL_H
#include <mutex>
#include <boost/core/noncopyable.hpp>
class pre_allocated_pool final : boost::noncopyable
{
	size_t total_size_, front_pointer_, back_pointer_;
	std::mutex buffer_modify_;
	unsigned char* bulk_;
public:
	[[nodiscard]] size_t total_size() const noexcept { return total_size_; }
	pre_allocated_pool(const pre_allocated_pool& that) = delete;
	explicit pre_allocated_pool(size_t total_size = get_bytes_ram_available());
	~pre_allocated_pool();
	[[nodiscard]] unsigned char* get(size_t bytes);
	void put_back(size_t bytes);
	[[nodiscard]] static size_t get_bytes_ram_available();
	void reset_queue();

};
#endif