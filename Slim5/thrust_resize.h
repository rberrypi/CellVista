#pragma once
#ifndef THRUST_RESIZE_H
#define THRUST_RESIZE_H
#include <thrust/device_vector.h>
#include "cuda_error_check.h"
#include "qli_runtime_error.h"
//maybe wrap in a namespace?

#define thrust_safe_resize(in,elements) thrust_safe_resize_imp(in, elements, __FILE__,__LINE__)
static bool thrust_enforce_no_allocation = false;

template<typename T> 
static void thrust_safe_resize_imp(T& in, size_t elements, const char* file, const int line)
{
	if (in.size() != elements)// in case this isn't implemented in the vector, already
	{
		const auto print_debug=[&]()
		{
			std::cout << "Error in GPU Memory Allocation: " << elements << std::endl;
			cuda_memory_debug(file, line);
		};
		if (thrust_enforce_no_allocation)
		{
			print_debug();
			const auto message = "Shouldn't memory reallocate, this means the buffer is wasn't pre-allocated!";
			qli_runtime_error(message);
		}
		try
		{
			auto old_capacity = in.capacity();
			in.resize(elements);
			if (in.capacity() != old_capacity)
			{
				cuda_memory_debug(file, line);
			}
		}
		catch (const thrust::system_error& e)
		{
			std::cout << e.what() << std::endl;			
			print_debug();
			throw;
		}
		catch (...)
		{
			print_debug();
			throw;
		}
	}
}
template<typename T> 
static T* thrust_safe_get_pointer(thrust::device_vector<T>& in, size_t elements)
{
	if (elements == 0)
	{
		qli_invalid_arguments();
	}
	thrust_safe_resize(in, elements);
	return thrust::raw_pointer_cast(in.data());
}


#endif