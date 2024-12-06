#pragma once
#ifndef PAGE_LOCKED_ALLOCATOR_H
#define PAGE_LOCKED_ALLOCATOR_H

struct page_locked_allocator_impl
{
	static void* allocation_impl(size_t bytes);
	static void deallocate_impl(void* p, size_t bytes);
	static void page_locked_increment(int inc);
};

template <class T>
struct page_locked_allocator : private page_locked_allocator_impl {
	typedef T value_type;
	page_locked_allocator() = default;
	template <class U> constexpr page_locked_allocator(const page_locked_allocator<U>&) noexcept
	{

	}
	T* allocate(const std::size_t n)
	{
		if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
		{
			throw std::bad_alloc();
		}
		//if (auto p = static_cast<T*>(std::malloc(n * sizeof(T))))
		if (auto p = static_cast<T*>(page_locked_allocator_impl::allocation_impl(n * sizeof(T))))
		{
			return p;
		}
		throw std::bad_alloc();
	}
	static void deallocate(T* p, const std::size_t bytes) noexcept
	{
		//std::free(p);
		page_locked_allocator_impl::deallocate_impl(p, bytes);
	}
};
template <class T, class U>
bool operator==(const page_locked_allocator<T>&, const page_locked_allocator<U>&) { return true; }
template <class T, class U>
bool operator!=(const page_locked_allocator<T>&, const page_locked_allocator<U>&) { return false; }

#endif