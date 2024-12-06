#include "stdafx.h"
#include "page_locked_allocator.h"
#include "qli_runtime_error.h"
#include <cassert>
#include <Memoryapi.h>
void* page_locked_allocator_impl::allocation_impl(const size_t bytes)
{
	const SIZE_T dwSize = bytes;
	const DWORD flAllocationType = MEM_COMMIT | MEM_RESERVE;
	const DWORD flProtect = PAGE_READWRITE;
	const auto lpAddress = VirtualAlloc(
		nullptr,
		dwSize,
		flAllocationType,
		flProtect
	);
	page_locked_increment(bytes);
	const auto success = VirtualLock(lpAddress, dwSize);
	if (lpAddress == NULL || success == NULL)
	{
		const auto some_error = GetLastError();
		const auto bad_alloc_message = "Bad Alloc: "  + std::to_string(some_error);
		throw std::bad_alloc();
		//qli_runtime_error(bad_alloc_message);
	}
	return lpAddress;
}

void page_locked_allocator_impl::page_locked_increment(const int inc)
{
	const auto process_handle = GetCurrentProcess();
	SIZE_T minWorkingSet, maxWorkingSet;
	BOOL bRes = GetProcessWorkingSetSize(process_handle, &minWorkingSet, &maxWorkingSet);
	assert(bRes);
	SIZE_T newWorkingSetSize = maxWorkingSet + inc;
	bRes = SetProcessWorkingSetSize(process_handle, newWorkingSetSize, newWorkingSetSize);
	assert(bRes);

}

void page_locked_allocator_impl::deallocate_impl(void* p, size_t bytes)
{
	//no throw lol
	VirtualFree(p, bytes, MEM_RELEASE);
	page_locked_increment((-1));
}
