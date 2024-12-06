#include "stdafx.h"
#include "time_priority.h"
#include "internal_windows_timer_helpers.h"
time_priority::time_priority()
{
	set_actual_resolution(100);
}