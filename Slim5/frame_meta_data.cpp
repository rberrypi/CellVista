#include "stdafx.h"
#include "frame_meta_data.h"

[[nodiscard]] bool frame_meta_data_before_acquire::is_valid() const noexcept
{
	return exposure_time.count() > 0;
}