#include "compute_engine_shared.h"
#include "qli_runtime_error.h"
[[nodiscard]] bool camera_frame_internal::is_valid() const noexcept
{
	if (!buffer)
	{
		return false;
	}
	return image_info::is_valid() && samples() == static_cast<int>(buffer->size());
}

void background_frame::load_buffer(const camera_frame_internal& frame)
{
	static_cast<internal_frame_meta_data&>(*this) = frame;
	buffer.resize(frame.samples());
#if _DEBUG
	if (!is_valid())
	{
		//something bad
		qli_runtime_error("Logic Problem");
	}
#endif
}