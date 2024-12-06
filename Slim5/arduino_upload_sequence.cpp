#include "stdafx.h"
#include "com_persistent_device.h"
#include "qli_runtime_error.h"
bool com_persistent_device::program_arduino(const std::string& asset_name, int)
{
	if (asset_name.empty())
	{
		return true;
	}
	qli_not_implemented();
}
