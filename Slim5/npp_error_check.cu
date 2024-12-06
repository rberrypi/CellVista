#include "npp_error_check.h"
#include <iostream>
#include <sstream>
#include "qli_runtime_error.h"
void npp_safe_call(NppStatus err, const char* file, const int line)
{
	if (NPP_SUCCESS != err)
	{
		std::stringstream error_msg;
		error_msg << "An NPP failed at " << file << ":" << line << ":" << err;
		const auto error_msg_str = error_msg.str();
		qli_runtime_error(error_msg_str);
	}
}