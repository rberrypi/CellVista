#pragma once
#ifndef RUNTIME_ERROR_HELPERS
#define RUNTIME_ERROR_HELPERS
#include <string>
#include <iostream>

using std::cout;
using std::endl;

#ifndef qli_assert
#   define qli_assert(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#endif

#ifndef qli_runtime_error
#define qli_runtime_error(error) {\
	const std::string error_str = std::string(error).empty() ? "Miscellaneous Runtime Error" : std::string(error); \
	std::cout << __FILE__ << ":" << __LINE__ << ": " << error_str << std::endl; \
	throw std::runtime_error(error_str); \
}
#endif

#ifndef qli_not_implemented
#define qli_not_implemented() {\
	std::cout << __FILE__ << ":" << __LINE__ << ": Not Implemented" << std::endl; \
	throw std::runtime_error("Not Implemented"); \
}
#endif

#ifndef qli_invalid_arguments
#define qli_invalid_arguments() {\
    const char* error_msg = "Function Preconditions Failed";    \
    std::cout << __FILE__ << ":" << __LINE__ << ": " << error_msg << std::endl;    \
    throw std::invalid_argument(error_msg); \
}
#endif


#ifndef qli_gui_mismatch
#define qli_gui_mismatch() {\
    const char* const error_msg = "GUI Mismatch";   \
    std::cout << __FILE__ << ":" << __LINE__ << ": " << error_msg << std::endl;    \
    throw std::runtime_error(error_msg);    \
}
#endif // !qli_gui_mismatch


#endif