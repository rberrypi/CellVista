#include "stdafx.h"
#if 1
//
#include "slim_four.h"
#include "device_factory.h"
#include "qli_styled_slim_application_with_debug.h"

// Welcome to your miserable life at QLI.
// If you are here this means that you are at the bottom 5%
// of the industry application pool and you probably got rejected
// from every interview or your resume is so weak that you didn't
// even get your foot in the door. As such the best place for you
// is in this lab with all other narrow minded and insecure fucks.

// If you think you deserve better than this, then use this as an
// opportunity to go to somewhere better.

// Also, if you are making less than 80k in this position which
// requires you to know ML, CUDA, and C++ you should really 
// consider finding a better job. If you are here in the hope
// that you might get to take classes, you need to get your head
// out your ass. That shit isn't happenning, not with your current
// supervisor. If you are a PhD student doing this shit - hope
// you like being exploited.


int main(int argc, char* argv[])
{
	auto a = qli_styled_slim_application_with_debug::get(argc,argv);
#if _DEBUG
	D = std::make_unique<device_factory>(virtual_camera_type::qdic_set_1);
#else
	D = std::make_unique<device_factory>(virtual_camera_type::prompt);
#endif
	int ret_code;
	{
		slim_four m_u(nullptr);//these 3 things should be returned
		a->splash_finish(&m_u);
		m_u.showMaximized();
		ret_code = qli_styled_slim_application_with_debug::exec();
	}
	return ret_code;
}
#endif

#if 0
#include <QResource>
#include "qli_runtime_error.h"
//SYSTEM TESTS
extern void system_tests();
extern void standalone_halo_removal(int argc, char* argv[]);
extern void cufft_wrapper_test();
extern void quick_camera_test();
extern void quick_scope_test();
extern void standalone_ml_processor();
extern void standalone_phase_retrieval();
extern void auto_contrast_tests();
extern int widget_tests(int argc, char* argv[]);
int main(const int argc, char* argv[])
{
	if (!QResource::registerResource("slim5.rcc"))
	{
		qli_runtime_error();
	}
	system_tests();
	return 0;
}

#endif
