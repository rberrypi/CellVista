#include "stdafx.h"

#include <QResource>
#include "qli_styled_slim_application_with_debug.h"
#include "device_factory.h"
#include "slm_control.h"
#include "settings_file.h"
extern void style_q_application();
int widget_tests(int argc, char* argv[])
{
	//
	auto a = qli_styled_slim_application_with_debug::get(argc,argv);
	a->splash_finish();
	QResource::registerResource("slim5.rcc");
	D = std::make_unique<device_factory>(virtual_camera_type::neurons_1);
	auto* control = new slm_control;
	QObject::connect(control,&slm_control::settings_file_changed,[&](const settings_file& new_settings)
	{
		D->load_slm_settings(new_settings.modulator_settings,false);
		control->reload_modulator_surface();
	});
	auto default_settings = settings_file::generate_default_settings_file(phase_retrieval::camera);
	default_settings.file_path = R"(C:\tests\Test.json)";
	control->set_settings_file(default_settings);
	control->show();
	return qli_styled_slim_application_with_debug::exec();
}