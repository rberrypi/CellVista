#include "stdafx.h"
#include "qli_styled_slim_application_with_debug.h"
#include <QGLFormat>
#include "cuda_error_check.h"
#include <QResource>
#include <QSplashScreen>
#include <QStyleFactory>
#include "qli_runtime_error.h"
#include "time_slice.h"
#include "cuda_runtime_api.h"
extern void fix_windows_console_selection();
Q_DECLARE_METATYPE(std::chrono::microseconds);
qli_styled_slim_application_with_debug::qli_styled_slim_application_with_debug(int& argc, char** argv) : QApplication_with_debug(argc, argv)
{
	//Step GUI
	fix_windows_console_selection();
	//Test Resources
	{
		const std::string resource_name = "slim5.rcc";
		const auto success = QResource::registerResource(resource_name.c_str());
		if (!success)
		{
			const auto str = "Missing " + resource_name;
			qli_runtime_error(str);
		}
	}
#if _DEBUG
	// ReSharper disable once StringLiteralTypo
	const auto pix_map_path=":/images/splashwhite_debug.png";
#else
	const auto pix_map_path=":/images/splashwhite.png";
#endif
	splash.setPixmap(QPixmap(pix_map_path));
	splash.show();
	setStyle(QStyleFactory::create("Fusion"));
	auto dark_palette = qApp->palette();
	dark_palette.setColor(QPalette::Highlight, QColor(42, 130, 218));
	dark_palette.setColor(QPalette::HighlightedText, Qt::black);
	dark_palette.setColor(QPalette::Highlight, QColor(255, 109, 10));// Illinois Orange
	setPalette(dark_palette);
	//
	setOrganizationName("QLI LAB at UIUC");
	setOrganizationDomain("light.ece.illinois.edu");
	setApplicationName("SLIMe");
	qRegisterMetaType<std::chrono::microseconds>();
	{
		std::cout << "Loading GPU Kernels" << std::endl << "(if this hangs you're using an unsupported GPU)" << std::endl;
		time_slice ts("GPU Kernels Took:");
		CUDASAFECALL(cudaSetDevice(0));
		CUDASAFECALL(cudaFree(nullptr));
		CUDASAFECALL(cudaDeviceSynchronize());
	}
	//OGL Version
	{
		if (QGLFormat::openGLVersionFlags() < QGLFormat::OpenGL_Version_4_3)
		{
			qli_runtime_error("This system does not support OpenGL 4.3 Contexts!");
		}
		QSurfaceFormat format;
		format.setRenderableType(QSurfaceFormat::OpenGL);
		QSurfaceFormat::setDefaultFormat(format);
	}
}

std::shared_ptr<qli_styled_slim_application_with_debug> qli_styled_slim_application_with_debug::get(int& argc, char** argv)
{
	setAttribute(Qt::ApplicationAttribute::AA_DontCheckOpenGLContextThreadAffinity);
	setAttribute(Qt::ApplicationAttribute::AA_EnableHighDpiScaling);
	setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::Floor);
	return std::make_shared<qli_styled_slim_application_with_debug>(argc,argv);
}


void qli_styled_slim_application_with_debug::splash_finish(QWidget *mainWin)
{
	if (mainWin)
	{
		splash.finish(mainWin);
	}
	else
	{
		splash.close();
	}
}

qli_styled_slim_application_with_debug::~qli_styled_slim_application_with_debug() = default;
