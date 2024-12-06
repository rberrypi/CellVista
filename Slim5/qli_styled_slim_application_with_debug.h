#pragma once
#ifndef QLI_STYLED_SLIM_APPLICATION_WITH_DEBUG_H
#define QLI_STYLED_SLIM_APPLICATION_WITH_DEBUG_H
#include <QSplashScreen>
#include "QApplication_with_debug.h"
class qli_styled_slim_application_with_debug final : public QApplication_with_debug
{
	Q_OBJECT
	QSplashScreen splash;

public:
	qli_styled_slim_application_with_debug(int& argc, char** argv);//don't directly call this!
	static std::shared_ptr<qli_styled_slim_application_with_debug> get(int& argc, char** argv);
	void splash_finish(QWidget *mainWin=nullptr);
	virtual ~qli_styled_slim_application_with_debug();
};
#endif