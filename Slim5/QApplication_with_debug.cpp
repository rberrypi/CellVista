#include "stdafx.h"
#include "QApplication_with_debug.h"

bool QApplication_with_debug::notify(QObject* receiver_, QEvent* event_)
{
	//will do wacky stuff but maybe the user will save the data?
	try
	{
		return QApplication::notify(receiver_, event_);
	}
	catch (std::exception& e)
	{
		std::cout << "Error:" << e.what() << std::endl;
		return false;
	}
	catch (...)
	{
		return false;
	}
}