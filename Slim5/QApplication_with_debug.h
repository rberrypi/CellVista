#pragma once
#ifndef QApplication_with_debug_H
#define QApplication_with_debug_H
#include <QApplication>
#include <iostream>
class QApplication_with_debug : public QApplication
{
	Q_OBJECT

public:
	QApplication_with_debug(int& argc, char** argv) :QApplication(argc, argv)
	{}

private:
	bool notify(QObject* receiver_, QEvent* event_) override;
};
#endif