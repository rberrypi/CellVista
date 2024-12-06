#include "stdafx.h"
#include "slim_four.h"
#include "ui_slim_four.h"

void slim_four::set_capture_progress(const size_t current) const
{
	QMetaObject::invokeMethod(ui_->progressBarCapture, "setValue", Qt::QueuedConnection, Q_ARG(int, current));
}

void slim_four::set_capture_total(const size_t total) const
{
	if (total > 0)
	{
		QMetaObject::invokeMethod(ui_->progressBarCapture, "setMaximum", Qt::QueuedConnection, Q_ARG(int, total));
	}
	else
	{
		QMetaObject::invokeMethod(ui_->progressBarCapture, "reset", Qt::QueuedConnection);
	}
}

void slim_four::set_io_progress(const size_t left)  const
{
	QMetaObject::invokeMethod(ui_->progressBarIO, "setValue", Qt::QueuedConnection, Q_ARG(int, left));
}

void slim_four::set_io_progress_total(const size_t total)  const
{
	if (total > 0)
	{
		QMetaObject::invokeMethod(ui_->progressBarIO, "setMaximum", Qt::QueuedConnection, Q_ARG(int, total));
	}
	else
	{
		QMetaObject::invokeMethod(ui_->progressBarIO, "reset", Qt::QueuedConnection);
	}
}

void slim_four::set_io_buffer_progress(const size_t current) const
{
	QMetaObject::invokeMethod(ui_->qsbIOBuffers, "setValue", Qt::QueuedConnection, Q_ARG(int, current));
}
