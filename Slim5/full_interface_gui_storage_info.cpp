#include "stdafx.h"
#include <QStorageInfo>
#include <QTimer>
#include "full_interface_gui.h"
void full_interface_gui::setup_disk_size()
{
	auto* timer = new QTimer(this);
	const auto wrangle_disk_size = [&] {
		const auto dir = get_dir();
		const auto valid_dir_ = !dir.isEmpty() && QDir(dir).exists();
		if (valid_dir_)
		{
			const QStorageInfo info(dir);
			const auto available_bytes = info.bytesAvailable();
			set_available_bytes(available_bytes);
		}
		else
		{
			set_available_bytes(0);
		}
	};
	wrangle_disk_size();
	connect(timer, &QTimer::timeout, this, wrangle_disk_size);
	timer->start(2000);
}