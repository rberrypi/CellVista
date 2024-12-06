#include "stdafx.h"
#include "snapshot_label.h"

snapshot_label::snapshot_label(QWidget* parent) : QLineEdit(parent),count(0)
{
	this->setReadOnly(true);
}

void snapshot_label::set_processing(const processing_double& processing) noexcept
{
	this->processing = processing;
	update_text();
}

void snapshot_label::set_label(const std::string& label) noexcept
{
	this->label=QString::fromStdString(label);
	update_text();
}

void snapshot_label::set_capture_count(const int count)  noexcept
{
	this->count = count;
	update_text();
}

int snapshot_label::get_capture_count() const noexcept
{
	return count;
}

void snapshot_label::update_text() noexcept
{
	const auto is_raw_frame = processing.is_raw_frame();
	const auto& processing_label = phase_processing_setting::settings.at(processing.processing).label;
	const auto new_label =is_raw_frame ? QString("Snap_%1_#%2.tif").arg(label).arg(count) : QString("Snap_%1_%2_#%3.tif").arg(label).arg(QString::fromStdString(processing_label)).arg(count);
	setText(new_label);
}
