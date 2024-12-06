#pragma once
#ifndef SNAPSHOT_LABEL_H
#define SNAPSHOT_LABEL_H
#include <QLineEdit>
#include "phase_processing.h"
class snapshot_label final : public QLineEdit
{
	Q_OBJECT

	int count;
	QString label;
	void update_text() noexcept;
	processing_double processing;
public:
	explicit snapshot_label(QWidget* parent = nullptr);

	[[nodiscard]] int get_capture_count() const noexcept;
	
public slots:
	void set_processing(const processing_double& processing) noexcept;
	void set_label(const std::string& label) noexcept;
	void set_capture_count(int count) noexcept;
};
#endif