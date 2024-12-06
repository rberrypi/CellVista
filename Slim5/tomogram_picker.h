#pragma once
#ifndef TOMOGRAM_PICKET_H
#define TOMOGRAM_PICKET_H

#include <QDialog>

struct tomogram final
{
	double z, z_inc;
	int steps;
};
namespace Ui {
	class tomogram_picker;
}

class tomogram_picker final : public QDialog
{
	Q_OBJECT

		std::unique_ptr<Ui::tomogram_picker> ui_;

public:
	explicit tomogram_picker(float xy_pixel_ratio, QWidget* parent);

public slots:
	void goto_top(bool enable) const;
	void goto_bottom(bool enable) const;
	void increment_change() const;
	[[nodiscard]] tomogram get_tomogram() const;
signals:
	void add_tomogram(tomogram tomogram);
};

#endif