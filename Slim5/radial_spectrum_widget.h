#pragma once
#ifndef RADIAL_SPECTRUM_WIDGET_H
#define RADIAL_SPECTRUM_WIDGET_H
#include <QWidget>
// ReSharper disable once CppInconsistentNaming
class QCustomPlot;
class radial_spectrum_widget final : public QWidget
{
public:
	explicit radial_spectrum_widget(QWidget* parent);
	QCustomPlot* custom_plot;
	QVector<qreal> axis, data;//this is generated each time.
	float pixel_ratio;
	void load_radial_ft_average();
};

#endif