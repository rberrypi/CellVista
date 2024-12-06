#include "stdafx.h"
#include "radial_spectrum_widget.h"
#include <QVBoxLayout>
#include "qcustomplot.h"

radial_spectrum_widget::radial_spectrum_widget(QWidget* parent) : QWidget(parent), pixel_ratio(1)
{
	auto* layout = new QVBoxLayout();
	custom_plot = new QCustomPlot();
	layout->addWidget(custom_plot);
	setLayout(layout);
}

void radial_spectrum_widget::load_radial_ft_average()
{
	if (data.empty())
	{
		return;
	}
	static auto number_of_elements = -1;
	const auto add_graph = number_of_elements != data.size();
	number_of_elements = data.size();
	if (add_graph)
	{
		custom_plot->clearGraphs();
		auto graph = custom_plot->addGraph();
		QVector<qreal> key_data;//static these?
		key_data.reserve(number_of_elements);
		QVector<qreal> value_data;
		value_data.reserve(number_of_elements);
		for (auto i = 0; i < number_of_elements; i++)//iota?
		{
			key_data << i;
			value_data << i;
		}
		graph->setData(key_data, value_data);
		custom_plot->xAxis->setTickLabels(true);
		custom_plot->yAxis->setTickLabels(false);
	}
	//
	//
	auto&& inside = custom_plot->graph(0)->data();
	auto counter = 0;
	auto scale_factoid = 2 * M_PI / (1 / pixel_ratio) / number_of_elements;
	std::transform(data.begin(), data.end(), inside->begin(), [&](const qreal val)
	{
		counter = counter + 1;  return QCPGraphData(counter * scale_factoid, val);
	});
	custom_plot->xAxis->rescale();
	custom_plot->yAxis->rescale();
	custom_plot->yAxis->setLabel("Spectrum (log1p)");
	custom_plot->xAxis->setLabel("Frequency Bins (q)");
	custom_plot->replot();
}