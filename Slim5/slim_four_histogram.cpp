#include "stdafx.h"
#include "slim_four.h"
#include "render_widget.h"
#include <iostream>
std::array<std::array<QCPItemLine*, 2>, 3> arrows;
#include "ui_slim_four.h"

void slim_four::setup_auto_contrast() const
{
	const auto afc_functor = [&]
	{
		auto current = ui_->wdg_display_settings->get_display_settings();
		const auto get_histogram_ranges = [&]
		{
			std::unique_lock<std::mutex> histogram(render_surface_->histogram_m);
			return render_surface_->histogram.predict_display_ranges();
		};
		auto predicted_ranges = get_histogram_ranges();
		//maybe std::transform next time
		for (auto c = 0; c < current.ranges.size(); ++c)
		{
			auto& current_range = current.ranges.at(c);
			auto& predicted_range = predicted_ranges.at(c);
			const auto bottom_current = current_range.min;
			const auto top_current = current_range.max;
			auto bottom = predicted_range.min;
			auto top = predicted_range.max;
			const static auto change_threshold = 0.02f;
			const auto percent_change = [](auto a, auto b) { return abs((b - a) / a); };
			const auto current_change_top = percent_change(top, top_current);
			const auto current_change_bottom = percent_change(bottom, bottom_current);
			const auto no_change = current_change_top < change_threshold&& current_change_bottom < change_threshold;
			if (no_change)
			{
				const auto processing = ui_->processing_quad->get_quad().processing;
				const auto expected_range = phase_processing_setting::settings.at(processing).display_range;
				bottom = expected_range.min;
				top = expected_range.max;
			}
			current_range.min = bottom;
			current_range.max = top;
		}
		ui_->wdg_display_settings->set_display_settings(current);
	};
	QObject::connect(ui_->btn_auto_contrast, &QPushButton::clicked, afc_functor);
}

void slim_four::load_auto_contrast_settings(const display_settings::display_ranges& range) const
{
	auto current_settings = ui_->wdg_display_settings->get_display_settings();
	current_settings.ranges = range;
	ui_->wdg_display_settings->set_display_settings(current_settings);
}

void slim_four::load_histogram()
{
	const auto display_ranges = ui_->wdg_display_settings->get_display_settings().ranges;
	std::unique_lock<std::mutex> lk(render_surface_->histogram_m);
	static auto samples_per_pixel = -1;
	const auto bit_depth = render_surface_->samples_per_pixel;//wtf is this?
	const auto bit_depth_changed = bit_depth != samples_per_pixel;
	if (bit_depth_changed)
	{
		samples_per_pixel = bit_depth;
		const auto illinois_orange = QColor(255, 109, 10);
		const auto pomegranate = QColor(192, 57, 43);
		const auto nephritis = QColor(39, 174, 96);
		const auto belize_hole = QColor(41, 128, 185);
		constexpr auto alpha_scale = 20;
		std::array<QPen, 4> pens = { pomegranate ,nephritis ,belize_hole ,illinois_orange };
		std::array<QBrush, 4> brushes = { QBrush(pomegranate),QBrush(nephritis), QBrush(belize_hole), QBrush(illinois_orange) };
		for (auto&& fix_me : brushes)
		{
			auto color = fix_me.color();
			color.setAlpha(alpha_scale);
			fix_me.setColor(color);
		}
		//
		ui_->widgetHistogram->clearItems();
		ui_->widgetHistogram->clearPlottables();
		ui_->widgetHistogram->clearGraphs();
		for (auto graph_idx = 0; graph_idx < samples_per_pixel; ++graph_idx)
		{
			auto* graph = ui_->widgetHistogram->addGraph();
			const auto mono = samples_per_pixel == 1;
			const auto pen = mono ? pens.back() : pens.at(graph_idx);
			const auto brush =mono ? brushes.back() : brushes.at(graph_idx);
			graph->setPen(pen);// Illinois Orange
			graph->setBrush(brush);// Illinois Orange
			constexpr auto two_fifty_six = 256;
			QVector<qreal> key_data(two_fifty_six);
			QVector<qreal> value_data(two_fifty_six);
			std::iota(key_data.begin(), key_data.end(), static_cast<qreal>(0));
			std::iota(value_data.begin(), value_data.end(), static_cast<qreal>(0));
			graph->setData(key_data, value_data);
			//
			if (samples_per_pixel > 1)
			{
				for (auto what : { 0,1 })
				{
					auto* item = new QCPItemLine(ui_->widgetHistogram);
					item->setHead(QCPLineEnding::esFlatArrow);
					item->setTail(QCPLineEnding::esNone);
					item->setPen(pen);
					arrows.at(graph_idx).at(what) =item;
				}
			}

		}
		ui_->widgetHistogram->xAxis->rescale();
		ui_->widgetHistogram->xAxis->setTickLabels(false);
		ui_->widgetHistogram->yAxis->setTickLabels(false);
		ui_->widgetHistogram->yAxis->setScaleType(QCPAxis::ScaleType::stLogarithmic);
		//
		label_ = new QCPItemText(ui_->widgetHistogram);
		label_->setPositionAlignment(Qt::AlignTop | Qt::AlignRight);
		label_->position->setType(QCPItemPosition::ptAxisRectRatio);
		label_->position->setCoords(0.9, 0); // place position at center/top of axis rect
		//
		//
	}
	//
	QString mean_label("m"), var_label;
	const static QChar math_symbol_sigma(0x03C3);
	var_label.prepend(math_symbol_sigma);
	for (auto i = 0; i < samples_per_pixel; ++i)
	{
		auto counter = 0;
		auto&& inside = ui_->widgetHistogram->graph(i)->data();
		auto&& channel = render_surface_->histogram.histogram_channels.at(i);
		auto&& info = render_surface_->histogram.info.at(i);
		auto display_range = display_ranges.at(i);
		//convert from compressed to uncompressed range, aka [0,255] to [min,max]
		std::transform(channel.begin(), channel.end(), inside->begin(), [&](auto val)
		{
			const auto x_axis = (display_range.max - display_range.min) * (counter - 0) / (255 - 0) + display_range.min;
			counter = counter + 1;
			return QCPGraphData(x_axis, val);
		});
		const auto mean = QString::asprintf(": %8.3f", info.median);
		const auto var = QString::asprintf(": %8.3f", info.standard_deviation);
		mean_label.append(mean);
		var_label.append(var);
	}
	const auto label = QString("%1\n%2").arg(mean_label,var_label);
	label_->setText(label);
	//
	ui_->widgetHistogram->yAxis->rescale();
	static display_settings::display_ranges old_display_ranges;
	if (old_display_ranges != display_ranges || bit_depth_changed)
	{
		ui_->widgetHistogram->xAxis->rescale();
		old_display_ranges = display_ranges;
		//update arrows
		if (samples_per_pixel > 1)
		{
			for (auto graph_idx = 0; graph_idx < samples_per_pixel; ++graph_idx)
			{
				const auto start = ui_->widgetHistogram->graph(graph_idx)->data()->begin()->key;
				const auto stop = (ui_->widgetHistogram->graph(graph_idx)->data()->end() - 1)->key;
				const auto range = ui_->widgetHistogram->yAxis->range();
				arrows[graph_idx][0]->start->setCoords(start, range.upper / 4);
				arrows[graph_idx][0]->end->setCoords(start, range.lower);
				arrows[graph_idx][1]->start->setCoords(stop, range.upper / 4);
				arrows[graph_idx][1]->end->setCoords(stop, range.lower);
			}
		}
	}
	ui_->widgetHistogram->replot();
}