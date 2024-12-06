#include "stdafx.h"
#include "exposure_sequence_control.h"
#include <QHBoxLayout>
#include <iostream>
#include "phase_processing.h"
#include "phase_shift_exposure_and_delay_selector.h"
#include "qli_runtime_error.h"

exposure_sequence_control::exposure_sequence_control(QWidget* parent) : QWidget{ parent }, current_phase_retrieval(phase_retrieval::camera)
{
	//nothing by default
}

bool phase_shift_exposures_and_delays_approx_equal(const phase_shift_exposures_and_delays& a, const phase_shift_exposures_and_delays& b, const std::chrono::microseconds& min_exposure)
{
	const auto predicate = [&](const phase_shift_exposure_and_delay& aa, const phase_shift_exposure_and_delay& bb)
	{
		return aa.approx_equal(bb, min_exposure);
	};
	const auto equal = std::equal(a.begin(), a.end(), b.begin(), b.end(), predicate);
	return equal;
}

void exposure_sequence_control::exposure_sequence_control::resize_exposures(const int patterns)
{
#if _DEBUG
	if (patterns<1)
	{
		qli_invalid_arguments();
	}
#endif
	auto current_settings = get_exposures_and_delays();
	const auto first_setting = current_settings.empty() ? phase_shift_exposure_and_delay() : current_settings.back();
	current_settings.resize(patterns, first_setting);
	set_phase_shift_exposures_and_delays(current_settings);
}

void exposure_sequence_control::set_phase_shift_exposures_and_delays(const phase_shift_exposures_and_delays& settings)
{
#if _DEBUG
	{
		if (settings.empty())
		{
			qli_invalid_arguments();
		}
	}
#endif
	std::unique_lock<std::recursive_mutex> lk(hack_);
	auto* main_layout = static_cast<QGridLayout*>(this->layout());//if empty do some half baked fixup?
	if (!main_layout)
	{
		auto* layout = new QGridLayout;
		layout->setMargin(0);
		setLayout(layout);
		set_phase_shift_exposures_and_delays(settings);
		return;
	}
	const auto add_item = [&](const phase_shift_exposure_and_delay& new_item, const int idx)
	{
		auto* wdg = new phase_shift_exposure_and_delay_selector;
		if (idx == 0)
		{
			QObject::connect(wdg, &phase_shift_exposure_and_delay_selector::phase_shift_exposure_and_delay_changed, this, &exposure_sequence_control::bulk_set_phase_shift_exposure_and_delay);
		}
		QObject::connect(wdg, &phase_shift_exposure_and_delay_selector::phase_shift_exposure_and_delay_changed, this, &exposure_sequence_control::update_values);
		main_layout->addWidget(wdg, idx / 2, idx % 2);
		setting_items_.push_back(wdg);
	};
	//set or push
	for (auto i = 0; i < settings.size(); ++i)
	{
		const auto& setting = settings.at(i);
		if (i >= setting_items_.size())
		{
			add_item(setting, i);
		}
		auto& wdg = setting_items_.at(i);
		const auto min_exposure = setting_items_.front()->get_minimum_exposure_time();
		QSignalBlocker blk(wdg);
		wdg->set_phase_shift_exposure_and_delay(setting);
		wdg->set_id(i);
		wdg->set_minimum_exposure_time(min_exposure);
	}
	while (setting_items_.size() > settings.size())
	{
		auto* widget = setting_items_.back();
		setting_items_.pop_back();
		main_layout->removeWidget(widget);
		widget->setParent(nullptr);
		delete widget;
	}
#if _DEBUG
	{
		const auto what_we_got = get_exposures_and_delays();
		const auto min_exposure = min_time();
		if (!phase_shift_exposures_and_delays_approx_equal(settings, what_we_got, min_exposure))
		{
			const auto print = [](const phase_shift_exposures_and_delays& a, const std::string& label)
			{
				std::cout << label << " ";
				for (auto& item : a)
				{
					std::cout << to_mili(item.slm_stability) << "," << to_mili(item.exposure_time);
				}
				std::cout << std::endl;
			};
			print(what_we_got, "what_we_got");
			print(settings, "settings");
			qli_gui_mismatch();
		}
	}
#endif
	current_phase_retrieval_changed(current_phase_retrieval);
}

void exposure_sequence_control::bulk_set_phase_shift_exposure_and_delay(const phase_shift_exposure_and_delay& phase_shift_exposure_and_delay)
{
	for (auto* wdg : setting_items_)
	{
		QSignalBlocker blk(wdg);
		wdg->set_phase_shift_exposure_and_delay(phase_shift_exposure_and_delay);
	}
	update_values();
}

void exposure_sequence_control::update_values()
{
	const auto values = get_exposures_and_delays();
	emit phase_shift_exposures_and_delays_changed(values);
}

std::chrono::microseconds exposure_sequence_control::min_time() const
{
	const auto min_exposure = setting_items_.front()->get_minimum_exposure_time();
	return min_exposure;
}

phase_shift_exposures_and_delays exposure_sequence_control::get_exposures_and_delays() const
{
	std::unique_lock<std::recursive_mutex> lk(hack_);
	phase_shift_exposures_and_delays delays;
	for (auto* item : setting_items_)
	{
		const auto value = item->get_phase_shift_exposure_and_delay();
		delays.push_back(value);
	}
	return delays;
}

void exposure_sequence_control::current_phase_retrieval_changed(const phase_retrieval phase_retrieval)
{
	//const auto patterns = phase_retrieval_setting::settings.at(phase_retrieval).modulator_patterns();
	// const auto force_all_the_patterns = patterns == pattern_count_from_file;
	// for (auto i = 0; i < setting_items_.size(); ++i)
	// {
	// 	const auto enable = force_all_the_patterns ? true : i < patterns;
	// 	setting_items_.at(i)->setEnabled(enable);
	// }
	current_phase_retrieval = phase_retrieval;
}

void exposure_sequence_control::set_minimum_exposure_time(const std::chrono::microseconds& minimum_time)
{
	//todo this should round up right?
	for (auto&& item : setting_items_)
	{
		item->set_minimum_exposure_time(minimum_time);
	}
}