#include "stdafx.h"
#include "psi_function_pairs_selector.h"
#include "psi_function_pair_selector.h"
#include "ui_psi_function_pairs_selector.h"
#include "qli_runtime_error.h"
#include <iostream>
void psi_function_pairs_selector::update_psi_function_pairs()
{
	const auto value = this->get_psi_function_pairs();
	emit psi_function_pairs_changed(value);
}

psi_function_pairs_selector::psi_function_pairs_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::make_unique<Ui::psi_function_pairs_selector>();
	ui_->setupUi(this);
	const psi_function_pairs blank(1);
	set_psi_function_pairs(blank);
}

void psi_function_pairs_selector::set_psi_function_pairs(const psi_function_pairs& settings)
{
#if _DEBUG
	{
		const auto items = settings.size();
		if (!(items == 3 || items == 1))
		{
			qli_invalid_arguments();
		}
	}
#endif
	const auto no_change =  [&]
	{
			const auto old_values = this->get_psi_function_pairs();
			const auto match = per_pattern_modulator_settings::psi_function_pairs_approx_equals(old_values, settings);
			return match;
	}();
	if (no_change)
	{
		return;
	}
	const auto add_item = [&](const psi_function_pair& new_item, const int idx)
	{
		auto* wdg = new psi_function_pair_selector;
		wdg->set_psi_function_pair(new_item);		
		QObject::connect(wdg,&psi_function_pair_selector::psi_function_pair_changed,this,&psi_function_pairs_selector::update_psi_function_pairs);
		ui_->horizontalLayout->addWidget(wdg);
		psi_function_widgets_.push_back(wdg);
	};
	//set or push
	for (auto i = 0; i < settings.size(); ++i)
	{
		const auto& setting = settings.at(i);
		if (i >= psi_function_widgets_.size())
		{
			add_item(setting, i);
		}
		auto& wdg = psi_function_widgets_.at(i);
		QSignalBlocker blocker(wdg);
		wdg->set_psi_function_pair(setting);
	}
	while (psi_function_widgets_.size() > settings.size())
	{
		auto* widget = psi_function_widgets_.back();
		psi_function_widgets_.pop_back();
		ui_->horizontalLayout->removeWidget(widget);
		widget->setParent(nullptr);
		delete widget;
	}
	update_psi_function_pairs();
#if _DEBUG
	{
		const auto what_we_got = get_psi_function_pairs();
		const auto match = per_pattern_modulator_settings::psi_function_pairs_approx_equals(what_we_got, settings);
		if (!match)
		{
			const auto print = [](const psi_function_pairs& pairs, const std::string& name )
			{
				std::cout << std::endl;				
				std::cout << "Title: " << name << " "<< pairs.size() <<  std::endl; 
				for (auto& pair : pairs)
				{
					std::cout << pair.top << "," <<pair.bot << "," << pair.constant << std::endl;
				}
			};
			print(settings,"what we set got");			
			print(what_we_got,"what we got");
			qli_gui_mismatch();
		}
	}
#endif
}

psi_function_pairs psi_function_pairs_selector::get_psi_function_pairs() const
{
	psi_function_pairs pairs;
	for (auto* item : psi_function_widgets_)
	{
		auto pair = item->get_psi_function_pair();
		pairs.push_back(pair);
	}
	return pairs;
}

void psi_function_pairs_selector::set_horizontal(const bool horizontal)
{
	for (auto* pair : psi_function_widgets_)
	{
		pair->set_horizontal(horizontal);
	}
}

