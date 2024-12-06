#include "stdafx.h"
#include "fixed_modulator_settings_selector.h"
#include <QHBoxLayout>
#include "ui_fixed_modulator_settings_selector.h"
#include "qli_runtime_error.h"
fixed_modulator_settings_selector::fixed_modulator_settings_selector(QWidget* parent) : QWidget(parent)
{
	ui_ = std::unique_ptr<Ui::fixed_modulator_settings_selector>();
	ui_->setupUi(this);
}

fixed_modulator_settings fixed_modulator_settings_selector::get_fixed_modulator_settings() const
{
	fixed_modulator_settings return_me;
	for (const auto& slm : slm_widgets)
	{
		const auto settings = slm->get_per_modulator_saveable_settings();
		return_me.push_back(settings);
	}
	return return_me;
}

void fixed_modulator_settings_selector::set_pattern(const int pattern)
{
	for (auto& widgets : slm_widgets)
	{
		widgets->set_pattern(pattern);
	}
}

void fixed_modulator_settings_selector::reload_modulators()
{
	for (auto& widgets : slm_widgets)
	{
		//lazy evaluation so not that dangerous
		widgets->reload_modulator();
	}
}

void fixed_modulator_settings_selector::set_darkfield_mode(const darkfield_mode mode)
{
	for (auto& widgets : slm_widgets)
	{
		widgets->set_darkfield_mode(mode);
	}
}

void fixed_modulator_settings_selector::set_slm_mode(const slm_mode mode)
{
	for (auto& widgets : slm_widgets)
	{
		widgets->set_slm_mode(mode);
	}
}

void fixed_modulator_settings_selector::update_fixed_modulator_settings()
{
	const auto values = get_fixed_modulator_settings();
	emit fixed_modulator_settings_changed(values);
}

void fixed_modulator_settings_selector::set_fixed_modulator_settings_silent(const fixed_modulator_settings& modulator_settings)
{
	QSignalBlocker blk(*this);
	set_fixed_modulator_settings(modulator_settings);
}

void fixed_modulator_settings_selector::set_fixed_modulator_settings(const fixed_modulator_settings& modulator_settings)
{
	auto* layout = reinterpret_cast<QHBoxLayout*>(this->layout());//if empty do some half baked fixup?
	if (!layout)
	{
		setLayout(new QHBoxLayout);
		set_fixed_modulator_settings(modulator_settings);
		return;
	}
	const auto add_item = [&](const per_modulator_saveable_settings& settings, const int idx)
	{
		auto* per_modulator = new per_modulator_saveable_settings_selector;
		per_modulator->set_slm_id(idx);
		per_modulator->set_per_modulator_saveable_settings(settings);
		QObject::connect(per_modulator, &per_modulator_saveable_settings_selector::per_modulator_saveable_settings_changed, this, &fixed_modulator_settings_selector::update_fixed_modulator_settings);
		QObject::connect(per_modulator, &per_modulator_saveable_settings_selector::clicked_pattern, this, &fixed_modulator_settings_selector::clicked_pattern);
		slm_widgets.push_back(per_modulator);
		layout->addWidget(per_modulator);
	};
	for (auto idx = 0; idx < modulator_settings.size(); ++idx)
	{
		const auto& settings = modulator_settings.at(idx);
		if (idx < slm_widgets.size())
		{
			slm_widgets.at(idx)->set_per_modulator_saveable_settings(settings);
		}
		else
		{
			add_item(settings, idx);
		}
	}
	while (slm_widgets.size() > modulator_settings.size())
	{
		auto* widget = slm_widgets.back();
		slm_widgets.pop_back();
		layout->removeWidget(widget);
		widget->setParent(nullptr);
		delete widget;
	}
	emit fixed_modulator_settings_changed(modulator_settings);
	//
#if _DEBUG
	{
		const auto what_we_got = this->get_fixed_modulator_settings();
		const auto predicate = [](const per_modulator_saveable_settings& a, const per_modulator_saveable_settings& b)
		{
			return a.item_approx_equals(b);
		};
		const auto matching_settings = std::equal(what_we_got.begin(), what_we_got.end(), modulator_settings.begin(), modulator_settings.end(), predicate);
		if (!matching_settings)
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void fixed_modulator_settings_selector::enable_buttons(bool enable) const
{
	qli_not_implemented();
}
