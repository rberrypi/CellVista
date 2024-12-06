#include "stdafx.h"
#include "compact_light_path_holder.h"
#include "compact_light_path_selector.h"
#include <QVBoxLayout>

#include "qli_runtime_error.h"

compact_light_path_holder::compact_light_path_holder(QWidget* parent) :QWidget{ parent }
{
}

void compact_light_path_holder::reindex_widgets()
{
	for (auto path_idx = 0; path_idx < light_path_widgets.size(); ++path_idx)
	{
		auto& widget = light_path_widgets.at(path_idx);
		widget->set_id(path_idx);
	}
}

void compact_light_path_holder::add_one()
{
	auto current = this->get_compact_light_paths();
	current.push_back(current.back());
	set_light_paths(current);
	emit channel_added(1);
}

void compact_light_path_holder::add_channel(const compact_light_path& new_light_path)
{
	auto current = this->get_compact_light_paths();
	current.push_back(new_light_path);
	set_light_paths(current);
}

void compact_light_path_holder::remove_one(const int idx)
{
	auto* layout = reinterpret_cast<QVBoxLayout*>(this->layout());
	auto* widget = light_path_widgets.at(idx);
	light_path_widgets.erase(light_path_widgets.begin() + idx);
	widget->setHidden(true);
	layout->removeWidget(widget);
	widget->setParent(nullptr);
	delete widget;
	reindex_widgets();
	update_values();
	emit channel_removed(idx);
}

void compact_light_path_holder::set_light_paths(const std::vector<compact_light_path>& light_paths)
{
	auto* layout = reinterpret_cast<QVBoxLayout*>(this->layout());//if empty do some half baked fixup?
	if (!layout)
	{
		setLayout(new QVBoxLayout);
		set_light_paths(light_paths);
		return;
	}
	const auto add_item = [&](const compact_light_path& settings, const int idx)
	{
		auto* light_path = new compact_light_path_selector;
		light_path->set_compact_light_path(settings);
		light_path->set_id(idx);
		QObject::connect(light_path, &compact_light_path_selector::remove_me, this, &compact_light_path_holder::remove_one);
		QObject::connect(light_path, &compact_light_path_selector::compact_light_path_changed, this, &compact_light_path_holder::update_values);
		QObject::connect(light_path, &compact_light_path_selector::camera_config_changed, this, &compact_light_path_holder::update_values);
		light_path_widgets.push_back(light_path);
		layout->addWidget(light_path);
	};
	for (auto idx = 0; idx < light_paths.size(); ++idx)
	{
		const auto& settings = light_paths.at(idx);
		if (idx < light_path_widgets.size())
		{
			auto* light_path = light_path_widgets.at(idx);
			QSignalBlocker blk(light_path);
			light_path->set_compact_light_path(settings);
		}
		else
		{
			add_item(settings, idx);
		}
	}
	while (light_path_widgets.size() > light_paths.size())
	{
		auto* widget = light_path_widgets.back();
		light_path_widgets.pop_back();
		layout->removeWidget(widget);
		widget->setParent(nullptr);
		delete widget;
	}
	reindex_widgets();
	emit value_changed(light_paths);
	//
#if _DEBUG
	{
		const auto right_number_of_paths = light_path_widgets.size() == light_paths.size();
		if (!(right_number_of_paths))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void compact_light_path_holder::update_values() const
{
	const auto values = get_compact_light_paths();
	emit value_changed(values);
}

std::vector<compact_light_path> compact_light_path_holder::get_compact_light_paths() const
{
	std::vector<compact_light_path> return_me;
	for (auto&& paths : light_path_widgets)
	{
		const auto light_path = paths->get_compact_light_path();
#if _DEBUG
		if (!light_path.is_valid())
		{
			qli_runtime_error("invalid quad");
		}
#endif
		return_me.push_back(light_path);
	}
	return return_me;
}

camera_config compact_light_path_holder::get_default_camera_config() const
{
	const auto has_something_loaded = !light_path_widgets.empty();
	if (has_something_loaded)
	{
		return static_cast<const camera_config&>(light_path_widgets.front()->get_compact_light_path());
	}
	return camera_config();
}


int compact_light_path_holder::get_number_channels() const
{
	return light_path_widgets.size();
}