#include "stdafx.h"
#include "roi_model.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "boost_cerealization.h"
#include <cereal/types/chrono.hpp>
#include <cereal/types/vector.hpp>
#include "qli_runtime_error.h"
#include <cereal/archives/json.hpp>
template <class Archive>
void serialize(Archive& archive, scope_location_xyz& cc)
{
	archive(
		cereal::make_nvp("x", cc.x),
		cereal::make_nvp("y", cc.y),
		cereal::make_nvp("z", cc.z)
	);
}

template <class Archive>
void serialize(Archive& archive, scope_location_xy& cc)
{
	archive(
		cereal::make_nvp("x", cc.x),
		cereal::make_nvp("y", cc.y)
	);
}

template <class Archive>
void serialize(Archive& archive, roi_item_dimensions& cc)
{

	archive(
		cereal::make_nvp("columns", cc.columns),
		cereal::make_nvp("rows", cc.rows),
		cereal::make_nvp("pages", cc.pages),
		cereal::make_nvp("column_step", cc.column_step),
		cereal::make_nvp("row_step", cc.row_step),
		cereal::make_nvp("page_step", cc.page_step)
	);
}

template <class Archive>
void serialize(Archive& archive, roi_item_shared& cc)
{
	archive(
		cereal::make_nvp("channels", cc.channels),
		cereal::make_nvp("repeats", cc.repeats),
		cereal::make_nvp("sets_bg", cc.sets_bg),
		cereal::make_nvp("io_sync_point", cc.io_sync_point),
		cereal::make_nvp("grid_selected", cc.grid_selected_)
	);
}


template <class Archive>
void serialize(Archive& archive, roi_item_serializable& cc)
{
	
	archive(
		cereal::make_nvp("roi_item_shared", cereal::base_class<roi_item_shared>(&cc)),
		cereal::make_nvp("scope_location_xy", cereal::base_class<scope_location_xy>(&cc)),
		cereal::make_nvp("roi_item_dimensions", cereal::base_class<roi_item_dimensions>(&cc)),
		cereal::make_nvp("focus_points", cc.focus_points)
	);
	
}

const static auto roi_points_name = "rois";
void roi_model::save_xml(cereal::JSONOutputArchive& archive) const
{
	std::vector<roi_item_serializable> row_items;
	for (auto idx = 0; idx < rowCount(); ++idx)
	{
		auto item = get_serializable_item(idx);
		row_items.push_back(item);
	}
	archive(cereal::make_nvp(roi_points_name, row_items));
}

void roi_model::load_xml(cereal::JSONInputArchive& archive)
{

	std::vector<roi_item_serializable> items;
	archive(cereal::make_nvp(roi_points_name, items));
	resize_to(items.size());
	for (auto idx = 0; idx < rowCount(); ++idx)
	{
		const auto& value = items.at(idx);
		set_serializable_item(value, idx);
	}
#if _DEBUG
	{
		for (uint i = 0; i < rowCount(); ++i) {
			const auto current_item = get_serializable_item(i);
			const auto& new_item = items[i];
			if(!current_item.is_load_valid(new_item))
			{
				qli_runtime_error();
			}
		}

	}
#endif
}
