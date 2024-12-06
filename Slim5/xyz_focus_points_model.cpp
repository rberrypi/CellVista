#include "stdafx.h"
#include "xyz_focus_points_model.h"
#include "limited_spinning_box_delegate.h"
#include "camera_device.h"
#include "approx_equals.h"
#include "roi_item.h"
#include "instrument_configuration.h"
#include "qli_runtime_error.h"

xyz_focus_points_model::xyz_focus_points_model(roi_item* scene, QObject* parent) : rectangle_model(parent), default_width_(50), default_height_(50), roi_ptr(scene)
{
	//is_model_selected_ = false;
	xyz_focus_points_model::insertRows(0, min_rows);
}

xyz_focus_points_model::~xyz_focus_points_model()
{
	for (auto& focus_point : data_view_)
	{
		delete focus_point;
	}
}

int xyz_focus_points_model::rowCount(const QModelIndex&) const
{
	return data_view_.size();
}

int xyz_focus_points_model::columnCount(const QModelIndex&) const
{
	return XYZ_MODEL_ITEM_VALID_IDX + 1;
}

QVariant xyz_focus_points_model::data(const QModelIndex& index, const int role) const
{
	if (index.isValid() && (role == Qt::DisplayRole || role == Qt::EditRole))
	{
		const auto& item = data_view_.at(index.row());
		switch (index.column())
		{
		case XYZ_MODEL_ITEM_ID_IDX:
			return item->id();
		case XYZ_MODEL_ITEM_X_IDX:
			return item->x_center();
		case XYZ_MODEL_ITEM_Y_IDX:
			return item->y_center();
		case XYZ_MODEL_ITEM_Z_IDX:
			return item->get_z();
		case XYZ_MODEL_ITEM_VALID_IDX:
			return item->get_verified();
		default:
			qli_invalid_arguments();
		}
	}
	if (role == Qt::TextAlignmentRole)
	{
		return Qt::AlignCenter;
	}
	return QVariant();
}

bool xyz_focus_points_model::setData(const QModelIndex& index, const QVariant& value, const int role)
{
	// Qt::UserRole is used to change the read only parameters
	if ((role == Qt::EditRole || role == Qt::UserRole) && index.isValid())
	{
		const auto row = index.row();
		const auto col = index.column();
		const auto special_bottom_rows = (role == Qt::EditRole) && (row < min_rows) && ((col == XYZ_MODEL_ITEM_X_IDX) || (col == XYZ_MODEL_ITEM_Y_IDX));
		if (special_bottom_rows)
		{
			return false;
		}
		auto& item = data_view_.at(index.row());
		auto has_alias = false;
		switch (index.column())
		{
		case XYZ_MODEL_ITEM_ID_IDX:
			item->set_id(value.toInt());
			break;
		case XYZ_MODEL_ITEM_X_IDX:
			has_alias = item->set_x_center(value.toFloat());
			setData(this->index(row, XYZ_MODEL_ITEM_VALID_IDX), false);
			break;
		case XYZ_MODEL_ITEM_Y_IDX:
			has_alias = item->set_y_center(value.toFloat());
			setData(this->index(row, XYZ_MODEL_ITEM_VALID_IDX), false);
			break;
		case XYZ_MODEL_ITEM_Z_IDX:
			has_alias = item->set_z_center(value.toFloat());
			break;
		case XYZ_MODEL_ITEM_VALID_IDX:
			item->set_verified(value.toBool());
			break;
		default:
			qli_invalid_arguments();
		}
		if (has_alias)
		{
			const auto start = createIndex(0, XYZ_MODEL_ITEM_Z_IDX);
			const auto stop = createIndex(rowCount(), XYZ_MODEL_ITEM_Z_IDX);
			emit dataChanged(start, stop);
		}
		emit dataChanged(index, index);

		set_visible();  //Verify Visibility of focus points every time data is changed 
		check_selection_status();	//Check if this model should be selected or not 

		if (roi_item::do_interpolation())
		{
			roi_ptr->update();
		}
	}
	return true;
}

Qt::ItemFlags xyz_focus_points_model::flags(const QModelIndex& /*index*/) const
{
	//actually special item shouldn't be selectable ?
	return Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsEnabled;
}

bool xyz_focus_points_model::insertRows(const int row_position, const int rows, const QModelIndex& index)
{
	Q_UNUSED(index);
	beginInsertRows(QModelIndex(), row_position, row_position + rows - 1);
	for (auto idx = row_position; idx < row_position + rows; ++idx)
	{
		auto data = scope_location_xyz(0, 0, 0);
		const auto size = QRectF(0, 0, default_width_, default_height_);
		const auto item = new xyz_focus_point_item(&roi_ptr->triangulator, idx, data, size);

		//item->set_selected(is_model_selected_);
		item->set_selected(roi_ptr->grid_selected_);

		data_view_.insert(idx, item);
		item->setParentItem(roi_ptr);
	}
	set_visible();
	reindex_data_view(row_position + rows);
	emit endInsertRows();
	return true;
}

bool xyz_focus_points_model::insertColumns(int, int, const QModelIndex&)
{
	return false;
}

bool xyz_focus_points_model::removeRows(const int row_position, const int rows, const QModelIndex& index)
{
	Q_UNUSED(index);
	if ((row_position + rows - 1) < min_rows)
	{
		return false;
	}
	const auto first = row_position - (rows - 1);
	beginRemoveRows(QModelIndex(), first, row_position);
	for (auto idx = first; idx <= row_position; ++idx)
	{
		const auto item_ptr = data_view_.at(idx);
		item_ptr->setParentItem(nullptr);
		delete item_ptr;
	}
	data_view_.remove(first, rows);
	set_visible();

	reindex_data_view(row_position + rows - 1);
	emit dataChanged(index, index);
	emit endRemoveRows();
	return true;
}

QVariant xyz_focus_points_model::headerData(const int section, const Qt::Orientation orientation, const int role) const
{
	if (role != Qt::DisplayRole)
		return QVariant();

	if (orientation == Qt::Horizontal)
	{
		switch (section) {
		case XYZ_MODEL_ITEM_ID_IDX:
			return tr("ID");
		case XYZ_MODEL_ITEM_X_IDX:
			return tr("X");
		case XYZ_MODEL_ITEM_Y_IDX:
			return tr("Y");
		case XYZ_MODEL_ITEM_Z_IDX:
			return tr("Z");
		case XYZ_MODEL_ITEM_VALID_IDX:
			return tr("Set?");
		default:
			return QVariant();
		}
	}
	if (orientation == Qt::Vertical)
	{
		return QString::number(section);
	}
	return QVariant();
}

QStyledItemDelegate* xyz_focus_points_model::get_column_delegate(const int col, QWidget* parent)
{
	switch (col)
	{
	case XYZ_MODEL_ITEM_X_IDX: return new limited_spinning_box_delegate(parent);
	case XYZ_MODEL_ITEM_Y_IDX: return new limited_spinning_box_delegate(parent);
	case XYZ_MODEL_ITEM_Z_IDX: return new limited_spinning_box_delegate(parent);
	default: return nullptr;
	}
}

int xyz_focus_points_model::find_unset_row(const int start_row_idx) const
{
	const auto predicate = [](const xyz_focus_point_item* item) {return item->get_verified(); };

	const auto first_item = std::find_if_not(data_view_.begin() + start_row_idx, data_view_.end(), predicate);
	const auto second_item = std::find_if_not(data_view_.begin(), data_view_.begin() + start_row_idx, predicate);

	if (first_item == data_view_.end())
	{
		if (second_item == data_view_.begin() + start_row_idx)
		{
			return -1;
		}
		return (*second_item)->id();
	}
	return (*first_item)->id();
}

scope_location_xyz xyz_focus_points_model::get_focus_point_location(const int row) const
{
	return data_view_.at(row)->get_serializable();
}

void xyz_focus_points_model::set_serializable_focus_points(const std::vector<scope_location_xyz>& points)
{
	resize_to(points.size());
	for (uint idx = 0; idx < points.size(); ++idx)		//before was row_count() instead of points.size()
	{
		const auto& value = points.at(idx);
		set_serializable_point(value, idx);
	}

#if _DEBUG
	{
		const auto current_points = get_serializable_focus_points();
		for (uint i = 0; i < points.size(); ++i)
		{
			const auto& current_point = current_points.at(i);
			const auto& point = points.at(i);
			if (!approx_equals(current_point.x, point.x) || !approx_equals(current_point.y, point.y) || !approx_equals(current_point.z, point.z))
			{
				qli_runtime_error();

			}
		}
	}
#endif
}


void xyz_focus_points_model::set_serializable_point(const scope_location_xyz& value, const int row)
{
	setData(createIndex(row, XYZ_MODEL_ITEM_X_IDX), value.x, Qt::UserRole);
	setData(createIndex(row, XYZ_MODEL_ITEM_Y_IDX), value.y, Qt::UserRole);
	setData(createIndex(row, XYZ_MODEL_ITEM_Z_IDX), value.z, Qt::UserRole);
}

std::vector<scope_location_xyz> xyz_focus_points_model::get_serializable_focus_points() const
{
	std::vector<scope_location_xyz> items;
	for (const auto& item : data_view_)
	{
		auto data = item->get_serializable();
		items.push_back(data);
	}
	return items;
}

void xyz_focus_points_model::reindex_data_view(const int start_idx)
{
	const auto items = data_view_.size();
	for (auto idx = start_idx; idx < items; ++idx)
	{
		setData(createIndex(idx, XYZ_MODEL_ITEM_ID_IDX), idx, Qt::UserRole);
	}
}

void xyz_focus_points_model::update_four_points(const grid_steps& steps, const scope_location_xy& center)
{
	const auto left_x = center.x - (steps.x_steps * steps.x_step / 2);
	const auto top_y = center.y - (steps.y_steps * steps.y_step / 2);
	const auto right_x = center.x + (steps.x_steps * steps.x_step / 2);
	const auto bot_y = center.y + (steps.y_steps * steps.y_step / 2);
	setData(createIndex(0, XYZ_MODEL_ITEM_X_IDX), left_x, Qt::UserRole);
	setData(createIndex(0, XYZ_MODEL_ITEM_Y_IDX), top_y, Qt::UserRole);
	setData(createIndex(1, XYZ_MODEL_ITEM_X_IDX), right_x, Qt::UserRole);
	setData(createIndex(1, XYZ_MODEL_ITEM_Y_IDX), top_y, Qt::UserRole);
	setData(createIndex(2, XYZ_MODEL_ITEM_X_IDX), right_x, Qt::UserRole);
	setData(createIndex(2, XYZ_MODEL_ITEM_Y_IDX), bot_y, Qt::UserRole);
	setData(createIndex(3, XYZ_MODEL_ITEM_X_IDX), left_x, Qt::UserRole);
	setData(createIndex(3, XYZ_MODEL_ITEM_Y_IDX), bot_y, Qt::UserRole);
	//
	const auto new_width = steps.x_step / 2;
	const auto new_height = steps.y_step / 2;
	if (default_width_ != new_width && default_height_ != new_height)
	{
		default_width_ = new_width;
		default_height_ = new_height;
		const auto dim = qMax(default_width_, default_height_);
		for (auto& item : data_view_)
		{
			auto rect = item->rect();
			auto old_center = rect.center();
			rect.setWidth(dim);
			rect.setHeight(dim);
			rect.moveCenter(old_center);
			item->setRect(rect);
		}
	}
}

void xyz_focus_points_model::set_visible()
{
	const auto check_z = [&]
	{
		const auto check = data_view_.at(0)->get_z();
		for (auto& focus_point : data_view_)
		{
			if (check != focus_point->get_z())
			{
				return true;
			}
		}
		return false;
	}();

	const auto visible = check_z || rowCount() > min_rows;
	for (auto& item : data_view_)
	{
		item->setVisible(visible);
	}
}

void xyz_focus_points_model::set_selection_color(const bool selected)
{
	//is_model_selected_ = selected;
	for (auto& item : data_view_)
	{
		item->set_selected(selected);
	}
}

void xyz_focus_points_model::check_selection_status()
{
	auto selected = false;
	if (roi_ptr)
	{
		selected = roi_ptr->grid_selected_;
	}
	set_selection_color(selected);
}
