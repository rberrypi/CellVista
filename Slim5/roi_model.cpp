#include "stdafx.h"
#include "full_interface_gui.h"
#include "limited_spinning_box_delegate.h"
#include "roi_item.h"
#include "device_factory.h"
#include "scope.h"
#include "channel_editor_delegate.h"
#include <boost/range/algorithm/find.hpp>
#include "xyz_focus_points_model.h"
#include "scope_delay_delegate.h"
#include "qli_runtime_error.h"

const auto plus_minus = QChar(0x00B1);
const auto mu = QChar(0x03BC);

roi_item_meta_data roi_model_info[ITEM_VERIFIED_IDX + 1] =
{
	{ "ID","ID",false, ITEM_ID_IDX },
	{ QString("X\n[%1m]").arg(mu),"X at ROI center",false, ITEM_X_IDX },
	{ QString("Y\n[%1m]").arg(mu),"Y at ROI center",false, ITEM_Y_IDX },
	{ "Cols\n[#]","# of columns",true, ITEM_COLS_IDX },
	{ "Rows\n[#]","# of rows",true, ITEM_ROWS_IDX },
	{ QString("Steps\n[").append(plus_minus).append("#]"),"# of z-slices, above and below",true, ITEM_PAGES_IDX },
	{ QString("Col Step\n[%1m]").arg(mu),"Column step size",false, ITEM_STEP_COL_IDX },
	{ QString("Row Step\n[%1m]").arg(mu),"Row step size",false, ITEM_STEP_ROW_IDX },
	{ QString("Slice Step\n[%1m]").arg(mu),"Slice steps",false, ITEM_STEP_PAGE_IDX },
	{ "ROI Delay\n[ms]","Delays before each XY motion, and the delay after processing an ROI [ms]",false, ITEM_DELAYS_IDX },
	{ "Channels\n[#]","Channels, comma separated",true, ITEM_CHANNEL_IDX },
	{ "Repeats\n[#]", "# of times to repeat step", true, ITEM_REPEATS_IDX },
	{ "Sets BG", "ROI sets a background that is later subtracted",true, ITEM_SETS_BG_IDX },
	{ "Sync IO", "Waits for files to be written before continuing",false, ITEM_SYNC_IDX },
	{ "Verified","Verify points before launching an experiment (recommended)",false, ITEM_VERIFIED_IDX },
};

roi_model::roi_model(QGraphicsScene* in, const get_default_serializable_item_functor& get_channel_info, QObject* parent) :rectangle_model(parent), scene_(in), get_default_serializable_item_(get_channel_info)
{

#if _DEBUG
	if (!scene_)
	{
		qli_invalid_arguments();
	}
#endif
}

int roi_model::rowCount(const QModelIndex&) const
{
	return data_view_.size();
}

int roi_model::columnCount(const QModelIndex&) const
{
	return ITEM_VERIFIED_IDX + 1;
}

QVariant roi_model::data(const QModelIndex& index, const int role) const
{
	if (index.isValid() && (role == Qt::DisplayRole || role == Qt::EditRole))
	{
		const auto row = index.row();
		const auto col = index.column();
		const auto& item = data_view_.at(row);
		switch (col)
		{
		case ITEM_ID_IDX: return QVariant(static_cast<unsigned int>(item->get_id()));
		case ITEM_X_IDX: return QVariant(item->get_x());
		case ITEM_Y_IDX: return QVariant(item->get_y());
		case ITEM_COLS_IDX: return QVariant(static_cast<unsigned int>(item->get_columns()));
		case ITEM_ROWS_IDX: return QVariant(static_cast<unsigned int>(item->get_rows()));
		case ITEM_PAGES_IDX: return QVariant(static_cast<unsigned int>(item->get_pages()));
		case ITEM_STEP_COL_IDX: return QVariant(item->get_column_step());
		case ITEM_STEP_ROW_IDX: return QVariant(item->get_row_step());
		case ITEM_STEP_PAGE_IDX: return QVariant(item->get_page_step());
		case ITEM_DELAYS_IDX: return QVariant::fromValue(static_cast<scope_delays>(*item));
		case ITEM_CHANNEL_IDX: return QVariant::fromValue(item->channels);
		case ITEM_REPEATS_IDX: return QVariant::fromValue(static_cast<unsigned int>(item->repeats));
		case ITEM_SETS_BG_IDX: return QVariant::fromValue(item->sets_bg);
		case ITEM_SYNC_IDX: return QVariant::fromValue(item->io_sync_point);
		case ITEM_VERIFIED_IDX: return QVariant::fromValue(item->focus_points_verified());
		default:
			qli_runtime_error();
		}
	}
	if (index.isValid() && role == Qt::ToolTipRole)
	{
		const auto col_idx = index.column();
		auto tooltip = roi_model_info[col_idx].tooltip;
		return tooltip;
	}
	if (role == Qt::TextAlignmentRole)
	{
		return Qt::AlignCenter;
	}
	return QVariant();
}

QStyledItemDelegate* roi_model::get_column_delegate(const int col, QWidget* parent)
{
	switch (col)
	{
	case ITEM_X_IDX: return new limited_spinning_box_delegate(parent);
	case ITEM_Y_IDX: return new limited_spinning_box_delegate(parent);
	case ITEM_STEP_COL_IDX: return new limited_spinning_box_delegate(parent);
	case ITEM_STEP_ROW_IDX: return new limited_spinning_box_delegate(parent);
	case ITEM_STEP_PAGE_IDX: return new limited_spinning_box_delegate(parent);
	case ITEM_DELAYS_IDX: return new scope_delay_delegate(parent);
	case ITEM_CHANNEL_IDX: return new channel_editor_delegate(parent);
	default: return nullptr;
	}
}

bool roi_model::setData(const QModelIndex& index, const QVariant& value, const int role)
{
	if ((role == Qt::EditRole || role == Qt::UserRole) && index.isValid())
	{
		const auto row = index.row();
		const auto col = index.column();
		auto& item = data_view_.at(row);
		const auto filter_negatives = [](const float value, const float epsilon = 0.001f) {return std::max(epsilon, value); };
		switch (col)
		{
		case ITEM_ID_IDX: item->set_id(value.value<unsigned int>()); break;
		case ITEM_X_IDX: item->set_x(value.value<float>()); break;
		case ITEM_Y_IDX: item->set_y(value.value<float>()); break;
		case ITEM_COLS_IDX: item->set_columns(value.value<unsigned int>()); break;
		case ITEM_ROWS_IDX: item->set_rows(value.value<unsigned int>()); break;
		case ITEM_PAGES_IDX: item->set_pages(value.value<unsigned int>()); break;
		case ITEM_STEP_COL_IDX: item->set_column_step(filter_negatives(value.value<float>())); break;
		case ITEM_STEP_ROW_IDX: item->set_row_step(filter_negatives(value.value<float>())); break;
		case ITEM_STEP_PAGE_IDX: item->set_page_step(filter_negatives(value.value<float>())); break;
		case ITEM_DELAYS_IDX: item->set_delays(value.value<scope_delays>()); break;
		case ITEM_CHANNEL_IDX:
		{
			const auto channels = value.value<fl_channel_index_list>();
			if (channel_validity_filter)
			{
				for (auto channel_index : channels)
				{
					const auto bad = !channel_validity_filter(channel_index);
					if (bad)
					{
						return false;
					}
				}
			}
			item->channels = channels;
			break;
		}
		case ITEM_REPEATS_IDX: item->repeats = filter_negatives(value.value<unsigned int>(), 1); break;
		case ITEM_SETS_BG_IDX: item->sets_bg = (value.value<bool>()); break;
		case ITEM_SYNC_IDX: item->io_sync_point = (value.value<bool>()); break;
		case ITEM_VERIFIED_IDX: item->verify_focus_points(value.value<bool>()); break;
		default:
			qli_runtime_error();
		}
		emit dataChanged(index, index);
		const auto has_sized_changed = roi_model_info[col].updates_size;
		if (has_sized_changed)
		{
			emit update_capture_info();
		}
		if (col == ITEM_CHANNEL_IDX)
		{
			emit updated_channel_info(row);
		}
		if (col == ITEM_COLS_IDX | col == ITEM_ROWS_IDX | col == ITEM_STEP_COL_IDX | col == ITEM_STEP_ROW_IDX)
		{
			emit row_col_changed();		//only centers the roi changed at the moment 
		}
		return true;
	}
	return false;
}

Qt::ItemFlags roi_model::flags(const QModelIndex& /*index*/) const
{
	return Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsEnabled;
}

bool roi_model::insertRows(const int row_position, const int rows, const QModelIndex& index)
{
	Q_UNUSED(index);
	beginInsertRows(QModelIndex(), row_position, row_position + rows - 1);
	auto data = get_default_serializable_item_();

	for (auto idx = row_position; idx < row_position + rows; ++idx)
	{
		const auto item = new roi_item(idx, data);
		const auto begin = data_view_.begin();
		data_view_.insert(begin + idx, item);
		scene_->addItem(item);
	}
	reindex_data_view(row_position + rows);
	emit setup_xyz(row_position);
	emit update_capture_info();

	emit endInsertRows();
	return true;
}

bool roi_model::insertColumns(int, int, const QModelIndex&)
{
	return false;
}

bool roi_model::removeRows(const int row_position, const int rows, const QModelIndex& index)
{
	Q_UNUSED(index);
	const auto first = row_position - (rows - 1);
	beginRemoveRows(QModelIndex(), first, row_position);
	for (auto idx = first; idx <= row_position; ++idx)
	{
		const auto& item_ptr = data_view_.at(idx);
		delete item_ptr;
	}
	const auto begin = data_view_.begin();
	data_view_.erase(begin + first, begin + first + rows);
	reindex_data_view(row_position + rows - 1);
	endRemoveRows();
	emit update_capture_info();
	return true;
}

QVariant roi_model::headerData(const int section, const Qt::Orientation orientation, const int role) const
{
	if (role == Qt::DisplayRole)
	{
		if (orientation == Qt::Horizontal)
		{
			auto label = roi_model_info[section].label;
			return label;
		}
		if (orientation == Qt::Vertical)
		{
			return QString::number(section);
		}
	}
	return QVariant();
}

void roi_model::reindex_data_view(const int start_idx)
{
	const auto items = data_view_.size();
	for (auto idx = start_idx; idx < items; ++idx)
	{
		setData(createIndex(idx, ITEM_ID_IDX), idx, Qt::UserRole);
	}
}

void roi_model::set_serializable_item(const roi_item_serializable& value, const int row)
{
	data_view_.at(row)->set_roi_item_serializable(value);
	const auto top_left = createIndex(row, 0);
	const auto top_right = createIndex(row, ITEM_VERIFIED_IDX);
	emit dataChanged(top_left, top_right);
}

roi_item_serializable roi_model::get_serializable_item(const int row) const
{
	return data_view_.at(row)->get_roi_item_serializable();
}

QRectF roi_model::get_bounding_rectangle() const
{
	auto right = std::numeric_limits<qreal>::lowest();
	auto left = std::numeric_limits<qreal>::max();
	auto top = std::numeric_limits<qreal>::lowest();
	auto bot = std::numeric_limits<qreal>::max();

	for (auto& item : data_view_)
	{
		auto rect = item->boundingRect();
		left = std::min(left, rect.left());
		right = std::max(left, rect.right());
		bot = std::min(bot, rect.bottom());
		top = std::max(top, rect.top());
	}
	return { left,top,right - left,top - bot };
}

int roi_model::find_unset_row(const int start_row_idx) const
{
	const auto predicate = [](const roi_item* item)
	{
		return item->focus_points_verified();
	};

	const auto first_item = std::find_if_not(data_view_.begin() + start_row_idx, data_view_.end(), predicate);
	const auto second_item = std::find_if_not(data_view_.begin(), data_view_.begin() + start_row_idx, predicate);

	if (first_item == data_view_.end())
	{
		if (second_item == data_view_.begin() + start_row_idx)
		{
			return -1;
		}
		return (*second_item)->get_id();
	}
	return (*first_item)->get_id();
}

std::vector<float> roi_model::get_z_for_whole_roi(const int roi_idx) const
{
	return data_view_.at(roi_idx)->get_z_all_focus_points();
}

void roi_model::increment_zee_for_whole_roi(const float value, const int idx) const
{
	data_view_.at(idx)->increment_all_focus_points(value);
}

void roi_model::set_zee_for_whole_roi(const float value, const int idx) const
{
	data_view_.at(idx)->set_all_focus_points(value);
}

void roi_model::set_zee_for_whole_roi(const std::vector<float>& z_values, const int idx) const
{
	data_view_.at(idx)->set_all_focus_points(z_values);
}

void roi_model::update_center(const int row, const scope_location_xy center)
{
	auto& item_ptr = data_view_.at(row);
	item_ptr->set_center(center);

	setData(createIndex(row, ITEM_X_IDX), center.x, Qt::UserRole);
	setData(createIndex(row, ITEM_Y_IDX), center.y, Qt::UserRole);
}

xy_pairs roi_model::get_xy_focus_points(const int idx) const
{
	return data_view_.at(idx)->get_xy_focus_points();
}

void roi_model::set_xy_focus_points(const int idx, const xy_pairs& xy_focus_points, float displacement_x, float displacement_y)
{
	data_view_.at(idx)->set_xy_focus_points(xy_focus_points, displacement_x, displacement_y);
}