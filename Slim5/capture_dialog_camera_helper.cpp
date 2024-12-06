#include "stdafx.h"
#include "capture_dialog_camera_helper.h"
#include <vector>
#include <QComboBox>
#include <QStandardItemModel> 
#include <QTableView>
#include <QHeaderView>
#include "camera_device.h"
#include "capture_modes.h"

void camera_settings_to_combobox(QComboBox* combobox, const std::vector<camera_device*>& cameras)
{
	for (auto mode : { capture_mode::sync_capture_sync_io, capture_mode::sync_capture_async_io , capture_mode::async_capture_async_io, capture_mode::burst_capture_async_io })
	{
		const auto& mode_name = capture_mode_settings::info.at(mode);
		combobox->addItem(QString::fromStdString(mode_name.name), QVariant::fromValue(mode));
	}
	//assume both cameras have same functionality for now!!!
	std::set<capture_mode> grayed_out_modes;
	const auto has_an_async = std::all_of(cameras.begin(), cameras.end(), [](camera_device* device)
	{
		return device->has_async_mode;
	});
	if (!has_an_async)
	{
		grayed_out_modes.insert(capture_mode::async_capture_async_io);
	}
	const auto has_a_burst = std::all_of(cameras.begin(), cameras.end(), [](camera_device* device) {return device->has_burst_mode; });
	if (!has_a_burst)
	{
		grayed_out_modes.insert(capture_mode::burst_capture_async_io);
	}
	//
	{
		const auto* model = qobject_cast<const QStandardItemModel*>(combobox->model());
		for (auto idx = 0; idx < model->rowCount(); ++idx)
		{
			auto* item = model->item(idx);
			auto data = item->data().value<capture_mode>();
			const auto disable = grayed_out_modes.find(data) != grayed_out_modes.end();
			{
				item->setFlags(disable ? item->flags() & ~(Qt::ItemIsSelectable | Qt::ItemIsEnabled)
					: Qt::ItemIsSelectable | Qt::ItemIsEnabled);
				// visually disable by graying out - works only if combobox has been painted already and palette returns the wanted color
				item->setData(disable ? combobox->palette().color(QPalette::Disabled, QPalette::Text)
					: QVariant(), // clear item data in order to use default color
					Qt::ForegroundRole);
			}
		}

	}
}

void style_table_view(QTableView* view)
{
	view->setSelectionBehavior(QAbstractItemView::SelectRows);
	view->setSelectionMode(QAbstractItemView::SingleSelection);
	view->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
}