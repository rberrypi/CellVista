#include "stdafx.h"
#include "full_interface_gui.h"
#include "device_factory.h"
#include "scope.h"
#include "click_surface.h"
#include "camera_device.h"
#include "xyz_focus_points_model.h"  
#include "tomogram_picker.h"
#include "compact_light_path_holder.h"
#include <QFileDialog>
#include <QStyledItemDelegate>
#include <QInputDialog>
#include "graphics_view_zoom.h"
#include "capture_dialog_camera_helper.h"
#include <QMessageBox>
#include "ui_full_interface_gui.h"
#include <fstream>
#include <QTimer>
#include <iostream>
#include "capture_modes.h"
#include "qli_runtime_error.h"
#include "safe_move_to_screen.h"


full_interface_gui::full_interface_gui(const live_gui_settings& live_gui_settings, const settings_file& settings_file, slim_four* slim_four_handle, QMainWindow* parent) : QMainWindow(parent), pointer_(nullptr), rois_(nullptr), trakem2_dialog_(nullptr), settings_file_(settings_file), slim_four_handle(slim_four_handle)
{
	setAttribute(Qt::WA_DeleteOnClose, true);
	ui_ = std::make_unique<Ui::full_interface_gui>();
	ui_->setupUi(this);
	scan_state_labels.front() = ui_->btnDoIt->text();
	setup_disk_size();
	setup_acquire_buttons();
	setup_file_grouping_modes();
	setup_graphics_surface();
	setup_rois_table();
	setup_buttons();				// connect buttons, set mem used (update _capture)
	setup_load_save_dialog();
	setup_navigation_buttons();
	setup_common_channels();
	setup_channels();				//connect add channel button in light path
	setup_trakem2();
	connect(rois_.get(), &roi_model::setup_xyz, this, &full_interface_gui::setup_xyz_model_item);
	connect(ui_->wdg_light_path_holder, &compact_light_path_holder::value_changed, this, &full_interface_gui::update_pointer_frame_size);
	auto* z = new graphics_view_zoom(ui_->gfxSurface);		//leaky?			
	z->set_modifiers(Qt::NoModifier);
	set_live_gui_settings(live_gui_settings);
	safe_move_to_screen(this, 1);
}

void full_interface_gui::setup_file_grouping_modes()
{
	for (const auto& item : filename_grouping_names)
	{
		const auto as_variant = QVariant::fromValue(item.first);
		ui_->cmb_file_grouping->addItem(QString::fromStdString(item.second), as_variant);
	}
}

void full_interface_gui::setup_graphics_surface()
{
	gfx_ = std::make_unique<QGraphicsScene>();		//managed by the view
	ui_->gfxSurface->setDragMode(QGraphicsView::ScrollHandDrag);
	ui_->gfxSurface->setScene(gfx_.get());
	ui_->gfxSurface->setInteractive(true);
	//ui_->gfxSurface->scale(0.2, 0.2);
	{
		const auto max_size = D->scope->xy_drive->get_stage_limits();
		if (max_size.valid)
		{
			ui_->gfxSurface->setSceneRect(max_size.xy);
		}
	}
	connect(ui_->btnCenterGrid, &QPushButton::clicked, this, &full_interface_gui::center_grid);
	roi_item::enable_interpolation<true>(false);		//temp fix to always start buttons as False
	roi_item::enable_interpolation<false>(false);		//TODO:: figure out whats really going on
	connect(ui_->btnUpdateInterp, &QPushButton::toggled, [&](const bool enable)
		{
			roi_item::enable_interpolation<true>(enable);
			if (enable)
			{
				ui_->btnUpdateInterpGlobal->setChecked(false);
			}
		});
	connect(ui_->btnUpdateInterpGlobal, &QPushButton::toggled, [&](const bool enable)
		{
			roi_item::enable_interpolation<false>(enable);
			if (enable)
			{
				ui_->btnUpdateInterp->setChecked(false);
			}
		});

	connect(ui_->gfxSurface, &click_surface::move_to, [&](const QPointF location, QGraphicsItem* item)
		{
			auto new_position = static_cast<scope_location_xyz>(D->scope->get_state(true));
			new_position.x = location.x();
			new_position.y = location.y();
			//if on a valid item then get z position
			if (item)
			{
				if (item->UserType == xyz_focus_point_item::UserType)
				{
					const auto* xyz_item = qgraphicsitem_cast<xyz_focus_point_item*>(item);
					if (xyz_item != nullptr)
					{
						new_position.z = xyz_item->get_z();
					}
				}
				if (item->UserType == roi_item::UserType)
				{
					const auto* roi_item_ptr = qgraphicsitem_cast<roi_item*>(item);
					if (roi_item_ptr != nullptr)
					{
						const auto loc = scope_location_xy(location.x(), location.y());
						const auto interpolated_z = roi_item_ptr->triangulator.interpolate_one(loc);
						new_position.z = interpolated_z;
						if (!std::isfinite(interpolated_z))
						{
							new_position.z = D->scope->get_state(true).z;
						}
					}
				}
			}
			D->scope->move_to(new_position, true);
		});

	connect(ui_->gfxSurface, &click_surface::clicked_graphics_item, [&](QGraphicsItem* item)
		{
			if (item->UserType == roi_item::UserType)
			{
				//some how this is required
				const auto* graphics_item = qgraphicsitem_cast<roi_item*>(item);
				if (graphics_item != nullptr)
				{
					const auto row = graphics_item->get_id();
					select_roi(row);
				}
			}

			if (item->UserType == xyz_focus_point_item::UserType)
			{
				//somehow this is required
				const auto* xyz_item = qgraphicsitem_cast<xyz_focus_point_item*>(item);
				if (xyz_item != nullptr)
				{
					const auto* roi_item_ptr = qgraphicsitem_cast<roi_item*> (xyz_item->parentItem());
					const auto roi_row = roi_item_ptr->get_id();
					const auto xyz_row = xyz_item->id();
					select_roi(roi_row);
					ui_->xyzList->selectRow(xyz_row);
				}
			}
		});
	connect(gfx_.get(), &QGraphicsScene::changed, this, [&] {  gfx_->update();  });

	auto* z = new graphics_view_zoom(ui_->gfxSurface);
	z->set_modifiers(Qt::NoModifier);

	pointer_ = new QGraphicsEllipseItem();
	const QColor blueish(255, 90, 0, 150);
	pointer_->setBrush(QBrush(blueish));
	pointer_->setZValue(std::numeric_limits<qreal>::max());
	gfx_->addItem(pointer_);
	update_pointer_frame_size();
	//ensures that pointer is 1/6 of the screen?
	ui_->gfxSurface->fitInView(pointer_, Qt::AspectRatioMode::KeepAspectRatioByExpanding);
}


void full_interface_gui::setup_buttons()
{


	//
	const auto add_item_functor = [&]()
	{
		const auto xyz_index = ui_->xyzList->currentIndex();
		const auto roi_row = get_roi_index();
		if (roi_row >= 0) {
			auto& xyz_model = rois_->data_view_.at(roi_row)->xyz_model;
			const auto new_row = std::max(xyz_index.isValid() ? 1 + xyz_index.row() : xyz_model.rowCount(), xyz_focus_points_model::min_rows);
			const auto position = static_cast<scope_location_xyz>(D->scope->get_state(true));  //TODO issue with this
			xyz_model.insertRow(new_row);
			xyz_model.set_serializable_point(position, new_row);
			ui_->xyzList->selectRow(new_row);
		}
	};
	connect(ui_->btn_add_focus_point, &QPushButton::clicked, add_item_functor);
	const auto remove_item_functor = [&]()
	{
		const auto xyz_index = ui_->xyzList->currentIndex();
		const auto roi_row = get_roi_index();
		if (xyz_index.isValid() && roi_row >= 0)
		{
			auto& xyz_model = rois_->data_view_.at(roi_row)->xyz_model;
			xyz_model.removeRows(xyz_index.row(), 1);
		}
	};
	connect(ui_->btn_remove_focus_point, &QPushButton::clicked, remove_item_functor);
	//
	const auto choose_next_focus_point_functor = [&]()
	{
		const auto index = ui_->xyzList->currentIndex();
		const auto roi_row = get_roi_index();
		if (index.isValid() && roi_row >= 0)
		{
			const auto xyz_row = index.row();
			set_focus();
			const auto& xyz_model = rois_->data_view_.at(roi_row)->xyz_model;
			const auto next = xyz_model.find_unset_row(xyz_row);
			if (next >= 0)
			{
				ui_->xyzList->selectRow(next);
			}
			else
			{

				const auto next_roi = rois_->find_unset_row(roi_row);
				if (next_roi >= 0)
				{
					select_roi(next_roi);
					goto_point();
					ui_->xyzList->selectRow(0);
				}
			}
		}
	};
	connect(ui_->btn_next_focus_point, &QPushButton::clicked, choose_next_focus_point_functor);
	connect(ui_->btnSetFocus, &QPushButton::clicked, this, &full_interface_gui::set_focus);
	connect(ui_->btnMakeHull, &QPushButton::clicked, this, &full_interface_gui::wrangle_convex_hull);
	connect(ui_->btnFillColumn, &QPushButton::clicked, this, &full_interface_gui::fill_column_from_selection);
	connect(ui_->btnPeakTop, &QPushButton::toggled, this, [&](const bool checked) {
		peak_top_bottom<true>(checked);
		if (checked)
		{
			ui_->btnPeakBottom->setChecked(false);
		}
		});
	connect(ui_->btnPeakBottom, &QPushButton::toggled, this, [&](const bool checked) {
		peak_top_bottom<false>(checked);
		if (checked)
		{
			ui_->btnPeakTop->setChecked(false);
		}
		});

	connect(ui_->iterationTimes, qOverload<int>(&QSpinBox::valueChanged), this, &full_interface_gui::update_capture_info);
	connect(ui_->btnInsertTomogram, &QPushButton::pressed, this, &full_interface_gui::insert_tomogram);
	connect(ui_->btnSixWellPlate, &QPushButton::clicked, this, &full_interface_gui::set_six_well_plate);

	connect(ui_->btnDoublePoints, &QPushButton::clicked, [&]
		{
			//so we're going to load and save the whole set
			bool okay = false;
			const auto replicates = QInputDialog::getInt(this, tr("Duplicate By Row"), tr("Number of copies"), 1, 1, 99, 1, &okay) + 1;
			if (okay)
			{
				std::vector<roi_item_serializable> items;
				for (auto idx = 0; idx < rois_->rowCount(); ++idx)
				{
					auto item = rois_->get_serializable_item(idx);
					items.push_back(item);
				}
				rois_->resize_to(items.size() * replicates);
				for (auto idx = 0; idx < items.size(); ++idx)
				{
					auto& item = items.at(idx);
					for (auto replicate = 0; replicate < replicates; ++replicate)
					{
						rois_->set_serializable_item(item, idx * replicates + replicate);
					}
				}
			}
		});
	connect(ui_->btnMergePoints, &QPushButton::clicked, [&]
		{
			const auto max_row_idx = rois_->rowCount() - 1;
			auto first_value_okay = false, second_value_okay = false;
			const auto current_idx = get_roi_index();
			const auto selected_row = current_idx >= 0 ? current_idx : std::max(max_row_idx - 2, 0);
			const auto row_from = QInputDialog::getInt(this, tr("Merge"), tr("Move this row"), selected_row, 0, max_row_idx, 1, &first_value_okay);
			const auto row_to = QInputDialog::getInt(this, tr("Merge"), tr("To this row"), max_row_idx, 0, max_row_idx, 1, &second_value_okay);
			if (first_value_okay && second_value_okay) {
				if (row_from != row_to)
				{
					//were going to remove both rows, and insert a new one into the place
					const auto b = rois_->get_serializable_item(row_from);
					auto a = rois_->get_serializable_item(row_to);
					roi_item_serializable::merge_b_into_a(a, b);
					rois_->set_serializable_item(a, row_to);
					rois_->removeRows(row_from, 1);
				}
			}
		});
	connect(ui_->btnNextROI, &QPushButton::clicked, this, &full_interface_gui::choose_next_roi);
	connect(ui_->btnRecalibrate, &QPushButton::clicked, [&]
		{
			const auto selected = get_roi_index();

			if (selected >= 0)
			{
				const auto here = static_cast<scope_location_xyz>(D->scope->get_state(true));
				const auto dest_row = selected;
				const auto dest_x = rois_->data(rois_->index(dest_row, ITEM_X_IDX), Qt::DisplayRole).value<float>();
				const auto dest_y = rois_->data(rois_->index(dest_row, ITEM_Y_IDX), Qt::DisplayRole).value<float>();
				const auto dest_z = get_z_for_whole_roi(dest_row);  //get all focus point z's
				const auto dest_z_ = rois_->data_view_.at(dest_row)->query_triangulator({ dest_x, dest_y });


				const auto displacement_x = dest_x - here.x;
				const auto displacement_y = dest_y - here.y;
				const auto displacement_z_ = std::isfinite(dest_z_) ? dest_z_ - here.z : here.z;

				for (auto idx = 0; idx < rois_->rowCount(); ++idx)
				{
					const auto local_dest_x = rois_->data(rois_->index(idx, ITEM_X_IDX), Qt::DisplayRole).value<float>();
					const auto local_dest_y = rois_->data(rois_->index(idx, ITEM_Y_IDX), Qt::DisplayRole).value<float>();
					auto local_dest_z = get_z_for_whole_roi(idx);

					auto xy_focus_points = rois_->get_xy_focus_points(idx);
					rois_->set_xy_focus_points(idx, xy_focus_points, displacement_x, displacement_y);

					rois_->setData(rois_->index(idx, ITEM_X_IDX), local_dest_x - displacement_x, Qt::EditRole);
					rois_->setData(rois_->index(idx, ITEM_Y_IDX), local_dest_y - displacement_y, Qt::EditRole);

					for (auto& z_value : local_dest_z)
					{
						z_value -= displacement_z_;
					}

					set_zee_for_whole_roi(local_dest_z, idx);
				}
			}
		});
	connect(ui_->btnUncheckAllROI, &QPushButton::clicked, [&]()
		{
			for (auto& roi : rois_->data_view_)
			{
				const auto roi_row = roi->get_id();
				un_verify_focus_points(roi_row);
				rois_->setData(rois_->index(roi_row, ITEM_VERIFIED_IDX), false);
			}
		});
	connect(ui_->btnUncheckAllXYZ, &QPushButton::clicked, [&]()
		{
			const auto roi_row = get_roi_index();
			un_verify_focus_points(roi_row);
		});


	connect(ui_->btnPlacePoints, &QPushButton::clicked, [&]
		{
			auto okay = false;
			const auto spacing = QInputDialog::getInt(this, tr("Automatic Focus Points Insert"), tr("Desired spacing between focus points:"), 1, 1, 99, 1, &okay);
			if (okay)
			{
				const auto roi_row = get_roi_index();
				if (roi_row >= 0)
				{
					const auto& item_ptr = rois_->data_view_.at(roi_row);
					auto& xyz_model = item_ptr->xyz_model;

					const auto steps = item_ptr->get_grid_steps();
					const auto center = item_ptr->get_grid_center();
					const auto left_x = center.x - (steps.x_steps * steps.x_step / 2);
					const auto top_y = center.y - (steps.y_steps * steps.y_step / 2);

					const auto cols = item_ptr->get_columns();
					const auto rows = item_ptr->get_rows();
					const auto col_step = item_ptr->get_column_step();
					const auto row_step = item_ptr->get_row_step();

					//Include everything except 4 corners
					for (auto i = 0; i <= rows; i = i + spacing) {
						for (auto j = 0; j <= cols; j = j + spacing)
						{

							if (!(i % rows) && !(j % cols))
							{
								continue;
							}
							const auto x_index = left_x + j * col_step;
							const auto y_index = top_y + i * row_step;

							//insert focus points
							auto interpolated_z = item_ptr->triangulator.interpolate_one(scope_location_xy(x_index, y_index));
							if (!std::isfinite(interpolated_z))
							{
								interpolated_z = D->scope->get_state(true).z;
							}
							const auto position = scope_location_xyz(x_index, y_index, interpolated_z);
							const auto xyz_row = xyz_model.rowCount();
							xyz_model.insertRow(xyz_row);
							xyz_model.set_serializable_point(position, xyz_row);
						}
					}
				}
			}

		});
}

bool full_interface_gui::is_valid_acquisition()
{
	const auto has_items = rois_->rowCount() > 0;
	const auto valid_acquisition = has_items;
	return valid_acquisition;
}

void full_interface_gui::setup_rois_table()
{
	get_default_serializable_item_functor functor = [&] {return get_default_item(); };
	rois_ = std::make_unique<roi_model>(gfx_.get(), functor);
	rois_->channel_validity_filter = [&](const int proposed_channel_index) {
		return ui_->wdg_light_path_holder->get_number_channels() > proposed_channel_index;
	};
	ui_->tblROI->setModel(rois_.get());
	for (auto i = 0; i < rois_->columnCount(); ++i)
	{
		auto* ptr = rois_->get_column_delegate(i, ui_->tblROI);
		if (ptr != nullptr)
		{
			ui_->tblROI->setItemDelegateForColumn(i, ptr);
		}
	}
	style_table_view(ui_->tblROI);
	ui_->tblROI->hideColumn(ITEM_ID_IDX);
	ui_->tblROI->setSelectionBehavior(QAbstractItemView::SelectionBehavior::SelectItems);
	connect(rois_.get(), &roi_model::updated_channel_info, this, &full_interface_gui::update_model_selection);
	connect(ui_->tblROI->selectionModel(), &QItemSelectionModel::currentRowChanged, [&](const QModelIndex& current, const QModelIndex& previous)
		{
			if (current.isValid())
			{
				if (previous.isValid())
				{
					const auto previous_row = previous.row();
					//rois_->data_view_.at(previous_row)->xyz_model.set_selection_color(false);
					rois_->data_view_.at(previous_row)->grid_selected(false);
				}
				const auto current_row = current.row();
				//rois_->data_view_.at(current_row)->xyz_model.set_selection_color(true);
				rois_->data_view_.at(current_row)->grid_selected(true);

				update_model_selection(current.row());
				ui_->grb_focus_points->setTitle(tr("Focus Points at ROI #%1").arg(current_row));
			}
		});
	connect(rois_.get(), &roi_model::update_capture_info, this, &full_interface_gui::update_capture_info);
	connect(ui_->btnGotoROI, &QPushButton::clicked, this, [&] { goto_point(-1); });
	connect(ui_->btnAddROI, &QPushButton::clicked, this, &full_interface_gui::insert_point);
	connect(ui_->btnAddROI, &QPushButton::clicked, this, &full_interface_gui::verify_acquire_button);
	connect(ui_->btnDropROI, &QPushButton::clicked, this, &full_interface_gui::remove_point);
	connect(ui_->btnDropROI, &QPushButton::clicked, this, &full_interface_gui::verify_acquire_button);
	connect(ui_->btnSetROIXYZ, &QPushButton::clicked, this, &full_interface_gui::set_roi_xyz);
	connect(ui_->btnSetROIZ, &QPushButton::clicked, this, [&]
		{
			Q_UNUSED(set_roi_z(-1));
		});
	connect(ui_->btnSetAllROIZ, &QPushButton::clicked, this, &full_interface_gui::set_all_roi_z);
	connect(rois_.get(), &roi_model::row_col_changed, this, &full_interface_gui::fit_roi_in_view);
}

void full_interface_gui::verify_acquire_button()
{
	const auto valid_acquisition = is_valid_acquisition();
	ui_->btnDoIt->setEnabled(valid_acquisition);
}

void full_interface_gui::setup_acquire_buttons()
{
	camera_settings_to_combobox(ui_->cmbAcquireModes, D->cameras);
	connect(ui_->cmbAcquireModes, qOverload<int>(&QComboBox::currentIndexChanged), [&](const int idx)
		{
			const auto data_variant = ui_->cmbAcquireModes->itemData(idx);
			const auto data = data_variant.value<capture_mode>();
			const auto has_async = capture_mode_settings::info.at(data).async_io;
			ui_->tblROI->setColumnHidden(ITEM_SYNC_IDX, !has_async);
			update_capture_info();
		});
	connect(ui_->btnDoIt, &QPushButton::clicked, this, &full_interface_gui::do_scan);		//connect acquire button
	//set up channel switching modes
	for (const auto& item : full_interface_gui::roi_switching_settings)
	{
		ui_->cmb_switch_channel->addItem(item.second, QVariant::fromValue(item.first));
	}
}

void full_interface_gui::fill_column_from_selection() const
{
	const auto selection = ui_->tblROI->currentIndex();
	if (selection.isValid())
	{
		const auto data = selection.data();
		const auto column_idx = selection.column();
		rois_->fill_column(data, column_idx);
	}
}

int full_interface_gui::get_roi_index() const
{
	const auto selection = ui_->tblROI->currentIndex();
	return selection.row();
}

//when you click on a particular ROI it selects it in the table model and update_model_selection() is also called
void full_interface_gui::select_roi(const int item_idx) const
{
	const auto idx_to_select = ui_->tblROI->model()->index(item_idx, ITEM_X_IDX);
	ui_->tblROI->setCurrentIndex(idx_to_select);
	ui_->tblROI->selectRow(item_idx);
}

//called when you change rows in the roi table model, sets the corresponding xyz model 
void full_interface_gui::update_model_selection(const int selected_roi_idx) const
{
	auto* roi_item_ptr = rois_->data_view_.at(selected_roi_idx);
	ui_->xyzList->setModel(&roi_item_ptr->xyz_model);
	connect(ui_->xyzList->selectionModel(), &QItemSelectionModel::currentRowChanged, this, [&](const QModelIndex& current, const QModelIndex&)
		{
			if (current.isValid())
			{
				update_xyz_model_selection(current.row());
			}
		});
}

template <bool IsTop>
void full_interface_gui::peak_top_bottom(const bool checked) const
{
	scope_location_xyz old_position;
	const auto row = get_roi_index();
	if (row >= 0)
	{
		//const auto item = rois_->get_serializable_item(row);	

		const auto& item_ptr = rois_->data_view_.at(row);
		const auto x = item_ptr->get_x();
		const auto y = item_ptr->get_y();
		const auto roi_position = scope_location_xy(x, y);
		const auto z_middle = item_ptr->query_triangulator(roi_position);

		const auto new_z = z_middle + (IsTop ? (1) : (-1)) * item_ptr->get_pages() * item_ptr->get_page_step();
		const auto new_position = scope_location_xyz(x, y, new_z);
		old_position = scope_location_xyz(x, y, z_middle);
		D->scope->move_to(new_position, true);
	}
	if (!checked)
	{
		D->scope->move_to(old_position, true);
	}
}

void full_interface_gui::set_microscope_state(const microscope_state& state)
{
	const QPointF p(state.x, state.y);
	auto rect = pointer_->rect();
	rect.moveCenter(p);
	this->pointer_->setRect(rect);
};

void full_interface_gui::update_pointer_frame_size() const
{
	const auto aoi = ui_->wdg_light_path_holder->get_default_camera_config();
	const auto current_size = D->cameras.at(aoi.camera_idx)->get_sensor_size(aoi);
	const auto new_size = get_step_sizes(current_size);
	auto current_rect = pointer_->rect();
	current_rect.setWidth(new_size.x);
	current_rect.setHeight(new_size.y);
	pointer_->setRect(current_rect);
}

void full_interface_gui::goto_point(const int roi_row) const
{
	const auto row = roi_row < 0 ? get_roi_index() : roi_row;
	if (row >= 0)
	{
		const auto& item = rois_->data_view_.at(row);
		const auto position = scope_location_xy(item->get_x(), item->get_y());
		const auto destination_zee_position = rois_->data_view_.at(row)->query_triangulator(position);
		const auto pos = scope_location_xyz(position.x, position.y, destination_zee_position);
		D->scope->move_to(pos, true);				//paint event implicitly called //Need to call centerOn after the paint event has finished
		fit_roi_in_view();
		auto rect = pointer_->rect();
		rect.moveCenter(QPointF(pos.x, pos.y));
		pointer_->setRect(rect);
	}
}

void full_interface_gui::insert_point() const
{
	update_pointer_frame_size();		// I don't think I need this anymore
	const auto row = get_roi_index() + 1;
	rois_->insertRows(row, 1);
	select_roi(row);
}

void full_interface_gui::remove_point() const
{
	const auto row = get_roi_index();
	if (row >= 0)
	{
		rois_->removeRow(row);
	}
}

void full_interface_gui::insert_tomogram()
{
	const auto pixel_ratio = settings_file_.pixel_ratio;
	auto t = new tomogram_picker(pixel_ratio, this);
	connect(t, &tomogram_picker::add_tomogram, [&](const tomogram tomogram)
		{
			const auto selected_idx = [&]
			{
				const auto  row = get_roi_index();
				const auto selection = row >= 0 ? row + 1 : rois_->rowCount();
				return selection;
			}();
			rois_->insertRows(selected_idx, 1);
			setup_xyz_model_item(selected_idx);
			auto& xyz_model = rois_->data_view_.at(selected_idx)->xyz_model;
			xyz_model.fill_column(tomogram.z, XYZ_MODEL_ITEM_Z_IDX);
			select_roi(selected_idx);

			auto info = rois_->get_serializable_item(selected_idx);
			info.pages = tomogram.steps;
			info.page_step = tomogram.z_inc;   //change everything to double?
			rois_->set_serializable_item(info, selected_idx);
		});
	t->show();
}

std::vector<float> full_interface_gui::get_z_for_whole_roi(const int roi_idx) const
{
	return rois_->get_z_for_whole_roi(roi_idx);
}

void full_interface_gui::increment_zee_for_whole_roi(const float zee, const int roi_idx) const
{
	rois_->increment_zee_for_whole_roi(zee, roi_idx);
}

void full_interface_gui::set_zee_for_whole_roi(const float zee, const int roi_idx) const
{
	rois_->set_zee_for_whole_roi(zee, roi_idx);
}

void full_interface_gui::set_zee_for_whole_roi(const std::vector<float>& z_values, const int roi_idx) const
{
	rois_->set_zee_for_whole_roi(z_values, roi_idx);

}

bool full_interface_gui::set_roi_z(const int roi_row) const
{
	const auto row = roi_row < 0 ? get_roi_index() : roi_row;
	if (row >= 0)
	{
		const auto where_we_at = D->scope->get_state();
		const auto where_we_were = rois_->data_view_.at(row)->query_triangulator(where_we_at);
		if (std::isfinite(where_we_were))
		{
			const auto difference = where_we_at.z - where_we_were;
			increment_zee_for_whole_roi(difference, row);
			rois_->setData(rois_->index(row, ITEM_VERIFIED_IDX), true);
			return true;
		}

		QMessageBox msg_box;
		msg_box.setText(tr("Microscope must be on top of ROI %1").arg(rois_->data_view_.at(row)->get_id()));
		msg_box.exec();
	}
	return false;
}

void full_interface_gui::set_roi_xyz() const
{
	const auto row = get_roi_index();
	if (row >= 0)
	{
		const auto here = D->scope->get_state();
		rois_->setData(rois_->index(row, ITEM_X_IDX), here.x);
		rois_->setData(rois_->index(row, ITEM_Y_IDX), here.y);
		Q_UNUSED(set_roi_z());
		rois_->data_view_.at(row)->verify_focus_points(true);
	}
}

void full_interface_gui::set_all_roi_z() const
{
	const auto current_scope_z = D->scope->get_state().z;

	for (auto& roi : rois_->data_view_)
	{
		const auto row = roi->get_id();
		set_zee_for_whole_roi(current_scope_z, row);
	}
}


frame_size full_interface_gui::default_sensor_size_in_pixels() const
{
	const auto aoi = ui_->wdg_light_path_holder->get_default_camera_config();
	const auto current_size = D->cameras.at(aoi.camera_idx)->get_sensor_size(aoi);
	return current_size;
}


roi_item_serializable full_interface_gui::get_default_item() const
{
	fl_channel_index_list channel;
	for (auto i = 0; i < ui_->wdg_light_path_holder->get_number_channels(); i++)
	{
		channel.push_back(i);
	}

	const auto current_size = default_sensor_size_in_pixels();
	const auto step_sizes = get_step_sizes(current_size);

	const auto current_scope = D->scope->get_state(true);
	const roi_item_shared shared = { scope_delays(),channel,1,false,false, false };
	const roi_item_dimensions dimensions = { 1,1,0,step_sizes.x,step_sizes.y,step_sizes.z };
	return roi_item_serializable(shared, current_scope, dimensions, std::vector<scope_location_xyz>());
}


void full_interface_gui::setup_xyz_model_item(const int row) const
{
	auto& model = rois_->data_view_.at(row)->xyz_model;
	{
		const auto initial_zee = D->scope->get_state().z;
		model.fill_column(initial_zee, XYZ_MODEL_ITEM_Z_IDX);
	}
	ui_->xyzList->setModel(&model);
	ui_->xyzList->hideColumn(XYZ_MODEL_ITEM_ID_IDX);
	for (auto i = 0; i < model.columnCount(); i++)
	{
		auto* ptr = model.get_column_delegate(i, ui_->xyzList);
		if (ptr != nullptr)
		{
			ui_->xyzList->setItemDelegateForColumn(i, ptr);
		}
	}
	style_table_view(ui_->xyzList);

	connect(ui_->xyzList->model(), &QAbstractItemModel::dataChanged, this,
		[&](const QModelIndex&, const QModelIndex& bottom_right)
		{
			const auto is_a_validity_change = bottom_right.column() == XYZ_MODEL_ITEM_VALID_IDX;
			if (is_a_validity_change)
			{
				const auto index = rois_->index(bottom_right.row(), ITEM_VERIFIED_IDX);
				rois_->dataChanged(index, index);
			}
		});
	connect(ui_->xyzList->model(), &QAbstractItemModel::rowsInserted, this, [&](const QModelIndex& parent) {
		rois_->dataChanged(parent, parent);
		});
	connect(ui_->xyzList->model(), &QAbstractItemModel::rowsRemoved, this, [&](const QModelIndex& parent) {
		rois_->dataChanged(parent, parent);
		});
}


void full_interface_gui::setup_navigation_buttons()
{
	connect(ui_->btnXPlus, &QPushButton::clicked, [&] {step_grid(true, true); });
	connect(ui_->btnXmin, &QPushButton::clicked, [&] {step_grid(true, false); });
	connect(ui_->btnYPlus, &QPushButton::clicked, [&] {step_grid(false, false); });
	connect(ui_->btnYMin, &QPushButton::clicked, [&] {step_grid(false, true); });
}

void full_interface_gui::step_grid(const bool x_axis, const bool inc) const
{
	const auto roi_row = get_roi_index();
	if (roi_row >= 0) {
		const auto& item_ptr = rois_->data_view_.at(roi_row);
		auto center = item_ptr->get_grid_center();
		const auto width_x = item_ptr->get_column_step();
		const auto width_y = item_ptr->get_row_step();
		if (x_axis)
		{
			center.x += (inc ? 1 : -1) * width_x;
		}
		else
		{
			center.y += (inc ? 1 : -1) * width_y;
		}
		rois_->update_center(roi_row, center);
	}
}

//all this does is put the current roi in view/middle of screen
void full_interface_gui::fit_roi_in_view() const
{
	const auto roi_row = get_roi_index();
	if (roi_row >= 0) {
		const auto& item_ptr = rois_->data_view_.at(roi_row);
		auto rectangle = item_ptr->boundingRect();
		const auto center = rectangle.center();
		constexpr auto expanding_factor = 1.61;
		rectangle.setWidth(rectangle.width() * expanding_factor);
		rectangle.setHeight(rectangle.height() * expanding_factor);
		rectangle.moveCenter(center);
		if (rectangle.isValid())
		{
			ui_->gfxSurface->fitInView(rectangle, Qt::KeepAspectRatio);
		}
	}
}

void full_interface_gui::setup_channels() const
{
	connect(ui_->wdg_light_path_holder, &compact_light_path_holder::channel_removed, this, [&](const int channel_idx)
		{
			const auto channel_count = ui_->wdg_light_path_holder->get_number_channels();
			const auto& roi_items = rois_->data_view_;
			for (auto roi_index = 0; roi_index < roi_items.size(); ++roi_index)
			{
				auto channel_list = roi_items.at(roi_index)->channels;
				const auto max = *std::max_element(channel_list.begin(), channel_list.end());
				if (max >= channel_count)
				{
					fl_channel_index_list new_channels;
					for (auto i = 0; i < channel_list.size(); ++i)
					{
						if (channel_list.at(i) < channel_count)
						{
							new_channels.push_back(channel_list.at(i));
						}
					}
					if (new_channels.empty())
					{
						new_channels.push_back(0);
					}
					rois_->setData(rois_->index(roi_index, ITEM_CHANNEL_IDX), QVariant::fromValue(new_channels), Qt::EditRole);
				}
			}
		});
	connect(ui_->btnAddChannel, &QPushButton::clicked, ui_->wdg_light_path_holder, &compact_light_path_holder::add_one);
	connect(ui_->wdg_light_path_holder, &compact_light_path_holder::value_changed, this, &full_interface_gui::update_capture_info);
}

std::vector<channel_settings> full_interface_gui::get_channel_settings() const
{
	const auto light_paths = ui_->wdg_light_path_holder->get_compact_light_paths();
	std::vector<channel_settings> channels;
	channels.reserve(light_paths.size());
	for (const auto& light_path : light_paths)
	{
		const auto render = render_settings(render_modifications(), light_path, ml_remapper(), render_shifter());
		const slim_bg_settings no_bg_settings;
		compute_and_scope_settings::background_frame_ptr no_background_frame;
		const material_info no_material_info;
		const auto compute_settings = compute_and_scope_settings(light_path, render, light_path, light_path, light_path, no_bg_settings, no_background_frame, no_material_info);
		const auto live_gui = live_gui_settings(compute_settings, light_path.frames);
		//modify the last one
		auto channel = channel_settings(this->settings_file_, live_gui);
		channel.z_offset = light_path.zee_offset;
		channel.label_suffix = light_path.label_suffix;
		channel.fixup_channel();
		channel.assert_validity();
		channels.push_back(channel);
	}
	return channels;
}

full_interface_gui_settings full_interface_gui::get_saveable_settings() const
{
	const auto light_paths = ui_->wdg_light_path_holder->get_compact_light_paths();
	const auto cmb_acquire_modes = ui_->cmbAcquireModes->currentIndex();
	const auto full_iteration_times = ui_->iterationTimes->value();
	const auto interpolate_roi_enabled = ui_->btnUpdateInterp->isChecked();
	const auto interpolate_roi_global_enabled = ui_->btnUpdateInterpGlobal->isChecked();
	const auto meta_data = ui_->metadata->toPlainText().toStdString();
	const auto switch_channel_mode = ui_->cmb_switch_channel->currentData().value<channel_switching_order>();
	const auto filename_grouping = ui_->cmb_file_grouping->currentData().value<filename_grouping_mode>();
	return full_interface_gui_settings(light_paths, cmb_acquire_modes, full_iteration_times, interpolate_roi_enabled, interpolate_roi_global_enabled, meta_data, switch_channel_mode, filename_grouping);
}

void full_interface_gui::set_saveable_settings(const full_interface_gui_settings& settings) const
{
	ui_->wdg_light_path_holder->set_light_paths(settings.light_paths);
	ui_->cmbAcquireModes->setCurrentIndex(settings.cmb_acquire_modes);
	ui_->iterationTimes->setValue(settings.full_iteration_times);
	ui_->btnUpdateInterp->setChecked(settings.interpolate_roi_enabled);
	ui_->btnUpdateInterpGlobal->setChecked(settings.interpolate_roi_global_enabled);
	ui_->metadata->setPlainText(QString::fromStdString(settings.meta_data));
	{
		const auto idx = ui_->cmb_switch_channel->findData(QVariant::fromValue(settings.switch_channel_mode));
		ui_->cmb_switch_channel->setCurrentIndex(idx);
	}
	{
		const auto idx = ui_->cmb_file_grouping->findData(QVariant::fromValue(settings.filename_grouping));
		ui_->cmb_file_grouping->setCurrentIndex(idx);
	}

#if _DEBUG
	{
		const auto what_we_got = get_saveable_settings();
		if (!what_we_got.item_approx_equals(settings))
		{
			qli_gui_mismatch();
		}
	}
#endif
}

void full_interface_gui::closeEvent(QCloseEvent* event)
{
	if (is_valid_acquisition())
	{

		const auto reply = QMessageBox::question(this, "Close?", "Close interface and discard unsaved points?", QMessageBox::Yes | QMessageBox::No);
		if (reply != QMessageBox::Yes)
		{
			event->ignore();
			return;
		}
	}
	QMainWindow::closeEvent(event);
}


void full_interface_gui::set_live_gui_settings(const live_gui_settings& settings)
{
	const compact_light_path compact_light_path(settings, settings, settings, settings, settings.z_offset, settings.exposures_and_delays, settings, settings.label_suffix);
	ui_->wdg_light_path_holder->add_channel(compact_light_path);
}

void full_interface_gui::update_xyz_model_selection(const int selected_xyz_idx) const
{
	const auto roi_row = get_roi_index();
	if (roi_row >= 0)
	{
		const auto& xyz_model = rois_->data_view_.at(roi_row)->xyz_model;
		const auto location = xyz_model.get_focus_point_location(selected_xyz_idx);
		D->scope->move_to(location, true);
	}
}

int full_interface_gui::choose_next_roi() const
{
	const auto success = set_roi_z();
	const auto row = get_roi_index();
	auto next = -1;
	if (success)
	{
		rois_->data_view_.at(row)->verify_focus_points(true);
		next = rois_->find_unset_row(row);
		if (next >= 0)
		{
			select_roi(next);
			goto_point();
		}
	}
	return next;
}

void full_interface_gui::un_verify_focus_points(const int roi_row) const
{
	if (roi_row >= 0)
	{
		auto& xyz_model = rois_->data_view_.at(roi_row)->xyz_model;
		xyz_model.fill_column(false, XYZ_MODEL_ITEM_VALID_IDX);
		xyz_model.set_selection_color(true);	 //red (true) is default for selected ROI
	}
}

void full_interface_gui::focus_system_engaged(bool enable)
{
	//not implemented, but don't crash...	
}

void full_interface_gui::set_available_bytes(const size_t bytes)
{
	const auto as_mega = bytes / 1024 / 1024;
	ui_->prgStorage->setMaximum(as_mega);
}

void full_interface_gui::resizeEvent(QResizeEvent* event)
{
	QMainWindow::resizeEvent(event);
	fit_roi_in_view();
	const auto p = D->scope->get_state();
	ui_->gfxSurface->centerOn(QPointF(p.x, p.y));
}

void full_interface_gui::set_focus() const
{
	const auto index = ui_->xyzList->currentIndex();
	const auto roi_row = get_roi_index();
	if (index.isValid() && roi_row >= 0)
	{
		const auto xyz_row = index.row();
		auto& xyz_model = rois_->data_view_.at(roi_row)->xyz_model;
		const auto current_z = D->scope->get_state(true).z;
		xyz_model.setData(xyz_model.index(xyz_row, XYZ_MODEL_ITEM_Z_IDX), current_z);
		xyz_model.setData(xyz_model.index(xyz_row, XYZ_MODEL_ITEM_VALID_IDX), true);
	}
}

void full_interface_gui::center_grid() const
{
	const auto roi_row = get_roi_index();
	if (roi_row >= 0) {
		const auto position = D->scope->get_state();
		rois_->update_center(roi_row, static_cast<scope_location_xy>(position));
	}
}

//updates used bytes for all rois
void full_interface_gui::update_capture_info()
{
	if (!rois_)
	{
		return;
	}
	const auto mode = ui_->cmbAcquireModes->currentData().value<capture_mode>();
	const auto regular_channels = get_channel_settings();
	size_t total_size = 0;
	for (auto i = 0; i < rois_->rowCount(); ++i)
	{
		const auto serializable_item = rois_->get_serializable_item(i);
		for (const auto& channel_idx : serializable_item.channels)
		{
			const auto& channel = regular_channels.at(channel_idx);
			const auto regular_z_steps = 2 * serializable_item.pages + 1;
			total_size += channel.bytes_per_capture_item_on_disk() * regular_z_steps * serializable_item.columns * serializable_item.rows * serializable_item.repeats;
		}
	}
	const size_t times = ui_->iterationTimes->value();
	const auto bytes = total_size * times;
	const auto as_mega = bytes / 1024 / 1024;
	const auto max_mega = static_cast<size_t>(ui_->prgStorage->maximum());
	const auto is_max_ed_out = as_mega >= max_mega;
	ui_->prgStorage->setValue(std::min(as_mega, max_mega));
	QStringList list;
	list << "KB" << "MB" << "GB" << "TB";
	QStringListIterator i(list);
	QString unit("Bytes");
	auto division_count = 0;
	auto shift_me = bytes;
	while (shift_me >= 1024.0 && i.hasNext())
	{
		unit = i.next();
		shift_me /= 1024.0;
		division_count = division_count + 1;
	}
	const auto fraction = powf(1024, division_count);
	const auto value = bytes / fraction;
	auto str = QString().setNum(value, 'f', 2) + " " + unit;
	if (is_max_ed_out)
	{
		str.append(" (!)");
	}
	ui_->prgStorage->setFormat(str);
}

void full_interface_gui::set_six_well_plate() const
{
	QMessageBox message;
	message.setText("Move microscope to first well");
	message.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
	message.setDefaultButton(QMessageBox::Ok);
	const auto ret = message.exec();
	if (ret == QMessageBox::Cancel)
	{
		return;
	}
	// For 6 well
	//const float length = 127.2;  // [mm]
	//const float width = 85.4;    // [mm]
	const float step_x = 40;     // [mm]
	const float step_y = 38;     // [mm]
	const auto roi_step_x = step_x * 1000;  //40 mm -> microns 
	const auto roi_step_y = step_y * 1000;  //38 mm -> microns
	//const int cols = floor(length / step_x);
	//const int rows = floor(width / step_y);
	const auto cols = 3;
	const auto rows = 2;

	//For 96 well
	// step_x = step_y = 9;   //[mm]
	//roi_step_x = roi_step_y = step_x * 1000;

	const auto original_position = D->scope->get_state();
	const auto n = snake_iterator::count(cols, rows);
	for (auto i = 0; i < n; i++)
	{
		const auto col_index = snake_iterator::iterate(i, cols).column;
		const auto row_index = snake_iterator::iterate(i, cols).row;
		// const auto x_new = col_index * roi_step_x + original_position.x;
		// const auto y_new = row_index * roi_step_y + original_position.y;
		const auto x_new = original_position.x - col_index * roi_step_x;
		const auto y_new = original_position.y - row_index * roi_step_y;

		insert_point();
		const auto roi_index = get_roi_index();
		rois_->data_view_.at(roi_index)->set_x(x_new);
		rois_->data_view_.at(roi_index)->set_y(y_new);
	}
}

void full_interface_gui::wrangle_convex_hull() const
{
	const auto roi_row = get_roi_index();
	if (roi_row >= 0) {
		const auto& item_ptr = rois_->data_view_.at(roi_row);
		const auto& xyz_model = item_ptr->xyz_model;

		const auto hull = [&]
		{
			auto left = std::numeric_limits<float>::infinity();
			auto right = -std::numeric_limits<float>::infinity();
			auto top = -std::numeric_limits<float>::infinity();
			auto bottom = std::numeric_limits<float>::infinity();

			const auto items = xyz_model.rowCount();

			//special case
			if (items == xyz_focus_points_model::min_rows + 1)
			{
				const auto center = item_ptr->get_grid_center();
				const auto steps = item_ptr->get_grid_steps();
				left = center.x - (steps.x_step * steps.x_steps) / 2;
				right = center.x + (steps.x_step * steps.x_steps) / 2;
				top = center.y + (steps.y_step * steps.y_steps) / 2;
				bottom = center.y - (steps.y_step * steps.y_steps) / 2;
			}
			const auto rect_original = QRectF(left, bottom, right - left, top - bottom);

			for (auto idx = xyz_focus_points_model::min_rows; idx < items; ++idx)
			{
				const auto idx_x = xyz_model.index(idx, XYZ_MODEL_ITEM_X_IDX);
				auto x = xyz_model.data(idx_x, Qt::DisplayRole).toFloat();
				left = std::min(left, x);
				right = std::max(right, x);
				const auto idx_y = xyz_model.index(idx, XYZ_MODEL_ITEM_Y_IDX);
				auto y = xyz_model.data(idx_y, Qt::DisplayRole).toFloat();
				bottom = std::min(bottom, y);
				top = std::max(top, y);
			}

			const auto rect = QRectF(left, bottom, right - left, top - bottom);
			const auto rect_changed = rect != rect_original;
			return rect_changed ? rect : QRectF();
		}();
		if (!hull.isValid())
		{
			return;
		}
		const auto center = hull.center();
		const auto gs = item_ptr->get_grid_steps();
		const auto x_step = gs.x_step;
		const int x_steps = ceil(hull.width() / x_step);
		const auto y_step = gs.y_step;
		const int y_steps = ceil(hull.height() / y_step);
		assert(x_steps > 0);
		assert(y_steps > 0);

		item_ptr->set_columns(x_steps);
		item_ptr->set_rows(y_steps);
		rois_->update_center(roi_row, scope_location_xy(center.x(), center.y()));
	}
}


void full_interface_gui::paintEvent(QPaintEvent* event)
{
	auto new_zee_max = -std::numeric_limits<float>::infinity();
	auto new_zee_min = std::numeric_limits<float>::infinity();

	for (auto& roi : rois_->data_view_)
	{
		const auto focus_points = roi->get_roi_item_serializable().focus_points;
		for (const auto& point : focus_points)
		{
			if (isfinite(point.z))
			{
				new_zee_min = std::min(new_zee_min, point.z);
				new_zee_max = std::max(new_zee_max, point.z);
			}
		}
	}
	roi_item::zee_max_global = new_zee_max;
	roi_item::zee_min_global = new_zee_min;

	QWidget::paintEvent(event);		//Question: Still works without this, whats the purpose?
}

//Secret keyboard command to perform six well plate tool
void full_interface_gui::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_6 && (event->modifiers() & Qt::AltModifier))
	{
		set_six_well_plate();
	}
	QWidget::keyPressEvent(event);
}

void full_interface_gui::set_file_settings(const settings_file& settings_file)
{
	this->settings_file_ = settings_file;
	emit pixel_ratio_changed(settings_file.pixel_ratio);
}

std::array<QString, 3> full_interface_gui::scan_state_labels = { { "", "Configuring...", "Acquiring..." } };

void full_interface_gui::do_scan()
{
	ui_->btnDoIt->setText(scan_state_labels.at(1));
	D->route.clear();
	D->route.output_dir = get_dir().toStdString();
	const auto success = wrangle_scan();
	if (success)
	{
		const auto text = ui_->btnDoIt->text();
		const auto mode = ui_->cmbAcquireModes->currentData().value<capture_mode>();
		start_acquisition(mode);
	}
	else
	{
		// change the button back to "Acquire"
		ui_->btnDoIt->setText(scan_state_labels.at(0));
	}
}


QRectF full_interface_gui::get_camera_size_in_stage_coordinates(const camera_config& settings, const pixel_dimensions& dimensions)
{
	const auto frame_size = D->cameras.at(settings.camera_idx)->get_sensor_size(settings);
	const auto scale_factor = dimensions.pixel_ratio;
	const auto real_frame_size = QRectF(0, 0, frame_size.width / scale_factor, frame_size.height / scale_factor);
	return real_frame_size;
}

scope_location_xyz full_interface_gui::get_step_sizes(const frame_size& frame) const noexcept
{
	const auto scale = settings_file_.pixel_ratio;
	const auto overlap = settings_file_.stage_overlap;
	const auto x_step_um = overlap * frame.width / scale;
	const auto y_step_um = overlap * frame.height / scale;
	const auto z_step_um = 1.0f / std::min(scale, scale);
	const auto step_sizes = scope_location_xyz(overlap * x_step_um, overlap * y_step_um, z_step_um);
	return step_sizes;
}

void full_interface_gui::gui_enable(const bool enable)
{
	const auto do_it_label_idx = enable ? 0 : 2;
	ui_->btnDoIt->setText(scan_state_labels.at(do_it_label_idx));
	ui_->btnDoIt->setEnabled(enable);
	//add more items here during usability testing
}